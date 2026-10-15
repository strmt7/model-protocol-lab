"""Cycle 7: property-based tests for the divergence classifier.

Hypothesis properties over ``model_protocol_lab.diffbackends`` --
``diff_backends`` (the divergence classifier) and
``divergence_section_data`` (the report-section aggregator):

- verdict consistency: ``verdict == "agree"`` iff ``divergences == []``;
- divergence well-formedness: every divergence dict carries a class
  from the 8-class vocabulary, a stop_text from the deduped stop texts,
  and a non-empty detail string; section-builder findings additionally
  pin ``severity`` to the ``DIVERGENCE_SEVERITY`` label for the class;
- severity totality: ``DIVERGENCE_SEVERITY`` keys are exactly the 8
  divergence classes;
- section aggregation invariants: section ``verdict == "agree"`` iff
  every per-fixture verdict is ``"agree"``, matrix counts are
  non-negative ints, and ``pairs_verdict`` matches the documented
  one-line summary shape;
- id-length bound: the documented output cap (``MAX_OUTPUT_IDS``) is pinned
  (over-long input rejected with the fixed ``output_ids_too_long`` code);
- empty input: empty ``output_ids`` never crash and yield a valid
  verdict.

All backends are fakes decoding through a fixed per-token text table;
no torch, no network, no model weights. Strategies stay small and fast,
and every hypothesis test is derandomized for deterministic CI.
"""

import unittest
from typing import Any, get_args

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from model_protocol_lab.diffbackends import (
    DIVERGENCE_SEVERITY,
    MAX_OUTPUT_IDS,
    DivergenceClass,
    _checked_ids,
    diff_backends,
    divergence_section_data,
)

FUZZ = settings(
    derandomize=True,
    max_examples=40,
    # Same rationale as tests/test_fuzz_boundary.py: differing_executors
    # fires on in-process re-runs (a harness artifact), not on flakiness;
    # these tests are deterministic, so the check is suppressed.
    suppress_health_check=[HealthCheck.differing_executors],
)

#: The 8-class divergence vocabulary, pinned from the Literal.
DIVERGENCE_CLASSES = frozenset(get_args(DivergenceClass))

#: Valid verdicts produced by the classifier and the section builder.
VERDICTS = ("agree", "diverge")

#: Fixed per-token text table for the fake backend. Single chars keep
#: decodes short; "ST" lets stop texts span token boundaries; the
#: non-ASCII entries exercise unicode stop matching.
_TOKEN_TEXTS = {1: "a", 2: "b", 3: "ST", 4: " ", 5: "\u00e9", 6: "\u4e2d"}

#: Small alphabet for stop texts: single chars (""-adjacent), ASCII
#: multi-char material, whitespace, and unicode (incl. non-BMP).
STOP_ALPHABET = "abST \u00e9\u4e2d\U0001f600"

#: Stop texts drawn from the fake backend's own token texts, so they can
#: actually fire and the classifier's diverge paths get exercised; mixed
#: with fully arbitrary texts from the small alphabet.
_firing_stop_texts_strategy = st.lists(
    st.sampled_from(sorted(set(_TOKEN_TEXTS.values()))),
    min_size=1,
    max_size=2,
    unique=True,
)
_random_stop_texts_strategy = st.lists(
    st.text(alphabet=STOP_ALPHABET, min_size=1, max_size=6),
    min_size=1,
    max_size=3,
    unique=True,
)
stop_texts_strategy = st.one_of(_firing_stop_texts_strategy, _random_stop_texts_strategy)
output_ids_strategy = st.lists(st.sampled_from(sorted(_TOKEN_TEXTS)), max_size=6).map(tuple)
min_tokens_strategy = st.integers(0, 3)


class PropertyTableBackend:
    """Fake backend decoding token ids through a fixed per-token text table.

    Mirrors the cycle-6/7 diff-backend fakes: whole-sequence decodes
    concatenate per-token texts.
    """

    fingerprint = "cycle7-properties-table-backend-not-a-model"

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(_TOKEN_TEXTS[token] for token in token_ids)


class DivergencePropertyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = PropertyTableBackend()

    def _classify(self, **kwargs: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "prompt_ids": (),
            "stop_texts": ("STOP",),
            "backend": self.backend,
        }
        base.update(kwargs)
        return diff_backends(**base)

    # -- verdict consistency -------------------------------------------

    @FUZZ
    @given(
        stop_texts=stop_texts_strategy,
        output_ids=output_ids_strategy,
        min_tokens=min_tokens_strategy,
    )
    def test_verdict_consistency(
        self,
        stop_texts: list[str],
        output_ids: tuple[int, ...],
        min_tokens: int,
    ) -> None:
        """verdict == "agree" iff divergences == [].

        For arbitrary stop texts (small alphabet incl. single-char,
        whitespace and unicode), arbitrary short output ids, and
        min_tokens in [0, 3].
        """
        result = self._classify(output_ids=output_ids, stop_texts=stop_texts, min_tokens=min_tokens)
        self.assertIn(result["verdict"], VERDICTS)
        self.assertEqual(result["verdict"] == "agree", result["divergences"] == [])

    # -- divergence well-formedness ------------------------------------

    @FUZZ
    @given(
        stop_texts=stop_texts_strategy,
        output_ids=output_ids_strategy,
        min_tokens=min_tokens_strategy,
    )
    def test_divergence_wellformedness(
        self,
        stop_texts: list[str],
        output_ids: tuple[int, ...],
        min_tokens: int,
    ) -> None:
        """Every divergence dict is well-formed.

        class in the 8-class vocabulary, stop_text in the deduped stop
        texts, detail a non-empty str.
        """
        result = self._classify(output_ids=output_ids, stop_texts=stop_texts, min_tokens=min_tokens)
        deduped = result["stop_texts"]
        for finding in result["divergences"]:
            self.assertIn(finding["class"], DIVERGENCE_CLASSES)
            self.assertIn(finding["stop_text"], deduped)
            self.assertIsInstance(finding["detail"], str)
            self.assertTrue(finding["detail"])

    # -- severity totality ---------------------------------------------

    def test_severity_totality(self) -> None:
        """DIVERGENCE_SEVERITY keys are exactly the 8 divergence classes."""
        self.assertEqual(set(DIVERGENCE_SEVERITY), DIVERGENCE_CLASSES)
        self.assertEqual(len(DIVERGENCE_SEVERITY), 8)

    # -- section aggregation invariants --------------------------------

    @FUZZ
    @given(
        data=st.data(),
        stop_texts=stop_texts_strategy,
        min_tokens=st.integers(0, 2),
    )
    def test_section_aggregation_invariants(
        self, data: Any, stop_texts: list[str], min_tokens: int
    ) -> None:
        """divergence_section_data aggregates soundly over 1-4 observations.

        section verdict == "agree" iff every per-fixture verdict is
        "agree"; matrix counts are non-negative ints; pairs_verdict
        matches the documented one-line summary; section findings are
        well-formed with the severity pinned to DIVERGENCE_SEVERITY.
        """
        observations = [
            {
                "fixture": f"prop-fx-{index}",
                "input_ids": (),
                "output_ids": data.draw(output_ids_strategy),
            }
            for index in range(data.draw(st.integers(1, 4)))
        ]
        section = divergence_section_data(
            observations=observations,
            stop_texts=stop_texts,
            eos_id=None,
            min_tokens=min_tokens,
            backend=self.backend,
        )
        # Every synthetic observation validates, so the section exists.
        self.assertIsNotNone(section)
        assert section is not None
        self.assertIn(section["verdict"], VERDICTS)
        per_fixture = section["per_fixture"]
        self.assertEqual(len(per_fixture), len(observations))
        self.assertEqual(
            section["verdict"] == "agree",
            all(entry["verdict"] == "agree" for entry in per_fixture),
        )
        for kind, row in section["matrix"].items():
            self.assertIn(kind, DIVERGENCE_CLASSES)
            for count in row.values():
                self.assertIsInstance(count, int)
                self.assertGreaterEqual(count, 0)
        self.assertRegex(
            section["pairs_verdict"],
            r"HF vs vLLM vs SGLang stop semantics: \d+/\d+ agree; \d+ divergences?",
        )
        for entry in per_fixture:
            self.assertIn(entry["verdict"], VERDICTS)
            for finding in entry["divergences"]:
                self.assertIn(finding["class"], DIVERGENCE_CLASSES)
                self.assertIn(finding["stop_text"], section["stop_texts"])
                self.assertIsInstance(finding["detail"], str)
                self.assertTrue(finding["detail"])
                # "severity" is the mapped label (a DIVERGENCE_SEVERITY
                # value), pinned exactly to the class's entry.
                self.assertEqual(finding["severity"], DIVERGENCE_SEVERITY[finding["class"]])

    # -- id-length bound -------------------------------------------------

    def test_id_length_bound(self) -> None:
        """The documented output cap: over-long input is rejected.

        The tuple is constructed directly (cheap); the rejection carries
        the fixed ``output_ids_too_long`` code, and the cap itself still
        passes the validator. (The old 131072 cap was quadratic-DoS sized;
        see ``MAX_OUTPUT_IDS``.)
        """
        with self.assertRaises(ValueError) as ctx:
            self._classify(output_ids=tuple([1] * (MAX_OUTPUT_IDS + 1)))
        self.assertEqual(str(ctx.exception), "output_ids_too_long")
        at_cap = tuple([1] * MAX_OUTPUT_IDS)
        self.assertEqual(_checked_ids(at_cap, "output_ids", cap=MAX_OUTPUT_IDS), at_cap)

    # -- empty input -----------------------------------------------------

    @FUZZ
    @given(stop_texts=stop_texts_strategy)
    def test_empty_output_ids_never_crash(self, stop_texts: list[str]) -> None:
        """Empty output_ids never crash; the verdict is a valid string."""
        result = self._classify(output_ids=(), stop_texts=stop_texts)
        self.assertIn(result["verdict"], VERDICTS)
        self.assertEqual(result["output_tokens"], 0)
        self.assertEqual(result["verdict"] == "agree", result["divergences"] == [])


if __name__ == "__main__":
    unittest.main()
