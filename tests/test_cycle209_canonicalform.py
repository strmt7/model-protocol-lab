"""Cycle 209: canonical-form ladder (``run --canonical-form``).

Generated fact-recall cases rendered under NFC and NFD per
candidate. The load-bearing pins:

* The scorer pin (red-team A-F1 / B-CF1, executed through the
  real ``core.evaluate``): an accented expected value flips a
  zero-defect echo model's verdict on scorer bytes alone, so the
  generator's values are ASCII by construction and a test pins
  that no generated value ever carries an accent.
* The headline population is id-divergent paired cases only;
  ids-identical cases are the inapplicable share and same-ids
  flips are a determinism finding, never fragility.
* The decision floors (6 paired, 2 discordant) and the winner
  headline are total functions, including the abstention rule:
  selection-vs-abstention is never a winner change.
* The diff validator's mirror verifies the TREATMENT (both
  arms' content digests regenerated from the seed), not just
  token counts.
"""

import copy
import re
import threading
import unicodedata
import unittest

from model_protocol_lab import canonicalform as cf
from model_protocol_lab import diffreports, experiments
from model_protocol_lab import reporting as reporting_mod
from model_protocol_lab.backends import Generation
from model_protocol_lab.core import Fixture, Format, Message, evaluate

_CHATML = Format(
    name="chatml",
    bos="",
    user_prefix="u:",
    assistant_prefix="a:",
    turn_end="</s>",
)
_LLAMA = Format(
    name="llama",
    bos="",
    user_prefix="L:",
    assistant_prefix="A:",
    turn_end="</s>",
)


class _WordBackend:
    """Word-token scripted backend with switchable behaviours.

    Modes: ``perfect`` (answer the planted code wherever it
    sits), ``nfd_fail`` (fail the NFD arm's first N generations,
    N = ``fail_count``), ``nfc_fail`` (same, NFC arm),
    ``both_fail`` (fail the first N generations of each arm),
    ``explode`` (generate raises). Arm detection is by form:
    NFC text round-trips through NFC unchanged, NFD text does
    not. ``known`` maps a topic to a memorized value answered
    even with no payload in view (unused here; kept for parity
    with the placement harness).
    """

    fingerprint = "sha256:test-canonicalform"

    def __init__(self, mode: str = "perfect", fail_count: int = 0) -> None:
        self.mode = mode
        self.fail_count = fail_count
        self.calls = 0
        self.nfc_fails = 0
        self.nfd_fails = 0
        self.vocab: dict[str, int] = {}
        self.rev: dict[int, str] = {}

    def encode(self, text):
        """Encode."""
        ids = []
        for token in text.replace("\n", " \n ").split(" "):
            if not token:
                continue
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 1
                self.rev[self.vocab[token]] = token
            ids.append(self.vocab[token])
        return tuple(ids)

    def decode(self, ids):
        """Decode."""
        return " ".join(self.rev.get(int(i), "?") for i in ids)

    @staticmethod
    def _is_nfd(text: str) -> bool:
        """Is nfd."""
        return unicodedata.normalize("NFC", text) != text

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        """Generate."""
        self.calls += 1
        if self.mode == "explode":
            raise RuntimeError("boom")
        text = self.decode(list(input_ids))
        found = re.findall(r"inspection code is (\S+?)\.", text)
        answer = found[-1] if found else "unknown"
        is_nfd = self._is_nfd(text)
        if self.mode == "nfd_fail" and is_nfd and self.nfd_fails < self.fail_count:
            self.nfd_fails += 1
            answer = "wrong-0000"
        elif self.mode == "nfc_fail" and not is_nfd and self.nfc_fails < self.fail_count:
            self.nfc_fails += 1
            answer = "wrong-0000"
        elif self.mode == "both_fail":
            if is_nfd and self.nfd_fails < self.fail_count:
                self.nfd_fails += 1
                answer = "wrong-0000"
            elif not is_nfd and self.nfc_fails < self.fail_count:
                self.nfc_fails += 1
                answer = "wrong-0000"
        boundary = self.encode("endmark")[0]
        out = list(self.encode(answer)) + [boundary]
        out = (out + [boundary] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


class _NormalizingBackend(_WordBackend):
    """Word backend whose encode NFC-normalizes first, so the
    NFC and NFD arms produce identical ids on every case: the
    instrument-inapplicable shape (SentencePiece-family
    checkpoint with a normalizer in the pipeline).
    """

    fingerprint = "sha256:test-canonicalform-normalizing"

    def encode(self, text):
        """Encode."""
        return super().encode(unicodedata.normalize("NFC", text))


class _AlternatingBackend(_NormalizingBackend):
    """Normalizing encode (ids identical) with a generate that
    fails every odd call: same-ids verdict flips by backend
    nondeterminism, which the aggregates must record as a
    determinism finding, never as headline discordance. The
    alternation counter is separate from the base call count,
    which the parent generate also increments.
    """

    fingerprint = "sha256:test-canonicalform-alternating"

    def __init__(self) -> None:
        super().__init__()
        self.alt_calls = 0

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        """Generate."""
        self.alt_calls += 1
        if self.alt_calls % 2 == 1:
            text = self.decode(list(input_ids))
            found = re.findall(r"inspection code is (\S+?)\.", text)
            answer = "wrong-0000" if found else "unknown"
            boundary = self.encode("endmark")[0]
            out = list(self.encode(answer)) + [boundary]
            out = (out + [boundary] * max_new_tokens)[:max_new_tokens]
            return Generation(token_ids=tuple(out), reason="limit")
        return _WordBackend.generate(
            self, input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel
        )


class _BalancedBackend(_WordBackend):
    """Fails the NFC arm on the first two cases and the NFD
    arm on the next two: balanced discordance (2 NFC-only, 2
    NFD-only). A backend that failed both arms on the SAME
    cases would produce concordant "neither" cases, not
    discordance -- the arm windows must not overlap.
    """

    fingerprint = "sha256:test-canonicalform-balanced"

    def __init__(self) -> None:
        super().__init__()
        self.nfc_seen = 0
        self.nfd_seen = 0

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        """Generate."""
        text = self.decode(list(input_ids))
        is_nfd = self._is_nfd(text)
        if is_nfd:
            self.nfd_seen += 1
            fail = 3 <= self.nfd_seen < 5
        else:
            self.nfc_seen += 1
            fail = self.nfc_seen <= 2
        if fail:
            found = re.findall(r"inspection code is (\S+?)\.", text)
            answer = "wrong-0000" if found else "unknown"
            boundary = self.encode("endmark")[0]
            out = list(self.encode(answer)) + [boundary]
            out = (out + [boundary] * max_new_tokens)[:max_new_tokens]
            return Generation(token_ids=tuple(out), reason="limit")
        return super().generate(input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel)

    # Fails the first generation of a configured (format, arm)
    # pair, so two candidates' pass sets can be made to swap
    # across arms (the winner-change shape) or to differ on one
    # arm only (the abstention-asymmetric shape). Format
    # detection is by the format's user prefix in the render.


class _PerFormatBackend(_WordBackend):
    """Fails the first generation of a configured (format, arm)
    pair, so two candidates' pass sets can be made to swap
    across arms (the winner-change shape) or to differ on one
    arm only (the abstention-asymmetric shape). Format
    detection is by the format's user prefix in the render.
    """

    fingerprint = "sha256:test-canonicalform-performat"

    def __init__(self, failures: dict[tuple[str, str], int]) -> None:
        super().__init__()
        self.failures = dict(failures)
        self.seen: dict[tuple[str, str], int] = {}

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        """Generate."""
        text = self.decode(list(input_ids))
        fmt = "chatml" if "u:" in text else "llama"
        arm = "nfd" if self._is_nfd(text) else "nfc"
        key = (fmt, arm)
        used = self.seen.get(key, 0)
        self.seen[key] = used + 1
        if used < self.failures.get(key, 0):
            found = re.findall(r"inspection code is (\S+?)\.", text)
            answer = "wrong-0000" if found else "unknown"
            boundary = self.encode("endmark")[0]
            out = list(self.encode(answer)) + [boundary]
            out = (out + [boundary] * max_new_tokens)[:max_new_tokens]
            return Generation(token_ids=tuple(out), reason="limit")
        return super().generate(input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel)


def _payload(backend=None, formats=(_CHATML,), seed=0, **stage_kwargs):
    """Payload."""
    backend = backend if backend is not None else _WordBackend()
    stage_kwargs.setdefault("max_context", 4096)
    stage = experiments.canonicalform_stage(
        backend,
        formats,
        boundary=tuple(backend.encode("endmark")),
        seed=seed,
        **stage_kwargs,
    )
    return experiments.canonicalform_analysis(stage, tuple(f.name for f in formats))


class GeneratorTests(unittest.TestCase):
    """Generator tests."""

    def test_case_count_and_shape(self):
        """Test case count and shape."""
        cases = cf.generate_cases(0)
        self.assertEqual(len(cases), cf.CASE_COUNT)
        self.assertEqual(cf.CASE_COUNT, 24)
        for case in cases:
            self.assertTrue(case["fixture_id"].startswith("canonicalform-"))
            self.assertEqual(case["expected"], case["value"])
            self.assertEqual(len(case["sentence_indices"]), 2)
            self.assertNotEqual(case["sentence_indices"][0], case["sentence_indices"][1])

    def test_values_ascii_distinct_and_deterministic(self):
        """Test values ascii distinct and deterministic."""
        first = cf.generate_cases(0)
        again = cf.generate_cases(0)
        self.assertEqual(first, again)
        values = [c["value"] for c in first]
        self.assertTrue(all(v.isascii() for v in values))
        self.assertEqual(len(set(values)), len(values))
        self.assertNotEqual(cf.generate_cases(0), cf.generate_cases(1))

    def test_authored_content_nfc_and_divergent(self):
        """Test authored content nfc and divergent."""
        for case in cf.generate_cases(0):
            content = case["content"]
            self.assertEqual(unicodedata.normalize("NFC", content), content)
            self.assertNotEqual(
                unicodedata.normalize("NFC", content),
                unicodedata.normalize("NFD", content),
            )
            self.assertEqual(cf.form_content(case, "nfc"), content)
            self.assertEqual(cf.form_content(case, "nfd"), unicodedata.normalize("NFD", content))
            self.assertNotEqual(cf.form_content(case, "nfc"), cf.form_content(case, "nfd"))

    def test_topics_are_nfc_divergent_and_singleton_free(self):
        """Test topics are nfc divergent and singleton free."""
        self.assertEqual(len(cf.TOPICS), 24)
        for topic in cf.TOPICS:
            self.assertEqual(unicodedata.normalize("NFC", topic), topic)
            self.assertNotEqual(
                unicodedata.normalize("NFC", topic), unicodedata.normalize("NFD", topic)
            )
            for char in topic:
                self.assertFalse(cf._is_singleton(char), f"singleton in topic {topic!r}")
                self.assertFalse(cf._is_hangul_syllable(char))

    def test_pool_sentences_divergent_and_clean(self):
        """Test pool sentences divergent and clean."""
        for sentence in cf.NEUTRAL_POOL:
            self.assertEqual(unicodedata.normalize("NFC", sentence), sentence)
            self.assertNotEqual(
                unicodedata.normalize("NFC", sentence),
                unicodedata.normalize("NFD", sentence),
            )
            self.assertFalse(any(ch.isdigit() for ch in sentence))
            # A neutral sentence must never plant a competing
            # code: the extractor takes the last "inspection
            # code is X." match, and only the fact template
            # contains that pattern.
            self.assertNotIn("inspection code", sentence)

    def test_effective_alphabet_reported_by_mapping(self):
        """Test effective alphabet reported by mapping."""
        # The alphabet is defined by the mapping predicate, not
        # by a block count: the frozen tables' decomposable
        # characters are exactly these twelve, and the known
        # non-decomposing Latin-1 letters are exactly the nine
        # the red team measured (absent by construction).
        for char in "ÆÐØÞßæðøþ":
            self.assertEqual(unicodedata.decomposition(char), "")
        material = "".join(cf.TOPICS) + "".join(cf.NEUTRAL_POOL)
        decomposable = {
            ch
            for ch in material
            if unicodedata.normalize("NFC", ch) != unicodedata.normalize("NFD", ch)
        }
        self.assertEqual(decomposable, set("âçèéêëïñôöûü"))

    def test_validation_rejects_accented_value_singleton_and_hangul(self):
        """Test validation rejects accented value singleton and hangul."""
        content = cf.generate_cases(0)[0]["content"]
        with self.assertRaises(ValueError):
            cf._validate_case_text(content, "café-1234")
        with self.assertRaises(ValueError):
            cf._validate_case_text("The \u212b café code is elm-1.", "elm-1")
        with self.assertRaises(ValueError):
            cf._validate_case_text("The \ud55c café code is elm-1.", "elm-1")
        with self.assertRaises(ValueError):
            cf._validate_case_text("The plain code is elm-1.", "elm-1")

    def test_validation_guard_messages_pin_which_guard_fired(self):
        """Test validation guard messages pin which guard fired."""
        # Same outcome class (ValueError) for every guard, so
        # the pin must assert the message: a message-blind
        # test passes vacuously when guards are reordered
        # (cycle-172 lesson). The singleton guard must fire
        # for U+2126 even though that character is also
        # non-NFC: the singleton check runs first by design.
        with self.assertRaisesRegex(ValueError, "singleton"):
            cf._validate_case_text("The \u2126 café code is elm-1.", "elm-1")
        with self.assertRaisesRegex(ValueError, "Hangul"):
            cf._validate_case_text("The \ud55c café code is elm-1.", "elm-1")
        with self.assertRaisesRegex(ValueError, "pure ASCII"):
            cf._validate_case_text(cf.generate_cases(0)[0]["content"], "café-1234")
        with self.assertRaisesRegex(ValueError, "divergent"):
            cf._validate_case_text("The plain code is elm-1.", "elm-1")
        with self.assertRaisesRegex(ValueError, "authored in NFC"):
            cf._validate_case_text(
                unicodedata.normalize("NFD", cf.generate_cases(0)[0]["content"]),
                "elm-1",
            )
        with self.assertRaisesRegex(ValueError, "nonempty"):
            cf._validate_case_text("", "elm-1")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            cf._validate_case_text(123, "elm-1")  # type: ignore[arg-type]

    def test_ground_and_protocol_digests(self):
        """Test ground and protocol digests."""
        self.assertEqual(cf.ground_sha256(0), cf.ground_sha256(0))
        self.assertNotEqual(cf.ground_sha256(0), cf.ground_sha256(1))
        protocol = cf.protocol_object(0, "sha256:test")
        self.assertEqual(protocol["arm_keys"], ["nfc", "nfd"])
        self.assertEqual(protocol["baseline_arm"], "nfc")
        self.assertEqual(protocol["min_paired_evaluable"], 6)
        self.assertEqual(protocol["min_discordant"], 2)
        self.assertEqual(protocol["generation_budget"], 2 * 24 * 24)
        self.assertEqual(
            cf.protocol_fingerprint(0, "sha256:test"),
            cf.protocol_fingerprint(0, "sha256:test"),
        )
        self.assertNotEqual(
            cf.protocol_fingerprint(0, "sha256:test"),
            cf.protocol_fingerprint(0, "sha256:other"),
        )

    def test_unknown_arm_rejected(self):
        """Test unknown arm rejected."""
        case = cf.generate_cases(0)[0]
        with self.assertRaises(ValueError):
            cf.form_content(case, "nfkc")


class ScorerPinTests(unittest.TestCase):
    """The red-team F1 mechanism, executed through the real
    ``core.evaluate``: a zero-defect echo model flips its verdict
    on scorer bytes alone when the expected value carries
    accents, and never flips when values are ASCII. This is why
    the generator's ASCII-value rule exists; the test pins the
    mechanism, and GeneratorTests pins the rule.
    """

    def _echo_evaluate(self, topic, value, form, expected):
        """Echo evaluate."""

        class EchoBackend:
            """Echo backend."""

            fingerprint = "sha256:test-cf-echo"

            @staticmethod
            def encode(text):
                """Encode."""
                return tuple(ord(c) for c in text)

            @staticmethod
            def decode(ids):
                """Decode."""
                return "".join(chr(int(i)) for i in ids)

            def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
                """Generate."""
                text = self.decode(tuple(input_ids))
                found = re.search(r"inspection code is (\S+?)\.", text)
                answer = found.group(1) if found else "unknown"
                boundary = self.encode("endmark")
                out = list(self.encode(answer)) + list(boundary)
                out = (out + list(boundary) * max_new_tokens)[:max_new_tokens]
                return Generation(token_ids=tuple(out), reason="limit")

        fact = f"The {topic} inspection code is {value}."
        question = f"What is the {topic} inspection code?"
        content = unicodedata.normalize(form, f"{fact} {question}")
        case = Fixture(
            id="cf-00",
            messages=(Message(role="user", content=content),),
            expected=expected,
            group="cf",
        )
        backend = EchoBackend()
        report = evaluate(
            backend,
            _CHATML,
            (case,),
            boundary=tuple(backend.encode("endmark")),
            max_new_tokens=64,
        )
        return report.observations[0]

    def test_accented_value_flips_zero_defect_echo(self):
        """Test accented value flips zero defect echo."""
        nfc = self._echo_evaluate("café", "café-1234", "NFC", "café-1234")
        nfd = self._echo_evaluate("café", "café-1234", "NFD", "café-1234")
        self.assertTrue(nfc.exact_match)
        self.assertFalse(nfd.exact_match)
        self.assertEqual(nfc.answer, "café-1234")

    def test_ascii_value_never_flips_zero_defect_echo(self):
        """Test ascii value never flips zero defect echo."""
        nfc = self._echo_evaluate("café", "elm-1234", "NFC", "elm-1234")
        nfd = self._echo_evaluate("café", "elm-1234", "NFD", "elm-1234")
        self.assertTrue(nfc.exact_match)
        self.assertTrue(nfd.exact_match)


class ClassificationTests(unittest.TestCase):
    """Classification tests."""

    def test_total_function_branches(self):
        """Test total function branches."""
        self.assertEqual(cf.classify_canonical_form(5, 3, 3), "insufficient")
        self.assertEqual(cf.classify_canonical_form(6, 0, 0), "no_discordance_observed")
        self.assertEqual(cf.classify_canonical_form(6, 1, 0), "below_threshold")
        self.assertEqual(cf.classify_canonical_form(6, 0, 1), "below_threshold")
        self.assertEqual(cf.classify_canonical_form(6, 2, 2), "discordant_balanced")
        self.assertEqual(cf.classify_canonical_form(6, 3, 0), "discordant_asymmetric")
        self.assertEqual(cf.classify_canonical_form(6, 0, 3), "discordant_asymmetric")
        self.assertEqual(cf.classify_canonical_form(24, 2, 1), "discordant_asymmetric")

    def test_falsification_direction(self):
        """Test falsification direction."""
        # Falsified means the INVARIANCE claim fell, stated
        # explicitly so the boolean cannot be misread.
        self.assertFalse(cf.invariance_falsified("insufficient"))
        self.assertFalse(cf.invariance_falsified("no_discordance_observed"))
        self.assertFalse(cf.invariance_falsified("below_threshold"))
        self.assertTrue(cf.invariance_falsified("discordant_balanced"))
        self.assertTrue(cf.invariance_falsified("discordant_asymmetric"))


def _selection(decision, winner=None, withheld=False):
    """Selection."""
    return {
        "decision": decision,
        "detail": "" if not withheld else "selection_withheld_mismatched_sets",
        "winner": winner,
        "withheld": withheld,
        "flip": False,
    }


class WinnerHeadlineTests(unittest.TestCase):
    """Winner headline tests."""

    def test_changed_stable_abstention_undefined(self):
        """Test changed stable abstention undefined."""
        self.assertEqual(
            cf.winner_headline(
                {"nfc": _selection("selected", "a"), "nfd": _selection("selected", "b")}
            )["headline"],
            "winner_changed",
        )
        self.assertTrue(
            cf.winner_headline(
                {"nfc": _selection("selected", "a"), "nfd": _selection("selected", "b")}
            )["winner_changed"]
        )
        self.assertEqual(
            cf.winner_headline(
                {"nfc": _selection("selected", "a"), "nfd": _selection("selected", "a")}
            )["headline"],
            "winner_stable",
        )
        self.assertFalse(
            cf.winner_headline(
                {"nfc": _selection("selected", "a"), "nfd": _selection("selected", "a")}
            )["winner_changed"]
        )
        # Selection-vs-abstention is NEVER a winner change.
        self.assertEqual(
            cf.winner_headline(
                {"nfc": _selection("selected", "a"), "nfd": _selection("ambiguous")}
            )["headline"],
            "winner_abstention_asymmetric",
        )
        self.assertFalse(
            cf.winner_headline(
                {"nfc": _selection("selected", "a"), "nfd": _selection("ambiguous")}
            )["winner_changed"]
        )
        self.assertEqual(
            cf.winner_headline({"nfc": _selection("ambiguous"), "nfd": _selection("no_candidate")})[
                "headline"
            ],
            "winner_undefined",
        )
        self.assertFalse(
            cf.winner_headline({"nfc": _selection("ambiguous"), "nfd": _selection("no_candidate")})[
                "winner_changed"
            ]
        )

    def test_withheld_gives_undefined_and_none(self):
        """Test withheld gives undefined and none."""
        result = cf.winner_headline(
            {"nfc": _selection("unavailable", withheld=True), "nfd": _selection("selected", "a")}
        )
        self.assertEqual(result["headline"], "winner_undefined")
        self.assertIsNone(result["winner_changed"])


class StageTests(unittest.TestCase):
    """Stage tests."""

    def test_perfect_backend_no_discordance(self):
        """Test perfect backend no discordance."""
        payload = _payload()
        self.assertTrue(payload["available"])
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["headline"]["n_paired"], 24)
        self.assertEqual(entry["headline"]["classification"], "no_discordance_observed")
        self.assertFalse(entry["invariance_falsified"])
        self.assertEqual(entry["ids"]["n_ids_divergent"], 24)
        self.assertEqual(entry["ids"]["n_ids_identical"], 0)
        self.assertEqual(entry["ids"]["n_same_ids_verdict_flips"], 0)
        self.assertEqual(payload["selection"]["winner"]["headline"], "winner_stable")
        self.assertEqual(payload["n_cases"], 24)
        self.assertEqual(payload["n_candidates"], 1)

    def test_single_discordance_is_below_threshold_not_falsified(self):
        """Test single discordance is below threshold not falsified."""
        # An NFD-arm failure is an NFC-ONLY pass: the NFC arm
        # passed where the NFD arm did not. The direction label
        # follows the passing arm, not the failing one.
        payload = _payload(_WordBackend(mode="nfd_fail", fail_count=1))
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["headline"]["n_nfc_only"], 1)
        self.assertEqual(entry["headline"]["classification"], "below_threshold")
        self.assertFalse(entry["invariance_falsified"])

    def test_discordant_asymmetric_falsifies(self):
        """Test discordant asymmetric falsifies."""
        payload = _payload(_WordBackend(mode="nfd_fail", fail_count=3))
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["headline"]["n_nfc_only"], 3)
        self.assertEqual(entry["headline"]["classification"], "discordant_asymmetric")
        self.assertTrue(entry["invariance_falsified"])

    def test_discordant_balanced_falsifies(self):
        """Test discordant balanced falsifies."""
        payload = _payload(_BalancedBackend())
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["headline"]["n_nfc_only"], 2)
        self.assertEqual(entry["headline"]["n_nfd_only"], 2)
        self.assertEqual(entry["headline"]["classification"], "discordant_balanced")
        self.assertTrue(entry["invariance_falsified"])

    def test_normalizing_backend_all_ids_identical(self):
        """Test normalizing backend all ids identical."""
        payload = _payload(_NormalizingBackend())
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["ids"]["n_ids_identical"], 24)
        self.assertEqual(entry["ids"]["n_ids_divergent"], 0)
        self.assertEqual(entry["headline"]["n_paired"], 0)
        self.assertEqual(entry["headline"]["classification"], "insufficient")
        self.assertFalse(entry["invariance_falsified"])

    def test_same_ids_flips_are_determinism_findings(self):
        """Test same ids flips are determinism findings."""
        payload = _payload(_AlternatingBackend())
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["ids"]["n_ids_identical"], 24)
        self.assertEqual(entry["ids"]["n_same_ids_verdict_flips"], 24)
        # Headline excludes ids-identical cases entirely.
        self.assertEqual(entry["headline"]["n_paired"], 0)
        self.assertEqual(entry["headline"]["classification"], "insufficient")
        self.assertFalse(entry["invariance_falsified"])

    def test_context_exceeded_cells_are_insufficient(self):
        """Test context exceeded cells are insufficient."""
        payload = _payload(max_context=8)
        entry = payload["candidates"]["chatml"]
        record = entry["records"][0]
        self.assertEqual(record["cell_details"]["nfc"], "arm_context_exceeded")
        self.assertEqual(record["cell_details"]["nfd"], "arm_context_exceeded")
        self.assertIsNone(record["cells"]["nfc"])
        self.assertEqual(entry["headline"]["classification"], "insufficient")

    def test_cancel_unwinds_to_fail_closed_payload(self):
        """Test cancel unwinds to fail closed payload."""
        event = threading.Event()
        event.set()
        backend = _WordBackend()
        stage = experiments.canonicalform_stage(
            backend,
            (_CHATML,),
            boundary=tuple(backend.encode("endmark")),
            max_context=4096,
            seed=0,
            cancel=event,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")
        analysis = experiments.canonicalform_analysis(stage, ("chatml",))
        self.assertFalse(analysis["available"])
        self.assertEqual(analysis["detail"], "cancelled")

    def test_duplicate_format_names_rejected(self):
        """Test duplicate format names rejected."""
        backend = _WordBackend()
        with self.assertRaises(ValueError):
            experiments.canonicalform_stage(
                backend,
                (_CHATML, _CHATML),
                boundary=tuple(backend.encode("endmark")),
                max_context=4096,
                seed=0,
            )

    def test_non_string_fingerprint_rejected(self):
        """Test non string fingerprint rejected."""

        class BadBackend(_WordBackend):
            """Bad backend."""

            fingerprint = 123  # type: ignore[assignment]

        backend = BadBackend()
        with self.assertRaises(ValueError):
            experiments.canonicalform_stage(
                backend,
                (_CHATML,),
                boundary=tuple(backend.encode("endmark")),
                max_context=4096,
                seed=0,
            )

    def test_winner_changed_across_arms(self):
        """Test winner changed across arms."""
        backend = _PerFormatBackend({("chatml", "nfd"): 1, ("llama", "nfc"): 1})
        payload = _payload(backend, formats=(_CHATML, _LLAMA))
        self.assertEqual(payload["conditions"]["nfc"]["winner"], "chatml")
        self.assertEqual(payload["conditions"]["nfd"]["winner"], "llama")
        self.assertEqual(payload["selection"]["winner"]["headline"], "winner_changed")
        self.assertTrue(payload["selection"]["winner"]["winner_changed"])

    def test_winner_abstention_asymmetric_is_not_a_change(self):
        """Test winner abstention asymmetric is not a change."""
        backend = _PerFormatBackend({("llama", "nfc"): 1})
        payload = _payload(backend, formats=(_CHATML, _LLAMA))
        self.assertEqual(payload["conditions"]["nfc"]["winner"], "chatml")
        self.assertEqual(payload["conditions"]["nfd"]["decision"], "ambiguous")
        self.assertEqual(payload["selection"]["winner"]["headline"], "winner_abstention_asymmetric")
        self.assertFalse(payload["selection"]["winner"]["winner_changed"])

    def test_error_reason_backend_yields_failed_cells_not_a_crash(self):
        """Test error reason backend yields failed cells not a crash."""

        class _ErrorBackend(_WordBackend):
            """Error backend."""

            @staticmethod
            def generate(input_ids, *, max_new_tokens, seed=None, cancel=None):
                """Generate."""
                return Generation(token_ids=(), reason="error")

        backend = _ErrorBackend()
        stage = experiments.canonicalform_stage(
            backend,
            (_CHATML,),
            boundary=tuple(backend.encode("endmark")),
            max_context=4096,
            seed=0,
        )
        # An "error" generation is non-eligible, not a raise:
        # the stage stays available and each cell degrades
        # individually to arm_evaluation_failed.
        self.assertTrue(stage["available"])
        payload = experiments.canonicalform_analysis(stage, ("chatml",))
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["records"][0]["cell_details"]["nfc"], "arm_evaluation_failed")
        self.assertEqual(entry["headline"]["classification"], "insufficient")


class DiffValidatorTests(unittest.TestCase):
    """Diff validator tests."""

    def test_identical_payloads_are_unchanged(self):
        """Test identical payloads are unchanged."""
        payload = _payload()
        diff = diffreports.diff_canonicalform(payload, copy.deepcopy(payload))
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])

    def test_missing_arm_payload_is_neither(self):
        """Test missing arm payload is neither."""
        diff = diffreports.diff_canonicalform(None, None)
        self.assertEqual(diff["available"], "neither")
        self.assertFalse(diff["changed"])

    def test_tampered_content_digest_is_invalid(self):
        """Test tampered content digest is invalid."""
        payload = _payload()
        bad = copy.deepcopy(payload)
        record = bad["candidates"]["chatml"]["records"][0]
        record["nfd_content_digest"] = record["nfc_content_digest"]
        self.assertFalse(diffreports._valid_canonicalform(bad))
        diff = diffreports.diff_canonicalform(payload, bad)
        self.assertEqual(diff["available"], "old")

    def test_tampered_verdict_binding_is_invalid(self):
        """Test tampered verdict binding is invalid."""
        payload = _payload()
        bad = copy.deepcopy(payload)
        bad["candidates"]["chatml"]["records"][0]["cells"]["nfc"]["verdict"] = False
        self.assertFalse(diffreports._valid_canonicalform(bad))

    def test_tampered_ids_consistency_is_invalid(self):
        """Test tampered ids consistency is invalid."""
        payload = _payload()
        bad = copy.deepcopy(payload)
        record = bad["candidates"]["chatml"]["records"][0]
        record["ids_identical"] = True
        record["nfd_tokens"] = record["nfc_tokens"] + 1
        record["id_length_delta"] = -1
        self.assertFalse(diffreports._valid_canonicalform(bad))

    def test_tampered_aggregate_is_invalid(self):
        """Test tampered aggregate is invalid."""
        payload = _payload()
        bad = copy.deepcopy(payload)
        bad["candidates"]["chatml"]["headline"]["n_paired"] = 0
        self.assertFalse(diffreports._valid_canonicalform(bad))

    def test_tampered_protocol_is_invalid(self):
        """Test tampered protocol is invalid."""
        payload = _payload()
        bad = copy.deepcopy(payload)
        bad["protocol"]["fingerprint"] = "sha256:0000"
        self.assertFalse(diffreports._valid_canonicalform(bad))

    def test_classification_change_is_changed(self):
        """Test classification change is changed."""
        first = _payload()
        second = _payload(_WordBackend(mode="nfd_fail", fail_count=3))
        diff = diffreports.diff_canonicalform(first, second)
        self.assertEqual(diff["available"], "both")
        self.assertTrue(diff["changed"])
        metrics = {(entry["candidate"], entry["metric"]) for entry in diff["candidate_drift"]}
        self.assertIn(("chatml", "classification"), metrics)
        self.assertIn(("chatml", "invariance_falsified"), metrics)

    def test_protocol_drift_marks_incomparable(self):
        """Test protocol drift marks incomparable."""
        first = _payload()
        second = _payload(seed=1)
        diff = diffreports.diff_canonicalform(first, second)
        self.assertFalse(diff["comparable"])
        self.assertTrue(diff["changed"])

    def test_diff_reports_wires_section(self):
        """Test diff reports wires section."""

        # diff_reports needs the full report shape (decision,
        # search summaries, held-out summary); the section
        # wiring is proved by identical reports that differ in
        # nothing, then in canonical_form only.
        def _report(payload):
            """Report."""
            return {
                "decision": "select_best",
                "selected_candidate": "alpha",
                "search": [
                    {
                        "format": {"name": "alpha"},
                        "summary": {
                            "planned": 2,
                            "exact_and_boundary_passes": 2,
                            "aggregate_eligible": True,
                        },
                        "observations": [
                            {"fixture": "f1", "exact_match": True, "boundary_found": True},
                            {"fixture": "f2", "exact_match": True, "boundary_found": True},
                        ],
                    }
                ],
                "held_out": {
                    "summary": {
                        "planned": 1,
                        "exact_and_boundary_passes": 1,
                        "aggregate_eligible": True,
                    },
                    "observations": [
                        {"fixture": "h1", "exact_match": True, "boundary_found": True}
                    ],
                },
                "canonical_form": payload,
            }

        payload = _payload()
        result = diffreports.diff_reports(_report(payload), _report(copy.deepcopy(payload)))
        self.assertIn("canonical_form", result)
        self.assertFalse(result["canonical_form"]["changed"])
        self.assertIn("canonical_form_changed", result["summary"])
        self.assertFalse(result["summary"]["canonical_form_changed"])
        self.assertFalse(result["summary"]["canonical_form_asymmetric"])
        self.assertTrue(result["summary"]["identical"])
        changed = _payload(_WordBackend(mode="nfd_fail", fail_count=3))
        result = diffreports.diff_reports(_report(payload), _report(changed))
        self.assertTrue(result["canonical_form"]["changed"])
        self.assertTrue(result["summary"]["canonical_form_changed"])
        self.assertFalse(result["summary"]["identical"])


class ReportingTests(unittest.TestCase):
    """Reporting tests."""

    def test_section_renders_headline_and_counts(self):
        """Test section renders headline and counts."""
        payload = _payload()
        html = reporting_mod._canonical_form_section({"canonical_form": payload})
        self.assertIn("canonical-form", html)
        self.assertIn("Canonical-form sensitivity", html)
        self.assertIn("no_discordance_observed", html)
        self.assertIn("winner_stable", html)

    def test_section_empty_when_unavailable(self):
        """Test section empty when unavailable."""
        self.assertEqual(
            reporting_mod._canonical_form_section(
                {"canonical_form": {"available": False, "detail": "canonical_form_unavailable"}}
            ),
            "",
        )
        self.assertEqual(reporting_mod._canonical_form_section({}), "")

    def test_section_escapes_hostile_payload_text(self):
        """Test section escapes hostile payload text."""
        # The section's contract is _escape_text's contract
        # (the cycle-100 pin): HTML metacharacters escaped, C0
        # controls and lone surrogates stripped. U+2028/U+2029
        # deliberately pass through in the main paths (SECURITY
        # residual #9, diagnostic fidelity), so they are NOT
        # asserted absent here.
        payload = _payload()
        hostile = "x<script>alert(7)</script>\x00\x1b\ud800"
        payload["candidates"][hostile] = copy.deepcopy(payload["candidates"]["chatml"])
        html = reporting_mod._canonical_form_section({"canonical_form": payload})
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script>alert", html)
        self.assertNotIn("\x00", html)
        self.assertNotIn("\x1b", html)
        self.assertNotIn("\ud800", html)


class CliDispatchTests(unittest.TestCase):
    """Cli dispatch tests."""

    def test_run_canonical_form_probe_attaches_and_fails_closed(self):
        """Test run canonical form probe attaches and fails closed."""
        result = {}
        backend = _WordBackend()
        from model_protocol_lab.cli import _run_canonical_form_probe

        _run_canonical_form_probe(
            backend,
            (_CHATML,),
            result,
            boundary=tuple(backend.encode("endmark")),
            boundary_map=None,
            max_new_tokens=64,
            max_context=4096,
            seed=0,
            cancel=None,
        )
        self.assertTrue(result["canonical_form"]["available"])
        self.assertEqual(result["canonical_form"]["n_cases"], 24)

        # A backend whose generate raises degrades per cell
        # (evaluate converts the raise to a non-eligible
        # report), so the runner-level fail-closed shape needs
        # a stage-level raise: duplicate format names violate
        # the stage's unique-names precondition.
        result = {}
        _run_canonical_form_probe(
            backend,
            (_CHATML, _CHATML),
            result,
            boundary=tuple(backend.encode("endmark")),
            boundary_map=None,
            max_new_tokens=64,
            max_context=4096,
            seed=0,
            cancel=None,
        )
        self.assertFalse(result["canonical_form"]["available"])
        self.assertEqual(result["canonical_form"]["detail"], "canonical_form_unavailable")


def _load_mutant(source_path, name):
    """Exec a mutated canonicalform copy under the package
    namespace (the cycle-169 pattern: the module's relative
    import of .confidentiality resolves through sys.modules).
    """
    import importlib.util
    import sys

    dotted = f"model_protocol_lab.canonicalform_mut_{name}"
    spec = importlib.util.spec_from_file_location(dotted, source_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


_MUTANTS = {
    "classify_floor": (
        "if nfc_only == nfd_only and nfc_only >= MIN_DISCORDANT:",
        "if nfc_only == nfd_only and nfc_only >= 1:",
    ),
    "falsified_always_false": (
        'return classification in ("discordant_balanced", "discordant_asymmetric")',
        "return False",
    ),
    "arm_swap": (
        'if arm == "nfc":\n        return unicodedata.normalize("NFC", content)',
        'if arm == "nfc":\n        return unicodedata.normalize("NFD", content)',
    ),
    "pair_block_includes_identical": (
        'if record.get("ids_identical") is not False:',
        'if record.get("ids_identical") is not True:',
    ),
    "ascii_pin_dropped": (
        "if not isinstance(value, str) or not value.isascii():",
        "if not isinstance(value, str):",
    ),
    "winner_change_inverted": (
        'if nfc.get("winner") != nfd.get("winner"):',
        'if nfc.get("winner") == nfd.get("winner"):',
    ),
    "content_digest_dropped": (
        'digest_text(form_content(case, "nfc")),',
        'digest_text(form_content(case, "nfd")),',
    ),
}


def _scenario_classify_floor(mod):
    """Scenario classify floor."""
    # One discordant pair each way is BELOW the floor, not
    # balanced: the floor mutant (>= 1) calls it balanced.
    assert mod.classify_canonical_form(6, 1, 1) == "below_threshold"
    assert mod.classify_canonical_form(6, 2, 2) == "discordant_balanced"


def _scenario_falsified(mod):
    """Scenario falsified."""
    assert mod.invariance_falsified("discordant_asymmetric") is True


def _scenario_arm_swap(mod):
    """Scenario arm swap."""
    case = mod.generate_cases(0)[0]
    assert mod.form_content(case, "nfc") == unicodedata.normalize("NFC", case["content"])
    assert mod.form_content(case, "nfd") == unicodedata.normalize("NFD", case["content"])


def _scenario_pair_block(mod):
    """Scenario pair block."""
    records = [
        {
            "ids_identical": True,
            "cells": {"nfc": {"verdict": True}, "nfd": {"verdict": False}},
            "cell_details": {"nfc": "", "nfd": ""},
        }
    ]
    aggregate = mod.aggregate_candidate(records)
    assert aggregate["headline"]["n_paired"] == 0


def _scenario_ascii_pin(mod):
    """Scenario ascii pin."""
    try:
        mod._validate_case_text("The café code is elm-1.", "café-1234")
    except ValueError:
        return
    raise AssertionError("accented value accepted")


def _scenario_winner_change(mod):
    """Scenario winner change."""
    headline = mod.winner_headline(
        {
            "nfc": _selection("selected", "a"),
            "nfd": _selection("selected", "b"),
        }
    )
    assert headline["headline"] == "winner_changed"
    assert headline["winner_changed"] is True


def _scenario_content_digest(mod):
    """Scenario content digest."""
    # canonical_generated_rows are lists: [fixture_id, topic,
    # value, sentence_indices, nfc_digest, nfd_digest].
    rows = mod.canonical_generated_rows(0)
    cases = mod.generate_cases(0)
    row = rows[0]
    case = cases[0]
    assert row[4] != row[5]
    assert row[4] == mod.digest_text(mod.form_content(case, "nfc"))
    assert row[5] == mod.digest_text(mod.form_content(case, "nfd"))


_SCENARIOS = {
    "classify_floor": _scenario_classify_floor,
    "falsified_always_false": _scenario_falsified,
    "arm_swap": _scenario_arm_swap,
    "pair_block_includes_identical": _scenario_pair_block,
    "ascii_pin_dropped": _scenario_ascii_pin,
    "winner_change_inverted": _scenario_winner_change,
    "content_digest_dropped": _scenario_content_digest,
}


class MutationGateTests(unittest.TestCase):
    """Tempdir gates (cycle-169/179 pattern): each mutant is
    killed by a NAMED scenario that passes on the real module
    and raises on the mutant. A gate that passes vacuously is
    not a gate: the scenario is first run against the real
    module in the same test.
    """

    def test_gates_kill_every_mutant(self):
        """Test gates kill every mutant."""
        import tempfile
        from pathlib import Path

        real_path = Path(cf.__file__)
        source = real_path.read_text(encoding="utf-8")
        for name, (original, mutant) in _MUTANTS.items():
            scenario = _SCENARIOS[name]
            scenario(cf)  # the scenario passes on the real module
            self.assertIn(original, source, f"anchor missing for {name}")
            mutated = source.replace(original, mutant, 1)
            self.assertNotEqual(mutated, source, f"mutant no-op for {name}")
            with tempfile.TemporaryDirectory(prefix="mpl_cf_gate_") as tmp:
                path = Path(tmp) / f"canonicalform_mut_{name}.py"
                path.write_text(mutated, encoding="utf-8")
                mod = _load_mutant(str(path), name)
                with self.assertRaises(AssertionError, msg=f"mutant survived: {name}"):
                    scenario(mod)


class GapCoverageTests(unittest.TestCase):
    """Scoped-coverage gap tests (cycle-209 gate): every line
    and branch the scoped run showed unhit in canonicalform.py
    gets a test that executes it, not a coverage-percentage
    argument. A guard proven unreachable would be removed
    instead (cycle-171); each guard below is reachable.
    """

    def test_generate_cases_rejects_non_int_seed(self):
        """Test generate cases rejects non int seed."""
        with self.assertRaises(ValueError):
            cf.generate_cases("0")
        with self.assertRaises(ValueError):
            cf.generate_cases(1.5)

    def test_value_collision_redraws(self):
        """Test value collision redraws."""
        # Seed 636: index 7's first draw (drift-1299) collides
        # with index 2's value. Without the redraw loop the
        # set would contain a duplicate; with it, index 7
        # carries the redrawn value.
        cases = cf.generate_cases(636)
        self.assertEqual(len({c["value"] for c in cases}), 24)
        self.assertEqual(cases[2]["value"], "drift-1299")
        self.assertEqual(cases[7]["value"], "quill-8528")

    def test_neither_case_in_pair_block(self):
        """Test neither case in pair block."""
        # Both arms failing the same case is the "neither"
        # outcome: it must not count as discordance, and it
        # exercises the pair block's final elif falling
        # through (both verdicts False).
        payload = _payload(_WordBackend(mode="both_fail", fail_count=1))
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["headline"]["n_neither"], 1)
        self.assertEqual(entry["headline"]["n_paired"], 24)
        self.assertEqual(entry["headline"]["classification"], "no_discordance_observed")

    def test_aggregate_tolerates_malformed_records(self):
        """Test aggregate tolerates malformed records."""
        # _cell's non-dict-cells branch, the non-int delta
        # coercion, the same-ids incomplete-cell skip, and a
        # record whose ids_identical is neither True nor False
        # (counted in neither share, delta still recorded).
        records = [
            {"ids_identical": False, "id_length_delta": 0, "cells": None},
            {
                "ids_identical": False,
                "id_length_delta": "bad",
                "cells": {"nfc": "junk", "nfd": None},
            },
            {
                "ids_identical": True,
                "id_length_delta": 0,
                "cells": {"nfc": {"verdict": True}, "nfd": None},
            },
            {"ids_identical": None, "id_length_delta": 3, "cells": None},
        ]
        aggregate = cf.aggregate_candidate(records)
        self.assertEqual(aggregate["ids"]["id_length_delta_by_case"], [0, 0, 0, 3])
        self.assertEqual(aggregate["ids"]["n_ids_identical"], 1)
        self.assertEqual(aggregate["ids"]["n_ids_divergent"], 2)
        self.assertEqual(aggregate["ids"]["n_same_ids_verdict_flips"], 0)
        self.assertEqual(aggregate["n_incomplete"], 4)
        self.assertEqual(aggregate["headline"]["n_paired"], 0)

    def test_shared_fixtures_empty(self):
        """Test shared fixtures empty."""
        self.assertEqual(cf.shared_fixtures({}), frozenset())

    def test_arm_selection_withheld_on_empty_completed_set(self):
        """Test arm selection withheld on empty completed set."""
        good = [
            {
                "fixture": "canonicalform-00",
                "cells": {"nfc": {"verdict": True}, "nfd": {"verdict": True}},
                "ids_identical": False,
                "id_length_delta": 0,
            }
        ]
        empty = [
            {
                "fixture": "canonicalform-00",
                "cells": {"nfc": None, "nfd": None},
                "ids_identical": False,
                "id_length_delta": 0,
            }
        ]
        selection = cf.arm_selection({"a": good, "b": empty}, "nfc")
        self.assertTrue(selection["withheld"])
        self.assertEqual(selection["decision"], "unavailable")

    def test_arm_selection_withheld_on_mismatched_completed_sets(self):
        """Test arm selection withheld on mismatched completed sets."""
        # Both candidates complete something, but different
        # sets: the restricted comparison is withheld rather
        # than computed over mismatched populations.
        both = [
            {
                "fixture": "canonicalform-00",
                "cells": {"nfc": {"verdict": True}, "nfd": {"verdict": True}},
                "ids_identical": False,
                "id_length_delta": 0,
            },
            {
                "fixture": "canonicalform-01",
                "cells": {"nfc": {"verdict": True}, "nfd": {"verdict": True}},
                "ids_identical": False,
                "id_length_delta": 0,
            },
        ]
        partial = [
            {
                "fixture": "canonicalform-00",
                "cells": {"nfc": {"verdict": True}, "nfd": {"verdict": True}},
                "ids_identical": False,
                "id_length_delta": 0,
            },
            {
                "fixture": "canonicalform-01",
                "cells": {"nfc": None, "nfd": None},
                "ids_identical": False,
                "id_length_delta": 0,
            },
        ]
        selection = cf.arm_selection({"a": both, "b": partial}, "nfc")
        self.assertTrue(selection["withheld"])
        self.assertEqual(selection["decision"], "unavailable")

    def test_winner_headline_missing_selection(self):
        """Test winner headline missing selection."""
        result = cf.winner_headline({})
        self.assertEqual(result["headline"], "winner_undefined")
        self.assertEqual(result["detail"], "selection_missing")
        self.assertIsNone(result["winner_changed"])
        result = cf.winner_headline({"nfc": "not-a-dict", "nfd": {}})
        self.assertEqual(result["detail"], "selection_missing")


class FullRunGapTests(unittest.TestCase):
    """Full-run integration gaps (cycle-209 gate): lines and
    branches the gate's coverage.json showed unhit in the NEW
    canonical-form ranges of diffreports.py, reporting.py,
    and experiments.py. The new module itself was already at
    100% scoped; these are the integration lines the cycle-174
    rule budgets into the feature cycle (validator reject
    branches, section defensive branches, stage screens).
    Each test executes its lines; none argues from a
    percentage. The two guards proven unreachable by
    construction (equal digests after two passing digest
    checks; the audit screen on token-shape-free frozen
    texts) are proven by stubs, not left as decoration
    (cycle-171).
    """

    @staticmethod
    def _mirror_cell(verdict, tokens, digest):
        """Mirror cell."""
        return {
            "verdict": verdict,
            "boundary_found": True,
            "boundary_text_match": True,
            "input_tokens": tokens,
            "answer_digest": digest,
        }

    @staticmethod
    def _mirror_record(case, tokens=(10, 12)):
        """Mirror record."""
        return {
            "fixture": case["fixture_id"],
            "topic": case["topic"],
            "expected_digest": diffreports._canonicalform_digest(str(case["value"])),
            "nfc_tokens": tokens[0],
            "nfd_tokens": tokens[1],
            "ids_identical": False,
            "id_length_delta": tokens[0] - tokens[1],
            "nfc_content_digest": diffreports._canonicalform_digest(case["nfc_content"]),
            "nfd_content_digest": diffreports._canonicalform_digest(case["nfd_content"]),
            "cells": {"nfc": None, "nfd": None},
            "cell_details": {"nfc": "arm_context_exceeded", "nfd": "arm_context_exceeded"},
        }

    @staticmethod
    def _recompute_entry(records):
        """Recompute entry."""
        entry = dict.fromkeys(diffreports._CANONICALFORM_AGGREGATE_KEYS)
        entry.update({"status": "available", "detail": "", "records": records})
        return entry

    # -- mirror helpers -------------------------------------------------

    def test_mirror_digest_ok_branches(self):
        """Test mirror digest ok branches."""
        ok = diffreports._canonicalform_digest_ok
        self.assertTrue(ok(None))
        self.assertFalse(ok(123))
        self.assertFalse(ok("short"))
        self.assertFalse(ok("g" * 64))
        self.assertTrue(ok("a" * 64))

    def test_mirror_generate_redraws(self):
        """Test mirror generate redraws."""
        # Seed 636 collides (index 7's first draw repeats
        # index 2's value); seed 1 draws equal sentence
        # indices at index 3 and redraws the second.
        cases = diffreports._canonicalform_generate(636)
        self.assertEqual(len({c["value"] for c in cases}), 24)
        cases = diffreports._canonicalform_generate(1)
        self.assertEqual(cases[3]["sentence_indices"][0], 8)
        self.assertEqual(cases[3]["sentence_indices"][1], 9)

    def test_mirror_cell_get_non_dict(self):
        """Test mirror cell get non dict."""
        get = diffreports._canonicalform_cell_get
        self.assertIsNone(get({"cells": "junk"}, "nfc"))
        self.assertIsNone(get({"cells": {"nfc": "junk"}}, "nfc"))
        self.assertIsNone(get({}, "nfc"))

    def test_mirror_pair_block_branches(self):
        """Test mirror pair block branches."""

        def rec(identical, nfc, nfd):
            """Rec."""
            return {
                "ids_identical": identical,
                "cells": {"nfc": nfc, "nfd": nfd},
            }

        records = [
            rec(True, {"verdict": True}, {"verdict": False}),
            rec(False, {"verdict": True}, None),
            rec(False, {"verdict": False}, {"verdict": True}),
            rec(False, {"verdict": True}, {"verdict": True}),
            rec(False, {"verdict": True}, {"verdict": False}),
            rec(False, {"verdict": False}, {"verdict": False}),
        ]
        out = diffreports._canonicalform_pair_block_divergent(records)
        self.assertEqual(out["n_paired"], 4)
        self.assertEqual(out["n_nfc_only"], 1)
        self.assertEqual(out["n_nfd_only"], 1)
        self.assertEqual(out["n_both"], 1)
        self.assertEqual(out["n_neither"], 1)
        self.assertEqual(out["rate_nfc"], 0.5)
        self.assertEqual(out["rate_nfd"], 0.5)

    def test_mirror_classify_all_outcomes(self):
        """Test mirror classify all outcomes."""
        classify = diffreports._canonicalform_classify
        self.assertEqual(classify(5, 0, 0), "insufficient")
        self.assertEqual(classify(6, 0, 0), "no_discordance_observed")
        self.assertEqual(classify(6, 2, 2), "discordant_balanced")
        self.assertEqual(classify(6, 3, 0), "discordant_asymmetric")
        self.assertEqual(classify(6, 1, 0), "below_threshold")

    def test_mirror_aggregate_branches(self):
        """Test mirror aggregate branches."""
        records = [
            {"ids_identical": False, "id_length_delta": 0, "cells": None},
            {
                "ids_identical": True,
                "id_length_delta": 0,
                "cells": {"nfc": {"verdict": True}, "nfd": None},
            },
            {
                "ids_identical": True,
                "id_length_delta": 2,
                "cells": {"nfc": {"verdict": True}, "nfd": {"verdict": False}},
            },
            {
                "ids_identical": None,
                "id_length_delta": "bad",
                "cells": {"nfc": {"verdict": True}, "nfd": {"verdict": True}},
            },
            {
                "ids_identical": True,
                "id_length_delta": 0,
                "cells": {"nfc": {"verdict": True}, "nfd": {"verdict": True}},
            },
            {
                "ids_identical": False,
                "id_length_delta": -1,
                "cells": {"nfc": {"verdict": False}, "nfd": {"verdict": True}},
            },
        ]
        out = diffreports._canonicalform_aggregate(records)
        self.assertEqual(out["ids"]["n_ids_identical"], 3)
        self.assertEqual(out["ids"]["n_ids_divergent"], 2)
        self.assertEqual(out["ids"]["id_length_delta_by_case"], [0, 0, 2, 0, 0, -1])
        self.assertEqual(out["ids"]["n_same_ids_verdict_flips"], 1)
        self.assertEqual(out["ids"]["max_abs_id_delta"], 2)
        self.assertEqual(out["arms"]["nfc"]["n_completed"], 5)
        self.assertEqual(out["arms"]["nfc"]["n_pass"], 4)
        self.assertEqual(out["arms"]["nfd"]["n_completed"], 4)
        self.assertEqual(out["arms"]["nfd"]["n_pass"], 3)
        self.assertEqual(out["n_incomplete"], 2)
        self.assertEqual(out["headline"]["n_paired"], 1)
        self.assertEqual(out["headline"]["classification"], "insufficient")

    def test_mirror_selection_branches(self):
        """Test mirror selection branches."""

        def rec(fixture, verdict):
            """Rec."""
            return {
                "fixture": fixture,
                "cells": {"nfc": ({"verdict": verdict} if verdict is not None else None)},
            }

        sel = diffreports._canonicalform_selection_over
        self.assertIsNone(sel({"a": [rec("f1", True)]}, "nfc", frozenset({"f2"})))
        self.assertIsNone(
            sel(
                {"a": [rec("f1", True)], "b": [rec("f1", True), rec("f2", True)]},
                "nfc",
                frozenset({"f1", "f2"}),
            )
        )
        self.assertEqual(
            sel(
                {"a": [rec("f1", True)], "b": [rec("f1", True)]},
                "nfc",
                frozenset({"f1"}),
            ),
            {"decision": "ambiguous", "winner": None},
        )
        self.assertEqual(
            sel(
                {"a": [rec("f1", True)], "b": [rec("f1", False)]},
                "nfc",
                frozenset({"f1"}),
            ),
            {"decision": "selected", "winner": "a"},
        )
        self.assertEqual(diffreports._canonicalform_shared_fixtures({}), frozenset())
        withheld = diffreports._canonicalform_arm_selection({"a": []}, "nfc")
        self.assertTrue(withheld["withheld"])
        self.assertEqual(withheld["decision"], "unavailable")
        baseline = diffreports._canonicalform_arm_selection(
            {"a": [rec("f1", True)], "b": [rec("f1", False)]}, "nfc"
        )
        self.assertFalse(baseline["flip"])
        other = diffreports._canonicalform_arm_selection(
            {"a": [rec("f1", True)], "b": [rec("f1", False)]}, "nfd"
        )
        self.assertFalse(other["flip"])

    def test_mirror_winner_headline_branches(self):
        """Test mirror winner headline branches."""
        headline = diffreports._canonicalform_winner_headline
        self.assertEqual(headline("junk")["detail"], "selection_missing")
        self.assertEqual(
            headline({"nfc": {"withheld": True}, "nfd": {"withheld": False}})["detail"],
            "selection_withheld_mismatched_sets",
        )
        self.assertEqual(
            headline(
                {
                    "nfc": {"decision": "selected", "winner": "a", "withheld": False},
                    "nfd": {"decision": "selected", "winner": "b", "withheld": False},
                }
            )["headline"],
            "winner_changed",
        )
        self.assertEqual(
            headline(
                {
                    "nfc": {"decision": "selected", "winner": "a", "withheld": False},
                    "nfd": {"decision": "selected", "winner": "a", "withheld": False},
                }
            )["headline"],
            "winner_stable",
        )
        self.assertEqual(
            headline(
                {
                    "nfc": {"decision": "selected", "winner": "a", "withheld": False},
                    "nfd": {"decision": "ambiguous", "winner": None, "withheld": False},
                }
            )["headline"],
            "winner_abstention_asymmetric",
        )
        self.assertEqual(
            headline(
                {
                    "nfc": {"decision": "ambiguous", "winner": None, "withheld": False},
                    "nfd": {"decision": "ambiguous", "winner": None, "withheld": False},
                }
            )["headline"],
            "winner_undefined",
        )

    def test_mirror_cell_ok_rejects_each_break(self):
        """Test mirror cell ok rejects each break."""
        payload = _payload()
        record = payload["candidates"]["chatml"]["records"][0]
        cell = copy.deepcopy(record["cells"]["nfc"])
        expected = record["expected_digest"]
        ok = diffreports._canonicalform_cell_ok
        self.assertTrue(ok(cell, expected))
        self.assertFalse(ok("junk", expected))
        broken = copy.deepcopy(cell)
        del broken["verdict"]
        self.assertFalse(ok(broken, expected))
        broken = copy.deepcopy(cell)
        broken["verdict"] = "yes"
        self.assertFalse(ok(broken, expected))
        broken = copy.deepcopy(cell)
        broken["boundary_found"] = 1
        self.assertFalse(ok(broken, expected))
        broken = copy.deepcopy(cell)
        broken["boundary_text_match"] = "yes"
        self.assertFalse(ok(broken, expected))
        broken = copy.deepcopy(cell)
        broken["input_tokens"] = -1
        self.assertFalse(ok(broken, expected))
        broken = copy.deepcopy(cell)
        broken["input_tokens"] = "3"
        self.assertFalse(ok(broken, expected))
        broken = copy.deepcopy(cell)
        broken["answer_digest"] = "xyz"
        self.assertFalse(ok(broken, expected))

    # -- recompute battery ----------------------------------------------

    def test_recompute_rejects_each_break(self):
        """Test recompute rejects each break."""

        def bad_payload(mutate):
            """Bad payload."""
            payload = _payload()
            mutate(payload)
            return payload

        def check(mutate):
            """Check."""
            self.assertIsNone(diffreports._canonicalform_recompute(bad_payload(mutate)))

        check(lambda p: p.update(protocol="junk"))
        check(lambda p: p["protocol"].update(seed="0"))
        check(lambda p: p["protocol"].update(backend_fingerprint=123))
        check(lambda p: p.update(candidates={}))
        check(lambda p: p.update(candidates="junk"))
        check(lambda p: p["candidates"].update(chatml="junk"))
        check(
            lambda p: p["candidates"].update(
                chatml={"status": "unavailable", "detail": "d", "records": [{"x": 1}]}
            )
        )
        check(lambda p: p["candidates"]["chatml"].update(status="weird"))
        check(lambda p: p["candidates"]["chatml"].update(detail="x"))
        check(lambda p: p["candidates"]["chatml"].pop("ids"))
        check(lambda p: p["candidates"]["chatml"].update(records=[]))
        check(lambda p: p["candidates"]["chatml"].update(records="junk"))
        check(
            lambda p: p["candidates"]["chatml"].update(
                records=p["candidates"]["chatml"]["records"][:10]
            )
        )
        check(
            lambda p: p.update(
                candidates={
                    "c": {
                        **dict.fromkeys(diffreports._CANONICALFORM_AGGREGATE_KEYS),
                        "status": "available",
                        "detail": "",
                        "records": [{"junk": "x" * 600000} for _ in range(24)],
                    }
                }
            )
        )
        check(lambda p: p["candidates"]["chatml"]["records"][0].pop("cells"))
        check(lambda p: p["candidates"]["chatml"]["records"][0].update(topic="other"))
        check(lambda p: p["candidates"]["chatml"]["records"][0].update(expected_digest="0" * 64))
        check(lambda p: p["candidates"]["chatml"]["records"][0].update(nfc_content_digest="0" * 64))
        check(lambda p: p["candidates"]["chatml"]["records"][0].update(nfc_tokens="3"))
        check(lambda p: p["candidates"]["chatml"]["records"][0].update(nfd_tokens=-1))
        check(lambda p: p["candidates"]["chatml"]["records"][0].update(ids_identical="yes"))

        def wrong_delta(p):
            """Wrong delta."""
            record = p["candidates"]["chatml"]["records"][0]
            record["id_length_delta"] = record["nfc_tokens"] - record["nfd_tokens"] + 1

        check(wrong_delta)
        check(
            lambda p: p["candidates"]["chatml"]["records"][0].update(
                cells={"nfc": p["candidates"]["chatml"]["records"][0]["cells"]["nfc"]}
            )
        )
        check(lambda p: p["candidates"]["chatml"]["records"][0].update(cell_details={"nfc": ""}))

        def bad_none_detail(p):
            """Bad none detail."""
            record = p["candidates"]["chatml"]["records"][0]
            record["cells"]["nfc"] = None
            record["cell_details"]["nfc"] = "bogus"

        check(bad_none_detail)

        def nonzero_detail(p):
            """Nonzero detail."""
            record = p["candidates"]["chatml"]["records"][0]
            record["cell_details"]["nfc"] = "arm_context_exceeded"

        check(nonzero_detail)

        def token_mismatch(p):
            """Token mismatch."""
            record = p["candidates"]["chatml"]["records"][0]
            record["cells"]["nfc"]["input_tokens"] += 1

        check(token_mismatch)

    def test_recompute_total_records_cap(self):
        """Test recompute total records cap."""
        cases = diffreports._canonicalform_generate(0)
        records = [self._mirror_record(case) for case in cases]
        entry = self._recompute_entry(records)
        payload = {
            "protocol": diffreports._canonicalform_protocol(0, "sha256:test"),
            "candidates": {f"c{i}": copy.deepcopy(entry) for i in range(25)},
        }
        self.assertIsNone(diffreports._canonicalform_recompute(payload))

    def test_recompute_empty_raw_records(self):
        """Test recompute empty raw records."""
        payload = {
            "protocol": diffreports._canonicalform_protocol(0, "sha256:test"),
            "candidates": {"ghost": {"status": "unavailable", "detail": "d", "records": []}},
        }
        self.assertIsNone(diffreports._canonicalform_recompute(payload))

    def test_recompute_equal_digests_guard_proven_by_stub(self):
        """Test recompute equal digests guard proven by stub."""
        # The equal-digests reject is unreachable through real
        # payloads: every generated case diverges, so any
        # record passing both digest checks already has
        # differing digests. Prove the guard fires with a
        # stubbed generator whose one case does not diverge
        # (cycle-171: remove or prove with a stub).
        from unittest import mock

        real_cases = diffreports._canonicalform_generate(0)
        stub_cases = [dict(case) for case in real_cases]
        stub_cases[0]["nfd_content"] = stub_cases[0]["nfc_content"]

        def build(cases):
            """Build."""
            return {
                "protocol": diffreports._canonicalform_protocol(0, "sha256:test"),
                "candidates": {"c": self._recompute_entry([self._mirror_record(c) for c in cases])},
            }

        with mock.patch.object(diffreports, "_canonicalform_generate", return_value=stub_cases):
            self.assertIsNone(diffreports._canonicalform_recompute(build(stub_cases)))
        self.assertIsNotNone(diffreports._canonicalform_recompute(build(real_cases)))

    # -- _valid mismatch branches ----------------------------------------

    def test_valid_mismatch_branches(self):
        """Test valid mismatch branches."""
        payload = _payload()
        bad = copy.deepcopy(payload)
        bad["extra"] = 1
        self.assertFalse(diffreports._valid_canonicalform(bad))
        bad = copy.deepcopy(payload)
        bad["detail"] = "x"
        self.assertFalse(diffreports._valid_canonicalform(bad))
        bad = copy.deepcopy(payload)
        bad["conditions"] = {}
        self.assertFalse(diffreports._valid_canonicalform(bad))
        bad = copy.deepcopy(payload)
        bad["selection"] = {}
        self.assertFalse(diffreports._valid_canonicalform(bad))
        bad = copy.deepcopy(payload)
        bad["n_candidates"] = 99
        self.assertFalse(diffreports._valid_canonicalform(bad))
        bad = copy.deepcopy(payload)
        bad["n_cases"] = 99
        self.assertFalse(diffreports._valid_canonicalform(bad))

    # -- diff availability / status drift ---------------------------------

    @staticmethod
    def _mixed_payload():
        """Mixed payload."""
        # A valid payload with one available and one
        # unavailable candidate, built through the real
        # analysis path (not hand-assembled).
        backend = _WordBackend()
        stage = experiments.canonicalform_stage(
            backend,
            (_CHATML,),
            boundary=tuple(backend.encode("endmark")),
            max_context=4096,
            seed=0,
        )
        stage["candidates"]["llama"] = {
            "status": "unavailable",
            "detail": "d",
            "records": [],
        }
        return experiments.canonicalform_analysis(stage, ("chatml", "llama"))

    def test_diff_new_availability_and_status_drift(self):
        """Test diff new availability and status drift."""
        payload = _payload()
        self.assertEqual(diffreports.diff_canonicalform(None, payload)["available"], "new")
        mixed = self._mixed_payload()
        self.assertTrue(diffreports._valid_canonicalform(mixed))
        self.assertEqual(mixed["candidates"]["llama"]["status"], "unavailable")
        old_two = _payload(_WordBackend(), formats=(_CHATML, _LLAMA))
        diff = diffreports.diff_canonicalform(old_two, mixed)
        self.assertEqual(diff["available"], "both")
        metrics = {(e["candidate"], e["metric"]) for e in diff["candidate_drift"]}
        self.assertIn(("llama", "status"), metrics)
        same = diffreports.diff_canonicalform(mixed, copy.deepcopy(mixed))
        self.assertEqual(same["candidate_drift"], [])

    # -- stage screens and preconditions -----------------------------------

    def test_stage_format_preconditions(self):
        """Test stage format preconditions."""
        backend = _WordBackend()
        with self.assertRaises(ValueError):
            experiments.canonicalform_stage(
                backend,
                [],
                boundary=tuple(backend.encode("endmark")),
                max_context=4096,
                seed=0,
            )
        with self.assertRaises(ValueError):
            experiments.canonicalform_stage(
                backend,
                (),
                boundary=tuple(backend.encode("endmark")),
                max_context=4096,
                seed=0,
            )

    def test_stage_record_budget_exceeded(self):
        """Test stage record budget exceeded."""
        backend = _WordBackend()
        formats = tuple(
            Format(
                name=f"cf{i}",
                bos="",
                user_prefix=f"u{i}:",
                assistant_prefix="a:",
                turn_end="</s>",
            )
            for i in range(25)
        )
        stage = experiments.canonicalform_stage(
            backend,
            formats,
            boundary=tuple(backend.encode("endmark")),
            max_context=4096,
            seed=0,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "canonical_form_record_budget_exceeded")

    def test_stage_special_token_screen_wiring_proven_by_stub(self):
        """Test stage special token screen wiring proven by stub."""
        # The audit screen's wiring, proven by stub (cycle-171):
        # the real audit cannot fire on this stage's input --
        # generated_text is frozen templates/topics/words/pool
        # containing zero token-shaped literals, so no format
        # inventory can ever collide with it. The stub proves
        # the contract that matters: IF the audit reports a
        # finding, the stage fails closed with screen_failed
        # instead of generating. Deleting the screen check
        # fails this test.
        from unittest import mock

        from model_protocol_lab import fixtures as fixtures_mod
        from model_protocol_lab.fixtures import AuditFinding

        backend = _WordBackend()
        finding = AuditFinding(
            fixture_id="generated",
            field="text",
            literal="<|im_start|>",
            format_name="screen",
            offset=0,
        )
        # The stage imports the audit function locally on
        # each call, so the patch target is its home module.
        with mock.patch.object(
            fixtures_mod,
            "audit_special_token_literals",
            return_value=(finding,),
        ):
            stage = experiments.canonicalform_stage(
                backend,
                (_CHATML,),
                boundary=tuple(backend.encode("endmark")),
                max_context=4096,
                seed=0,
            )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "canonical_form_screen_failed")

    def test_stage_boundary_text_screen(self):
        """Test stage boundary text screen."""
        backend = _WordBackend()
        stage = experiments.canonicalform_stage(
            backend,
            (_CHATML,),
            boundary=tuple(backend.encode("elm")),
            max_context=4096,
            seed=0,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "canonical_form_screen_failed")

    def test_stage_undecodable_boundary_screens_nothing(self):
        """Test stage undecodable boundary screens nothing."""

        class _RaiseBoundaryBackend(_WordBackend):
            """Raise boundary backend."""

            def decode(self, ids):
                """Decode."""
                if tuple(ids) == tuple(self.encode("endmark")):
                    raise ValueError("undecodable")
                return super().decode(ids)

        backend = _RaiseBoundaryBackend()
        stage = experiments.canonicalform_stage(
            backend,
            (_CHATML,),
            boundary=tuple(backend.encode("endmark")),
            max_context=4096,
            seed=0,
        )
        self.assertTrue(stage["available"])

    def test_stage_mid_arm_cancel_unwinds(self):
        """Test stage mid arm cancel unwinds."""

        class _CancelOnRenderBackend(_WordBackend):
            """Cancel on render backend."""

            def __init__(self, event):
                super().__init__()
                self.event = event

            def encode(self, text):
                """Encode."""
                if "inspection code" in text:
                    self.event.set()
                return super().encode(text)

        event = threading.Event()
        backend = _CancelOnRenderBackend(event)
        stage = experiments.canonicalform_stage(
            backend,
            (_CHATML,),
            boundary=tuple(backend.encode("endmark")),
            max_context=4096,
            seed=0,
            cancel=event,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")

    # -- reporting defensive branches ---------------------------------------

    def test_section_non_dict_candidates_is_empty(self):
        """Test section non dict candidates is empty."""
        self.assertEqual(
            reporting_mod._canonical_form_section(
                {"canonical_form": {"available": True, "candidates": "junk"}}
            ),
            "",
        )

    def test_stage_cell_none_none_fallthrough_proven_by_stub(self):
        """Test stage cell none none fallthrough proven by stub."""
        # _canonical_form_cell never returns (None, None):
        # every path yields a detail, a cell, or a raise. The
        # arm loop's fall-through (neither assignment) is
        # therefore unreachable in reality; the stub proves
        # it is harmless (cell stays None, detail stays "")
        # rather than leaving it as decoration (cycle-171).
        from unittest import mock

        backend = _WordBackend()
        with mock.patch.object(experiments, "_canonical_form_cell", return_value=(None, None)):
            stage = experiments.canonicalform_stage(
                backend,
                (_CHATML,),
                boundary=tuple(backend.encode("endmark")),
                max_context=4096,
                seed=0,
            )
        self.assertTrue(stage["available"])
        record = stage["candidates"]["chatml"]["records"][0]
        self.assertIsNone(record["cells"]["nfc"])
        self.assertEqual(record["cell_details"]["nfc"], "")

    def test_section_fat_payload_branches(self):
        """Test section fat payload branches."""
        probe = {
            "available": True,
            "detail": "",
            "candidates": {
                "bad": "not-a-dict",
                "off": {"status": "unavailable"},
                "h": {
                    "status": "available",
                    "headline": "x",
                    "invariance_falsified": False,
                    "ids": "x",
                    "arms": "x",
                },
                "m": {
                    "status": "available",
                    "headline": {
                        "n_paired": "many",
                        "n_nfc_only": 1,
                        "n_nfd_only": 2,
                        "n_both": 3,
                        "n_neither": 4,
                        "classification": "weird",
                    },
                    "invariance_falsified": True,
                    "ids": {
                        "n_ids_identical": "x",
                        "n_ids_divergent": 1,
                        "n_same_ids_verdict_flips": 0,
                        "max_abs_id_delta": 2,
                    },
                    "arms": {"nfc": {"rate": "bad"}, "nfd": {"rate": None}},
                },
            },
            "selection": {"winner": {"headline": "h", "winner_changed": True}},
        }
        html = reporting_mod._canonical_form_section({"canonical_form": probe})
        self.assertIn("unavailable", html)
        self.assertIn("n/a", html)
        self.assertNotIn(">bad<", html)
        self.assertEqual(html.count("<tr>"), 4)

        base = _payload()
        no_selection = reporting_mod._canonical_form_section(
            {"canonical_form": {**base, "selection": "x"}}
        )
        self.assertNotIn("Winner headline", no_selection)
        no_winner = reporting_mod._canonical_form_section(
            {"canonical_form": {**base, "selection": {"winner": "x"}}}
        )
        self.assertNotIn("Winner headline", no_winner)
        maybe = reporting_mod._canonical_form_section(
            {
                "canonical_form": {
                    **base,
                    "selection": {"winner": {"headline": "h", "winner_changed": "maybe"}},
                }
            }
        )
        self.assertIn("undefined", maybe)


if __name__ == "__main__":
    unittest.main()
