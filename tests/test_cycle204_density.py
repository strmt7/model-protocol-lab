"""Cycle 204: instruction-density ladder (`run --density`).

The instrument (docs/DENSITY_DESIGN.md): generated cases with
16 simultaneously-satisfiable constraints, answered solo and
stacked (2/4/8/16) plus a length-matched sham; answers are
committed as ordered per-word hash lists and every verdict is
derived from the commitment. These tests pin the design's
load-bearing claims with spec citations: satisfiability by
construction (a perfect follower passes every arm -- the
cycle-204 red-team F1 invariant), the frozen normalization
table (B-F3), the exact sign test and its case-level
aggregation (F2), the withheld verdict below six discordant
cases, and the diff validator's full-recompute standard.
"""

from __future__ import annotations

import copy
import re
import threading
import unittest
from unittest import mock

from model_protocol_lab import cli, density, diffreports, experiments
from model_protocol_lab.core import Format, Generation
from model_protocol_lab.diffreports import _valid_density, diff_density
from model_protocol_lab.reporting import render_html

FORMATS = (
    Format(name="dn-a", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>"),
    Format(name="dn-b", bos="", user_prefix="U>", assistant_prefix="A>", turn_end="<e>"),
)

_FACT_RE = re.compile(r"The (.+?) signal word is ([a-z]+)\.")
_INCLUDE_RE = re.compile(r"Include the word ([a-z]+) in your answer\.")
_START_RE = re.compile(r"Start your answer with the word ([a-z]+)\.")
_END_RE = re.compile(r"End your answer with the word ([a-z]+)\.")


class _DensityBackend:
    """Word-token scripted backend with switchable behaviours.

    Modes: ``perfect`` (satisfy every instruction line in view:
    start target, include targets, the task value, end target --
    at most seven words, below every count limit the generator
    can draw), ``first_only`` (satisfy only Instruction 1 plus
    the task value -- the density failure the decision must
    catch), ``wrong`` (never satisfy anything), ``explode``
    (generate raises).
    """

    fingerprint = "sha256:test-density"

    def __init__(self, mode: str = "perfect", cancel_after: int | None = None):
        self.mode = mode
        self.cancel_after = cancel_after
        self.cancel_event: threading.Event | None = None
        self.calls = 0
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

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        """Generate."""
        self.calls += 1
        if self.mode == "explode":
            raise RuntimeError("boom")
        if (
            self.cancel_after is not None
            and self.cancel_event is not None
            and self.calls >= self.cancel_after
        ):
            self.cancel_event.set()
        text = self.decode(list(input_ids))
        fact = _FACT_RE.search(text)
        value = fact.group(2) if fact else "unknown"
        if self.mode == "wrong":
            answer = "unknown"
        elif self.mode == "first_only":
            first = text.split("Instruction 1:")[-1].split("Instruction 2:")[0]
            words: list[str] = []
            start = _START_RE.search(first)
            if start:
                words.append(start.group(1))
            words.extend(_INCLUDE_RE.findall(first))
            words.append(value)
            end = _END_RE.search(first)
            if end:
                words.append(end.group(1))
            answer = " ".join(words)
        else:
            words = []
            start = _START_RE.search(text)
            if start:
                words.append(start.group(1))
            words.extend(_INCLUDE_RE.findall(text))
            words.append(value)
            end = _END_RE.search(text)
            if end:
                words.append(end.group(1))
            answer = " ".join(words)
        boundary = self.encode("endmark")[0]
        out = list(self.encode(answer)) + [boundary]
        out = (out + [boundary] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


def _payload(backend=None, formats=FORMATS, seed=0, **stage_kwargs):
    """Payload."""
    backend = backend if backend is not None else _DensityBackend()
    stage_kwargs.setdefault("max_context", 8192)
    stage = experiments.density_stage(
        backend,
        formats,
        boundary=tuple(backend.encode("endmark")),
        seed=seed,
        **stage_kwargs,
    )
    return experiments.density_analysis(stage, tuple(f.name for f in formats))


def _record(case_index, arm, hashes, **overrides):
    """Record."""
    record = {
        "candidate": "x",
        "case_index": case_index,
        "arm": arm,
        "detail": "",
        "case_screened": False,
        "boundary_found": True,
        "answer_digest": density.digest_text("answer"),
        "word_hashes": list(hashes),
        "word_list_digest": density.commitment_digest(list(hashes)),
        "word_count": len(hashes),
        "input_tokens": 10,
        "content_digest": "0" * 64,
    }
    record.update(overrides)
    return record


class TestProtocol(unittest.TestCase):
    """Test protocol."""

    def test_constants(self):
        """Test constants."""
        self.assertEqual(density.CASE_COUNT, 20)
        self.assertEqual(density.CONSTRAINT_COUNT, 16)
        self.assertEqual(len(density.ARM_KEYS), 21)
        self.assertNotIn("k1", density.ARM_KEYS)  # B-F1: k1 merged into solo_1
        self.assertEqual(density.ARM_KEYS[0], "solo_1")
        self.assertEqual(density.ARM_KEYS[-1], "sham")
        self.assertEqual(density.MAX_DENSITY_RECORDS, 21 * 20 * 24)
        self.assertEqual(density.MAX_DENSITY_PAYLOAD_BYTES, 52 * 1024 * 1024)
        self.assertEqual(density.MIN_DISCORDANT_CASES, 6)
        self.assertEqual(density.ALPHA, 0.05)
        self.assertEqual(
            density.RECORD_DETAILS, ("", "arm_evaluation_failed", "arm_context_exceeded")
        )
        self.assertEqual(
            density.STAGE_DETAILS,
            ("cancelled", "density_record_budget_exceeded", "density_screen_failed"),
        )

    def test_protocol_object_and_fingerprint(self):
        """Test protocol object and fingerprint."""
        protocol = density.protocol_object(3, "sha256:x")
        self.assertEqual(protocol["backend_fingerprint"], "sha256:x")
        self.assertEqual(protocol["seed"], 3)
        self.assertEqual(protocol["arm_keys"], list(density.ARM_KEYS))
        fingerprint = density.protocol_fingerprint(3, "sha256:x")
        self.assertEqual(len(fingerprint), 64)
        self.assertNotEqual(fingerprint, density.protocol_fingerprint(4, "sha256:x"))
        self.assertNotEqual(fingerprint, density.protocol_fingerprint(3, "sha256:y"))


class TestGeneration(unittest.TestCase):
    """Test generation."""

    def test_determinism(self):
        """Test determinism."""
        self.assertEqual(density.generate_cases(7), density.generate_cases(7))
        self.assertNotEqual(density.ground_sha256(7), density.ground_sha256(8))

    def test_pool_hygiene(self):
        """Test pool hygiene."""
        topic_words: set[str] = set()
        for topic in density.TOPICS:
            topic_words.update(density.normalize_words(topic))
            self.assertEqual(len(density.normalize_words(topic)), 2)
        self.assertEqual(len(set(density.TOPICS)), len(density.TOPICS))
        for topic in density.TOPICS:
            for other in density.TOPICS:
                if topic != other:
                    self.assertNotIn(topic, other)
        self.assertFalse(set(density.WORDS) & topic_words)
        self.assertFalse(set(density.TARGET_WORDS) & topic_words)
        self.assertFalse(set(density.WORDS) & set(density.TARGET_WORDS))
        self.assertEqual(len(set(density.WORDS)), len(density.WORDS))
        self.assertEqual(len(set(density.TARGET_WORDS)), len(density.TARGET_WORDS))
        pool = set(density.WORDS) | set(density.TARGET_WORDS) | topic_words
        for statement in density.NEUTRAL_STATEMENTS:
            self.assertFalse(set(density.normalize_words(statement)) & pool, statement)

    def test_kind_balance(self):
        """Test kind balance."""
        # Each position carries each kind exactly CASE_COUNT/4
        # times across the case set (design section 3).
        for position in range(1, density.CONSTRAINT_COUNT + 1):
            kinds = [density.kind_at(i, position) for i in range(density.CASE_COUNT)]
            for kind in density.KINDS:
                self.assertEqual(kinds.count(kind), density.CASE_COUNT // 4)

    def test_satisfiability_invariant(self):
        """Test satisfiability invariant."""
        # Design section 3 / red-team F1: by construction a
        # perfect follower passes every constraint of every
        # case; pin it over a seed sweep, not one seed.
        for seed in range(50):
            for case in density.generate_cases(seed):
                constraints = case["constraints"]
                starts = {c["target"] for c in constraints if c["kind"] == "start"}
                ends = {c["target"] for c in constraints if c["kind"] == "end"}
                includes = [c["target"] for c in constraints if c["kind"] == "include"]
                limits = [c["limit"] for c in constraints if c["kind"] == "count"]
                self.assertEqual(len(starts), 1)
                self.assertEqual(len(ends), 1)
                self.assertEqual(len(set(includes)), 4)
                distinct = set(includes) | starts | ends
                self.assertEqual(len(distinct), 6)
                self.assertNotIn(case["value"], distinct)
                self.assertEqual(sorted(limits), sorted(density.COUNT_PARAMS))
                self.assertGreater(min(limits), len(distinct))
                answer = " ".join(
                    [next(iter(starts))] + includes + [case["value"], next(iter(ends))]
                )
                hashes = density.word_hashes(answer)
                self.assertTrue(all(density.satisfied(c, hashes) for c in constraints))
                self.assertTrue(density.task_present(case, hashes))

    def test_position_one_target_not_in_neutrals(self):
        """Test position one target not in neutrals."""
        # The sham-neutrality precondition (design section 3):
        # position 1's word target never occurs in the case's
        # neutral statements.
        for seed in range(50):
            for case in density.generate_cases(seed):
                neutral_words: set[str] = set()
                for statement in case["neutrals"]:
                    neutral_words.update(density.normalize_words(statement))
                first = case["constraints"][0]
                if first["kind"] != "count":
                    self.assertNotIn(first["target"], neutral_words)

    def test_sham_block_reaches_k16_length(self):
        """Test sham block reaches k16 length."""
        for seed in range(20):
            for case in density.generate_cases(seed):
                self.assertGreaterEqual(
                    density.arm_block_words(case, "sham"),
                    density.arm_block_words(case, "k16"),
                )

    def test_arm_content_labels(self):
        """Test arm content labels."""
        case = density.generate_cases(0)[0]
        solo = density.arm_content(case, "solo_7")
        self.assertIn("Instruction 7:", solo)
        self.assertNotIn("Instruction 8:", solo)
        full = density.arm_content(case, "k16")
        for position in range(1, 17):
            self.assertIn(f"Instruction {position}:", full)
        sham = density.arm_content(case, "sham")
        self.assertIn("Instruction 1:", sham)
        self.assertNotIn("Instruction 2:", sham)


class TestChecking(unittest.TestCase):
    """Test checking."""

    def test_normalization_table(self):
        """Test normalization table."""
        # B-F3: emphasis, brackets, and separators never hide
        # or invent words; digits are different words from
        # their spelled forms.
        self.assertEqual(density.normalize_words("**alpha**"), ["alpha"])
        self.assertEqual(density.normalize_words("[beta]"), ["beta"])
        self.assertEqual(density.normalize_words("alpha/beta"), ["alpha", "beta"])
        self.assertEqual(density.normalize_words("---"), [])
        self.assertEqual(density.normalize_words("..."), [])
        self.assertEqual(density.normalize_words("well-known"), ["well", "known"])
        self.assertEqual(density.normalize_words("ALPHA Alpha"), ["alpha", "alpha"])
        self.assertEqual(density.normalize_words(""), [])

    def test_satisfied_per_kind(self):
        """Test satisfied per kind."""
        hashes = density.word_hashes("red blue green")
        self.assertTrue(density.satisfied({"kind": "include", "target": "blue"}, hashes))
        self.assertFalse(density.satisfied({"kind": "include", "target": "gold"}, hashes))
        self.assertTrue(density.satisfied({"kind": "start", "target": "red"}, hashes))
        self.assertFalse(density.satisfied({"kind": "start", "target": "blue"}, hashes))
        self.assertTrue(density.satisfied({"kind": "end", "target": "green"}, hashes))
        self.assertFalse(density.satisfied({"kind": "end", "target": "blue"}, hashes))
        self.assertTrue(density.satisfied({"kind": "count", "limit": 3}, hashes))
        self.assertFalse(density.satisfied({"kind": "count", "limit": 2}, hashes))
        empty: list[str] = []
        self.assertFalse(density.satisfied({"kind": "start", "target": "red"}, empty))
        self.assertFalse(density.satisfied({"kind": "end", "target": "red"}, empty))
        self.assertTrue(density.satisfied({"kind": "count", "limit": 8}, empty))

    def test_digit_is_not_the_number_word(self):
        """Test digit is not the number word."""
        # The frozen rule counts "7" and "seven" as different
        # words (design section 4, stated measurement choice).
        hashes = density.word_hashes("answer 7")
        self.assertFalse(density.satisfied({"kind": "include", "target": "seven"}, hashes))
        self.assertTrue(density.satisfied({"kind": "include", "target": "7"}, hashes))

    def test_commitment_order_and_multiplicity(self):
        """Test commitment order and multiplicity."""
        forward = density.word_hashes("red blue")
        reverse = density.word_hashes("blue red")
        self.assertNotEqual(density.commitment_digest(forward), density.commitment_digest(reverse))
        doubled = density.word_hashes("red red")
        single = density.word_hashes("red")
        derived = density.derive_from_hashes(doubled)
        self.assertEqual(derived["word_count"], 2)
        self.assertEqual(derived["first"], single[0])
        self.assertEqual(derived["last"], single[0])
        # A swapped first hash flips a start verdict: the
        # forgery class the sequence commitment exists for.
        case_hashes = density.word_hashes("red blue")
        forged = [density.digest_text("gold"), case_hashes[1]]
        self.assertFalse(density.satisfied({"kind": "start", "target": "red"}, forged))
        self.assertTrue(density.satisfied({"kind": "start", "target": "red"}, case_hashes))


class TestSignTest(unittest.TestCase):
    """Test sign test."""

    def test_exact_values(self):
        """Test exact values."""
        self.assertEqual(density.sign_test_p(0, 0), 1.0)
        self.assertEqual(density.sign_test_p(6, 0), 0.03125)
        self.assertEqual(density.sign_test_p(5, 1), 0.21875)
        self.assertEqual(density.sign_test_p(3, 3), 1.0)
        self.assertAlmostEqual(density.sign_test_p(10, 2), 0.03857421875, places=12)

    def test_symmetry_and_monotonicity(self):
        """Test symmetry and monotonicity."""
        self.assertEqual(density.sign_test_p(2, 9), density.sign_test_p(9, 2))
        self.assertLess(density.sign_test_p(9, 1), density.sign_test_p(6, 4))


def _perfect_hashes(case):
    """Perfect hashes."""
    starts = [c["target"] for c in case["constraints"] if c["kind"] == "start"][0]
    ends = [c["target"] for c in case["constraints"] if c["kind"] == "end"][0]
    includes = [c["target"] for c in case["constraints"] if c["kind"] == "include"]
    return density.word_hashes(" ".join([starts] + includes + [case["value"], ends]))


def _first_only_hashes(case):
    """First only hashes."""
    first = case["constraints"][0]
    words: list[str] = []
    if first["kind"] == "start":
        words.append(first["target"])
    if first["kind"] == "include":
        words.append(first["target"])
    words.append(case["value"])
    if first["kind"] == "end":
        words.append(first["target"])
    return density.word_hashes(" ".join(words))


def _synthetic_records(cases, k16_hashes, solo_hashes=None):
    """Synthetic records."""
    records = []
    for ci, case in enumerate(cases):
        for arm in density.ARM_KEYS:
            if arm == "k16":
                hashes = k16_hashes(case)
            elif arm.startswith("solo_") and solo_hashes is not None:
                hashes = solo_hashes(case, int(arm.split("_")[1]))
            else:
                hashes = _perfect_hashes(case)
            records.append(_record(ci, arm, hashes))
    return records


class TestAggregate(unittest.TestCase):
    """Test aggregate."""

    def setUp(self):
        """Set up the test fixture."""
        self.cases = density.generate_cases(4)

    def test_perfect_model(self):
        """Test perfect model."""
        records = _synthetic_records(self.cases, _perfect_hashes)
        agg = density.aggregate_candidate(records, self.cases)
        self.assertTrue(agg["available"])
        self.assertEqual(agg["n_live"], 420)
        self.assertEqual(agg["joint_k16"]["rate"], 1.0)
        self.assertTrue(all(row["rate"] == 1.0 for row in agg["by_density"]))
        # No discordance at all: the verdict is withheld as
        # insufficient, never defaulted to acquittal (design
        # section 7).
        self.assertEqual(agg["decision"]["verdict"], "insufficient")
        self.assertIsNone(agg["decision"]["falsified"])
        self.assertEqual(agg["decision"]["f"], 0)
        self.assertEqual(agg["decision"]["discordant_cases"], 0)

    def test_first_only_model_is_falsified(self):
        """Test first only model is falsified."""
        records = _synthetic_records(self.cases, _first_only_hashes)
        agg = density.aggregate_candidate(records, self.cases)
        self.assertEqual(agg["decision"]["verdict"], "falsified")
        self.assertIs(agg["decision"]["falsified"], True)
        self.assertGreater(agg["decision"]["f"], agg["decision"]["p"])
        self.assertLessEqual(agg["decision"]["sign_p"], 0.05)
        self.assertLess(agg["joint_k16"]["rate"], 1.0)

    def test_insufficient_below_six_discordant_cases(self):
        """Test insufficient below six discordant cases."""

        def k16(case):
            """K16."""
            index = self.cases.index(case)
            return _first_only_hashes(case) if index < 3 else _perfect_hashes(case)

        records = _synthetic_records(self.cases, k16)
        agg = density.aggregate_candidate(records, self.cases)
        self.assertEqual(agg["decision"]["discordant_cases"], 3)
        self.assertEqual(agg["decision"]["verdict"], "insufficient")
        self.assertIsNone(agg["decision"]["falsified"])

    def test_zero_word_records_excluded_from_count_denominators(self):
        """Test zero word records excluded from count denominators."""
        records = _synthetic_records(self.cases, lambda case: [])
        agg = density.aggregate_candidate(records, self.cases)
        self.assertEqual(agg["n_zero_word"], 20)
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        count_row = next(r for r in agg["by_kind"] if r["kind"] == "count")
        self.assertEqual(count_row["k16_checked"], 0)
        self.assertIsNone(count_row["k16_rate"])
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        include_row = next(r for r in agg["by_kind"] if r["kind"] == "include")
        self.assertEqual(include_row["k16_checked"], 80)

    def test_screen_accounting_is_disjoint(self):
        """Test screen accounting is disjoint."""
        records = _synthetic_records(self.cases, _perfect_hashes)
        for record in records[:21]:
            record["case_screened"] = True
            record["boundary_found"] = False
        for record in records[21:42]:
            record["detail"] = "arm_evaluation_failed"
            record["boundary_found"] = False
        for record in records[42:63]:
            record["boundary_found"] = False
        agg = density.aggregate_candidate(records, self.cases)
        self.assertEqual(agg["n_generated"], 420)
        self.assertEqual(agg["n_boundary_screened"], 21)
        self.assertEqual(agg["n_detail_screened"], 21)
        self.assertEqual(agg["n_no_boundary"], 21)
        self.assertEqual(
            agg["n_generated"],
            agg["n_boundary_screened"]
            + agg["n_detail_screened"]
            + agg["n_no_boundary"]
            + agg["n_live"],
        )

    def test_no_live_records_unavailable(self):
        """Test no live records unavailable."""
        agg = density.aggregate_candidate([], self.cases)
        self.assertFalse(agg["available"])
        self.assertEqual(agg["detail"], "density_unavailable")
        self.assertIsNone(agg["joint_k16"]["rate"])
        self.assertEqual(agg["decision"]["verdict"], "insufficient")


class TestStage(unittest.TestCase):
    """Test stage."""

    def test_perfect_backend_payload(self):
        """Test perfect backend payload."""
        payload = _payload()
        self.assertTrue(payload["available"])
        self.assertTrue(_valid_density(payload))
        for entry in payload["candidates"].values():
            self.assertEqual(entry["n_live"], 420)
            self.assertEqual(entry["joint_k16"]["rate"], 1.0)
            self.assertEqual(entry["decision"]["verdict"], "insufficient")
            for row in entry["task_presence"]:
                self.assertEqual(row["rate"], 1.0)

    def test_first_only_backend_is_falsified(self):
        """Test first only backend is falsified."""
        payload = _payload(_DensityBackend("first_only"))
        self.assertTrue(_valid_density(payload))
        for entry in payload["candidates"].values():
            self.assertEqual(entry["decision"]["verdict"], "falsified")
            self.assertIs(entry["decision"]["falsified"], True)

    def test_wrong_backend(self):
        """Test wrong backend."""
        payload = _payload(_DensityBackend("wrong"))
        self.assertTrue(_valid_density(payload))
        for entry in payload["candidates"].values():
            self.assertEqual(entry["joint_k16"]["rate"], 0.0)
            presence = {row["arm"]: row["rate"] for row in entry["task_presence"]}
            self.assertEqual(presence["k16"], 0.0)

    def test_explode_backend_degrades_records(self):
        """Test explode backend degrades records."""
        payload = _payload(_DensityBackend("explode"))
        self.assertTrue(_valid_density(payload))
        for entry in payload["candidates"].values():
            self.assertFalse(entry["available"])
            self.assertEqual(entry["detail"], "density_unavailable")
            self.assertEqual(entry["n_detail_screened"], 420)
            self.assertEqual(entry["n_live"], 0)

    def test_cancel_unwinds_stage(self):
        """Test cancel unwinds stage."""
        backend = _DensityBackend(cancel_after=5)
        event = threading.Event()
        backend.cancel_event = event
        payload = _payload(backend, cancel=event)
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "cancelled")

    def test_record_budget(self):
        """Test record budget."""
        formats = tuple(
            Format(name=f"dn-{i}", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>")
            for i in range(25)
        )
        backend = _DensityBackend()
        stage = experiments.density_stage(
            backend,
            formats,
            boundary=tuple(backend.encode("endmark")),
            max_context=8192,
            seed=0,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "density_record_budget_exceeded")

    def test_boundary_screen_fails_stage_closed(self):
        """Test boundary screen fails stage closed."""

        class _ScreenedBackend(_DensityBackend):
            """Screened backend."""

            def decode(self, ids):
                """Decode."""
                if tuple(ids) == self.boundary_ids:
                    return "signal word"
                return super().decode(ids)

        backend = _ScreenedBackend()
        backend.boundary_ids = tuple(backend.encode("endmark"))
        stage = experiments.density_stage(
            backend,
            FORMATS,
            boundary=backend.boundary_ids,
            max_context=8192,
            seed=0,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "density_screen_failed")

    def test_duplicate_format_names_rejected(self):
        """Test duplicate format names rejected."""
        backend = _DensityBackend()
        formats = (FORMATS[0], FORMATS[0])
        with self.assertRaises(ValueError):
            experiments.density_stage(
                backend,
                formats,
                boundary=tuple(backend.encode("endmark")),
                max_context=8192,
                seed=0,
            )


class TestDiff(unittest.TestCase):
    """Test diff."""

    def test_self_diff_unchanged(self):
        """Test self diff unchanged."""
        payload = _payload()
        result = diff_density(payload, payload)
        self.assertEqual(result["available"], "both")
        self.assertFalse(result["changed"])
        self.assertTrue(result["comparable"])

    def test_verdict_drift_detected(self):
        """Test verdict drift detected."""
        old = _payload()
        new = _payload(_DensityBackend("first_only"))
        result = diff_density(old, new)
        self.assertEqual(result["available"], "both")
        self.assertTrue(result["changed"])
        metrics = {entry["metric"] for entry in result["candidate_drift"]}
        self.assertIn("verdict", metrics)
        self.assertIn("falsified", metrics)

    def test_one_sided(self):
        """Test one sided."""
        payload = _payload()
        self.assertEqual(diff_density(payload, None)["available"], "old")
        self.assertEqual(diff_density(None, payload)["available"], "new")
        self.assertEqual(diff_density(None, None)["available"], "neither")
        self.assertFalse(diff_density(payload, None)["changed"])

    def test_seed_drift_not_comparable(self):
        """Test seed drift not comparable."""
        old = _payload(seed=0)
        new = _payload(seed=1)
        result = diff_density(old, new)
        self.assertTrue(result["generator_drift"])
        self.assertFalse(result["comparable"])
        self.assertTrue(result["changed"])

    def test_tamper_battery(self):
        """Test tamper battery."""
        payload = _payload()

        def forged(mutate):
            """Forged."""
            candidate = copy.deepcopy(payload)
            mutate(candidate)
            return candidate

        cases = {
            "aggregate_number": lambda p: p["candidates"]["dn-a"]["decision"].update(f=999),
            "verdict_word": lambda p: p["candidates"]["dn-a"]["decision"].update(
                verdict="not_falsified"
            ),
            "protocol_fingerprint": lambda p: p["protocol"].update(fingerprint="0" * 64),
            "record_order": lambda p: p["candidates"]["dn-a"]["records"].reverse(),
            "commitment_swap": lambda p: p["candidates"]["dn-a"]["records"][0].update(
                word_hashes=density.word_hashes("gold gold gold"),
                word_count=3,
                word_list_digest=density.commitment_digest(density.word_hashes("gold gold gold")),
            ),
            "content_digest": lambda p: p["candidates"]["dn-a"]["records"][5].update(
                content_digest="0" * 64
            ),
            "truncated_records": lambda p: p["candidates"]["dn-a"].update(
                records=p["candidates"]["dn-a"]["records"][:-1]
            ),
            "screened_with_answer": lambda p: p["candidates"]["dn-a"]["records"][0].update(
                case_screened=True
            ),
            "joint_rate": lambda p: p["candidates"]["dn-a"]["joint_k16"].update(rate=0.5),
        }
        for name, mutate in cases.items():
            with self.subTest(tamper=name):
                self.assertFalse(_valid_density(forged(mutate)))


class TestReportingAndCli(unittest.TestCase):
    """Test reporting and cli."""

    def test_section_renders(self):
        """Test section renders."""
        payload = _payload(_DensityBackend("first_only"))
        html = render_html({"density": payload})
        self.assertIn("Instruction density", html)
        self.assertIn("density falsified", html)
        self.assertIn("Satisfaction by stack depth", html)
        self.assertIn("Sham (first constraint plus length-matched", html)

    def test_section_absent_without_payload(self):
        """Test section absent without payload."""
        self.assertNotIn('id="density"', render_html({}))
        self.assertNotIn('id="density"', render_html({"density": {"available": False}}))

    def test_section_hostile_payload(self):
        """Test section hostile payload."""
        payload = _payload()
        hostile = {"density": copy.deepcopy(payload)}
        hostile["density"]["candidates"] = {
            "<script>alert(1)</script>": {"status": "available"},
        }
        html = render_html(hostile)
        self.assertNotIn("<script>alert(1)</script>", html)

    def test_section_na_rates(self):
        """Test section na rates."""
        payload = _payload()
        sparse = copy.deepcopy(payload)
        entry = sparse["candidates"]["dn-a"]
        entry["by_density"] = [{"density": 1}]
        entry["joint_k16"] = {}
        entry["position_bands"] = []
        entry["by_kind"] = [{"kind": "include"}]
        entry["task_presence"] = []
        entry["sham"] = {}
        entry["decision"] = {}
        html = render_html({"density": sparse})
        self.assertIn("Instruction density", html)

    def test_cli_helper_attaches_payload(self):
        """Test cli helper attaches payload."""
        backend = _DensityBackend()
        result: dict = {}
        cli._run_density_probe(
            backend,
            FORMATS,
            result,
            boundary=tuple(backend.encode("endmark")),
            boundary_map=None,
            max_new_tokens=64,
            max_context=8192,
            seed=0,
            cancel=None,
        )
        self.assertTrue(result["density"]["available"])
        self.assertTrue(_valid_density(result["density"]))

    def test_cli_helper_never_raises(self):
        """Test cli helper never raises."""

        class _ExplodingEncode(_DensityBackend):
            """Exploding encode."""

            @staticmethod
            def encode(text):
                """Encode."""
                raise RuntimeError("no encode")

        result: dict = {}
        cli._run_density_probe(
            _ExplodingEncode(),
            FORMATS,
            result,
            boundary=(1,),
            boundary_map=None,
            max_new_tokens=64,
            max_context=8192,
            seed=0,
            cancel=None,
        )
        # The stage fails closed on its own (screen) -- a
        # legitimate never-raises outcome.
        self.assertFalse(result["density"]["available"])

        class _ExplodingFingerprint(_DensityBackend):
            """Exploding fingerprint."""

            @property
            def fingerprint(self):  # noqa: ANN202 -- test fake
                raise RuntimeError("no fingerprint")

        result = {}
        cli._run_density_probe(
            _ExplodingFingerprint(),
            FORMATS,
            result,
            boundary=(1,),
            boundary_map=None,
            max_new_tokens=64,
            max_context=8192,
            seed=0,
            cancel=None,
        )
        # A stage-construction failure lands in the wrapper's
        # fail-closed fallback instead.
        self.assertEqual(result["density"], {"available": False, "detail": "density_unavailable"})


if __name__ == "__main__":
    unittest.main()


# A neutral pool that violates the shipped pool-hygiene pin on
# purpose (every statement contains three target words), used to
# prove the generator's position-1 redraw branch -- unreachable
# while the pin holds, kept as the enforcement for a future pool
# edit (the cycle-171 rule: a guard is removed or proven with a
# stub, never left as decoration).
_PATCHED_NEUTRALS = tuple(f"candle mirror bridge item{i} rests quietly today" for i in range(24))


class TestGapCoverage(unittest.TestCase):
    """Test gap coverage."""

    def test_generator_rejects_non_int_seed(self):
        """Test generator rejects non int seed."""
        with self.assertRaises(ValueError):
            density.generate_cases("0")  # type: ignore[arg-type]

    def test_unknown_arm_and_kind_raise(self):
        """Test unknown arm and kind raise."""
        case = density.generate_cases(0)[0]
        with self.assertRaises(ValueError):
            density.arm_content(case, "bogus")
        with self.assertRaises(ValueError):
            density.satisfied({"kind": "bogus", "target": "x"}, [])

    def test_position_one_redraw_with_patched_pool(self):
        """Test position one redraw with patched pool."""
        # Seeds/cases found by executing the draw sequence with
        # the patched pool: at each, the naive position-1 target
        # is "mirror" (inside the case's neutrals), and the
        # generator must have redrawn it.
        expected = {(2, 0, "include"), (4, 1, "start"), (8, 18, "end")}
        for seed, case_index, kind in expected:
            with self.subTest(seed=seed, kind=kind):
                with mock.patch.object(density, "NEUTRAL_STATEMENTS", _PATCHED_NEUTRALS):
                    cases = density.generate_cases(seed)
                    first = cases[case_index]["constraints"][0]
                    self.assertEqual(first["kind"], kind)
                    self.assertNotEqual(first["target"], "mirror")
                    self.assertNotIn(first["target"], {"candle", "mirror", "bridge"})
                with (
                    mock.patch.object(density, "NEUTRAL_STATEMENTS", _PATCHED_NEUTRALS),
                    mock.patch.object(diffreports, "_DENSITY_NEUTRALS", _PATCHED_NEUTRALS),
                ):
                    self.assertEqual(diffreports._density_generate(seed), cases)

    def test_zero_word_solo_records_leave_count_denominators(self):
        """Test zero word solo records leave count denominators."""
        cases = density.generate_cases(4)
        records = _synthetic_records(cases, _perfect_hashes, solo_hashes=lambda case, j: [])
        agg = density.aggregate_candidate(records, cases)
        self.assertEqual(agg["n_zero_word"], 320)
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        count_row = next(r for r in agg["by_kind"] if r["kind"] == "count")
        self.assertEqual(count_row["solo_checked"], 0)
        self.assertIsNone(count_row["solo_rate"])
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        include_row = next(r for r in agg["by_kind"] if r["kind"] == "include")
        self.assertEqual(include_row["solo_checked"], 80)
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        k16_row = next(r for r in agg["by_density"] if r["density"] == 16)
        self.assertEqual(k16_row["solo_checked"], 240)

    def test_reverse_discordance_is_not_falsified(self):
        """Test reverse discordance is not falsified."""
        cases = density.generate_cases(4)
        records = _synthetic_records(
            cases,
            _perfect_hashes,
            solo_hashes=lambda case, j: density.word_hashes(case["value"]),
        )
        agg = density.aggregate_candidate(records, cases)
        self.assertEqual(agg["decision"]["f"], 0)
        self.assertEqual(agg["decision"]["p"], 240)
        self.assertEqual(agg["decision"]["cases_negative"], 20)
        self.assertEqual(agg["decision"]["verdict"], "not_falsified")
        self.assertIs(agg["decision"]["falsified"], False)


class TestStageGaps(unittest.TestCase):
    """Test stage gaps."""

    def test_formats_must_be_a_nonempty_tuple(self):
        """Test formats must be a nonempty tuple."""
        backend = _DensityBackend()
        with self.assertRaises(ValueError):
            experiments.density_stage(
                backend,
                [FORMATS[0]],  # type: ignore[arg-type]
                boundary=tuple(backend.encode("endmark")),
                max_context=8192,
                seed=0,
            )

    def test_context_exceeded_records(self):
        """Test context exceeded records."""
        payload = _payload(max_context=1)
        self.assertTrue(_valid_density(payload))
        for entry in payload["candidates"].values():
            self.assertFalse(entry["available"])
            self.assertEqual(entry["n_detail_screened"], 420)
            self.assertEqual(entry["n_live"], 0)
            details = {r["detail"] for r in entry["records"]}
            self.assertEqual(details, {"arm_context_exceeded"})

    def test_undecodable_boundary_proceeds(self):
        """Test undecodable boundary proceeds."""

        class _Undecodable(_DensityBackend):
            """Undecodable."""

            def decode(self, ids):
                """Decode."""
                if tuple(ids) == self.boundary_ids:
                    raise RuntimeError("no decode")
                return super().decode(ids)

        backend = _Undecodable()
        backend.boundary_ids = tuple(backend.encode("endmark"))
        payload = _payload(backend)
        self.assertTrue(payload["available"])
        self.assertTrue(_valid_density(payload))
        for entry in payload["candidates"].values():
            self.assertEqual(entry["joint_k16"]["rate"], 1.0)

    def test_per_case_boundary_screen(self):
        """Test per case boundary screen."""

        class _CaseScreened(_DensityBackend):
            """Case screened."""

            def decode(self, ids):
                """Decode."""
                if tuple(ids) == self.boundary_ids:
                    return "Instruction 7:"
                return super().decode(ids)

        backend = _CaseScreened()
        backend.boundary_ids = tuple(backend.encode("endmark"))
        payload = _payload(backend)
        self.assertTrue(_valid_density(payload))
        for entry in payload["candidates"].values():
            self.assertFalse(entry["available"])
            self.assertEqual(entry["n_boundary_screened"], 420)
            self.assertEqual(entry["n_live"], 0)
            self.assertTrue(all(r["case_screened"] for r in entry["records"]))

    def test_cell_context_and_cancel_directly(self):
        """Test cell context and cancel directly."""
        from model_protocol_lab.core import Fixture, Message

        backend = _DensityBackend()
        fixture = Fixture(
            id="density-00-solo_1",
            messages=(Message("user", "hi"),),
            expected="ochre",
            group="density",
        )
        detail, fields = experiments._density_cell(
            backend,
            FORMATS[0],
            fixture,
            boundary_ids=(1,),
            max_new_tokens=64,
            max_context=10,
            input_tokens=50,
            seed=0,
            cancel=None,
        )
        self.assertEqual(detail, "arm_context_exceeded")
        self.assertIsNone(fields)
        event = threading.Event()
        event.set()
        with self.assertRaises(experiments._DensityCancelled):
            experiments._density_cell(
                backend,
                FORMATS[0],
                fixture,
                boundary_ids=(1,),
                max_new_tokens=64,
                max_context=8192,
                input_tokens=1,
                seed=0,
                cancel=event,
            )

    def _event_backend(self, fire_on_call: int):
        """Event backend."""

        class _EventOnEncode(_DensityBackend):
            """Event on encode."""

            def encode(self, text):
                """Encode."""
                ids = super().encode(text)
                if self.calls_encode == fire_on_call:
                    self.event.set()
                self.calls_encode += 1
                return ids

        backend = _EventOnEncode()
        backend.calls_encode = 0
        backend.event = threading.Event()
        return backend

    def test_encode_event_cancels_at_arm_check(self):
        """Test encode event cancels at arm check."""
        # The boundary encode in this test is call 0; stage arm
        # encodes are calls 1..21 per case. Firing on call 3
        # (arm 3 of case 0, whose cell then returns
        # context-exceeded without a cancel check) makes the
        # stage's own per-arm cancel check observe the event.
        backend = self._event_backend(3)
        boundary = tuple(backend.encode("endmark"))
        stage = experiments.density_stage(
            backend,
            FORMATS,
            boundary=boundary,
            max_context=1,
            seed=0,
            cancel=backend.event,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")

    def test_encode_event_cancels_at_case_boundary(self):
        """Test encode event cancels at case boundary."""
        # Firing on call 21 (the last arm encode of case 0)
        # leaves the event for the stage's per-case check.
        backend = self._event_backend(21)
        boundary = tuple(backend.encode("endmark"))
        stage = experiments.density_stage(
            backend,
            FORMATS,
            boundary=boundary,
            max_context=1,
            seed=0,
            cancel=backend.event,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")

    def test_analysis_unavailable_candidate(self):
        """Test analysis unavailable candidate."""
        backend = _DensityBackend()
        stage = experiments.density_stage(
            backend,
            FORMATS,
            boundary=tuple(backend.encode("endmark")),
            max_context=8192,
            seed=0,
        )
        stage["candidates"]["dn-b"] = {
            "status": "unavailable",
            "detail": "candidate_evaluation_failed",
            "records": [],
        }
        payload = experiments.density_analysis(stage, tuple(f.name for f in FORMATS))
        self.assertEqual(
            payload["candidates"]["dn-b"],
            {"status": "unavailable", "detail": "candidate_evaluation_failed", "records": []},
        )
        self.assertTrue(_valid_density(payload))


class TestDiffTamperRoundTwo(unittest.TestCase):
    """Test diff tamper round two."""

    def test_tamper_battery_round_two(self):
        """Test tamper battery round two."""
        payload = _payload()

        def forged(mutate):
            """Forged."""
            candidate = copy.deepcopy(payload)
            mutate(candidate)
            return candidate

        cases = {
            "record_missing_key": lambda p: p["candidates"]["dn-a"]["records"][0].pop("arm"),
            "record_candidate_field": lambda p: p["candidates"]["dn-a"]["records"][0].update(
                candidate="dn-b"
            ),
            "screened_with_boundary": lambda p: p["candidates"]["dn-a"]["records"][0].update(
                case_screened=True, boundary_found=True
            ),
            "malformed_answer_digest": lambda p: p["candidates"]["dn-a"]["records"][0].update(
                answer_digest="zz" * 32
            ),
            "negative_input_tokens": lambda p: p["candidates"]["dn-a"]["records"][0].update(
                input_tokens=-1
            ),
            "bool_input_tokens": lambda p: p["candidates"]["dn-a"]["records"][0].update(
                input_tokens=True
            ),
            "bool_seed": lambda p: p["protocol"].update(seed=True),
            "extra_entry_key": lambda p: p["candidates"]["dn-a"].update(extra=1),
            "unavailable_with_records": lambda p: p["candidates"].__setitem__(
                "dn-b",
                {
                    "status": "unavailable",
                    "detail": "x",
                    "records": p["candidates"]["dn-b"]["records"][:1],
                },
            ),
            "bad_detail_word": lambda p: p["candidates"]["dn-a"]["records"][0].update(
                detail="bogus_detail"
            ),
            "word_count_mismatch": lambda p: p["candidates"]["dn-a"]["records"][0].update(
                word_count=99
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(tamper=name):
                self.assertFalse(_valid_density(forged(mutate)))


class TestReportingGaps(unittest.TestCase):
    """Test reporting gaps."""

    def test_section_candidates_not_a_dict(self):
        """Test section candidates not a dict."""
        html = render_html({"density": {"available": True, "candidates": []}})
        self.assertNotIn('id="density"', html)

    def test_section_junk_entries(self):
        """Test section junk entries."""
        payload = _payload()
        mixed = copy.deepcopy(payload)
        mixed["candidates"] = {
            "junk": 42,
            "off": {"status": "unavailable"},
            "dn-a": payload["candidates"]["dn-a"],
        }
        html = render_html({"density": mixed})
        self.assertIn("Instruction density", html)
        self.assertIn("unavailable", html)

    def test_section_non_dict_rows(self):
        """Test section non dict rows."""
        payload = _payload()
        sparse = copy.deepcopy(payload)
        entry = sparse["candidates"]["dn-a"]
        entry["by_density"] = [42, {"density": 2}]
        entry["position_bands"] = [42]
        entry["by_kind"] = [42, {"kind": "include"}]
        entry["task_presence"] = [42]
        html = render_html({"density": sparse})
        self.assertIn("Instruction density", html)


class TestFinalGaps(unittest.TestCase):
    """Test final gaps."""

    def test_screen_audit_findings_fail_closed(self):
        """Test screen audit findings fail closed."""
        backend = _DensityBackend()
        with mock.patch(
            "model_protocol_lab.fixtures.audit_special_token_literals",
            return_value=(object(),),
        ):
            stage = experiments.density_stage(
                backend, FORMATS, boundary=(1,), max_context=8192, seed=0
            )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "density_screen_failed")

    def test_mirror_raises_on_unknown_arm_and_kind(self):
        """Test mirror raises on unknown arm and kind."""
        case = diffreports._density_generate(0)[0]
        with self.assertRaises(ValueError):
            diffreports._density_arm_block(case, "bogus")
        with self.assertRaises(ValueError):
            diffreports._density_satisfied({"kind": "bogus", "target": "x"}, [])

    def test_mixed_population_payload_validates(self):
        """Test mixed population payload validates."""
        # One payload exercising every record category at once
        # (perfect, value-only, zero-word, no-boundary,
        # detail-failed, case-screened), so the validator's
        # mirror aggregation runs its mixed branches -- the
        # fuzz-shaped population the pure-model payloads miss.
        seed = 5
        cases = density.generate_cases(seed)
        records = []
        for ci, case in enumerate(cases):
            for ai, arm in enumerate(density.ARM_KEYS):
                category = (ci + ai) % 7
                base = {
                    "candidate": "dn-a",
                    "case_index": ci,
                    "arm": arm,
                    "content_digest": density.digest_text(density.arm_content(case, arm)),
                    "input_tokens": 12,
                }
                if category == 0:
                    hashes = _perfect_hashes(case)
                    base.update(
                        detail="",
                        case_screened=False,
                        boundary_found=True,
                        answer_digest=density.digest_text("a"),
                        word_hashes=hashes,
                        word_list_digest=density.commitment_digest(hashes),
                        word_count=len(hashes),
                    )
                elif category == 1:
                    hashes = density.word_hashes(case["value"])
                    base.update(
                        detail="",
                        case_screened=False,
                        boundary_found=True,
                        answer_digest=density.digest_text("a"),
                        word_hashes=hashes,
                        word_list_digest=density.commitment_digest(hashes),
                        word_count=len(hashes),
                    )
                elif category == 2:
                    base.update(
                        detail="",
                        case_screened=False,
                        boundary_found=True,
                        answer_digest=density.digest_text("a"),
                        word_hashes=[],
                        word_list_digest=density.commitment_digest([]),
                        word_count=0,
                    )
                elif category == 3:
                    hashes = _perfect_hashes(case)
                    base.update(
                        detail="",
                        case_screened=False,
                        boundary_found=False,
                        answer_digest=density.digest_text("a"),
                        word_hashes=hashes,
                        word_list_digest=density.commitment_digest(hashes),
                        word_count=len(hashes),
                    )
                elif category == 4:
                    base.update(
                        detail="arm_evaluation_failed",
                        case_screened=False,
                        boundary_found=False,
                        answer_digest=None,
                        word_hashes=[],
                        word_list_digest=density.commitment_digest([]),
                        word_count=0,
                    )
                elif category == 5:
                    base.update(
                        detail="",
                        case_screened=True,
                        boundary_found=False,
                        answer_digest=None,
                        word_hashes=[],
                        word_list_digest=density.commitment_digest([]),
                        word_count=0,
                    )
                else:
                    hashes = _first_only_hashes(case)
                    base.update(
                        detail="",
                        case_screened=False,
                        boundary_found=True,
                        answer_digest=density.digest_text("a"),
                        word_hashes=hashes,
                        word_list_digest=density.commitment_digest(hashes),
                        word_count=len(hashes),
                    )
                records.append(base)
        protocol = dict(density.protocol_object(seed, "sha256:test-density"))
        protocol["fingerprint"] = density.protocol_fingerprint(seed, "sha256:test-density")
        aggregate = density.aggregate_candidate(records, cases)
        payload = {
            "available": True,
            "detail": "",
            "protocol": protocol,
            "candidates": {"dn-a": {**aggregate, "status": "available", "records": records}},
            "n_candidates": 1,
            "n_cases": 20,
        }
        self.assertTrue(_valid_density(payload))
        self.assertFalse(diff_density(payload, payload)["changed"])
        self.assertEqual(
            aggregate["n_generated"],
            aggregate["n_boundary_screened"]
            + aggregate["n_detail_screened"]
            + aggregate["n_no_boundary"]
            + aggregate["n_live"],
        )

    def test_diff_status_and_membership_drift(self):
        """Test diff status and membership drift."""
        two = _payload()
        one = _payload(formats=FORMATS[:1])
        result = diff_density(two, one)
        self.assertEqual(result["candidates_removed"], ["dn-b"])
        self.assertTrue(result["changed"])
        backend = _DensityBackend()
        stage = experiments.density_stage(
            backend,
            FORMATS,
            boundary=tuple(backend.encode("endmark")),
            max_context=8192,
            seed=0,
        )
        stage["candidates"]["dn-b"] = {
            "status": "unavailable",
            "detail": "candidate_evaluation_failed",
            "records": [],
        }
        mixed = experiments.density_analysis(stage, tuple(f.name for f in FORMATS))
        result = diff_density(two, mixed)
        status_drift = [
            entry
            for entry in result["candidate_drift"]
            if entry["metric"] == "status" and entry["candidate"] == "dn-b"
        ]
        self.assertEqual(len(status_drift), 1)
        self.assertTrue(result["changed"])

    def test_diff_zero_word_drift(self):
        """Test diff zero word drift."""

        class _EmptyBackend(_DensityBackend):
            """Empty backend."""

            def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
                """Generate."""
                self.calls += 1
                boundary = self.encode("endmark")[0]
                return Generation(token_ids=(boundary,) * max_new_tokens, reason="limit")

        perfect = _payload()
        empty = _payload(_EmptyBackend())
        for entry in empty["candidates"].values():
            self.assertEqual(entry["n_zero_word"], 420)
        result = diff_density(perfect, empty)
        metrics = {entry["metric"] for entry in result["candidate_drift"]}
        self.assertIn("n_zero_word", metrics)
        self.assertTrue(result["changed"])

    def test_validator_structural_rejections(self):
        """Test validator structural rejections."""
        payload = _payload()
        self.assertFalse(_valid_density(None))
        self.assertFalse(_valid_density({}))
        self.assertFalse(_valid_density({"available": True}))
        bad_detail = copy.deepcopy(payload)
        bad_detail["detail"] = "density_unavailable"
        self.assertFalse(_valid_density(bad_detail))
        missing_key = copy.deepcopy(payload)
        missing_key.pop("n_cases")
        self.assertFalse(_valid_density(missing_key))
        all_unavailable = copy.deepcopy(payload)
        for name in list(all_unavailable["candidates"]):
            all_unavailable["candidates"][name] = {
                "status": "unavailable",
                "detail": "candidate_evaluation_failed",
                "records": [],
            }
        self.assertFalse(_valid_density(all_unavailable))
        no_answer = copy.deepcopy(payload)
        no_answer["candidates"]["dn-a"]["records"][0]["answer_digest"] = None
        self.assertFalse(_valid_density(no_answer))

    def test_validator_record_and_byte_caps(self):
        """Test validator record and byte caps."""
        one = _payload(formats=FORMATS[:1])
        entry = one["candidates"]["dn-a"]
        clones: dict = {}
        for i in range(25):
            name = f"cand-{i:02d}"
            cloned = copy.deepcopy(entry)
            for record in cloned["records"]:
                record["candidate"] = name
            clones[name] = cloned
        over_total = {
            "available": True,
            "detail": "",
            "protocol": one["protocol"],
            "candidates": clones,
            "n_candidates": 25,
            "n_cases": 20,
        }
        self.assertFalse(_valid_density(over_total))
        fat = copy.deepcopy(one)
        fat["candidates"]["dn-a"]["records"][0]["word_hashes"] = ["a" * 64] * 900_000
        fat["candidates"]["dn-a"]["records"][0]["word_count"] = 900_000
        self.assertFalse(_valid_density(fat))
