"""Cycle 191: needle-depth x template (wave-8 rank 3).

Pins the frozen contract in docs/NEEDLE_DESIGN.md (design v2,
after red-team A's DO-NOT-BUILD verdict, red-team B's
cross-check that upheld the verdict, downgraded A's
impossibility claim, and refuted the coordinator's first
repair sketch): the seed-derived case generator, the
delta-based dose and the prefix-anchored offset measurement
``D``, the phased sizing search (including a merging digraph
tokenizer, the class that killed the marker anchor), the
stage's per-cell degradation vocabulary and context
pre-check, the shared headline population, the duplicate
independence baseline, the paired-restricted flip baseline,
the full-recompute diff validator (one break at a time), the
diff, the reporting section, and the CLI helper.
"""

from __future__ import annotations

import copy
import hashlib
import re
import threading
import unittest

from model_protocol_lab import cli, experiments, needle
from model_protocol_lab.core import Format, Generation
from model_protocol_lab.diffreports import _valid_needle, diff_needle
from model_protocol_lab.reporting import render_html

FORMATS = (
    Format(name="nd-a", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>"),
    Format(name="nd-b", bos="", user_prefix="U>", assistant_prefix="A>", turn_end="<e>"),
)

_NEEDLE_RE = re.compile(r"code is ([a-z]+-\d+)\.")


class _WordBackend:
    """Word-token scripted backend with switchable behaviours.

    Modes: ``perfect`` (answer the planted code), ``midfail``
    (answer wrongly when the needle sits in the middle stretch
    of the prompt -- the depth-sensitive mode: a
    treatment-blind fake would make the stage tests vacuous),
    ``wrong`` (every answer wrong), ``noboundary`` (no boundary
    token emitted), ``explode`` (generate raises).
    """

    fingerprint = "sha256:test-needle"

    def __init__(self, mode: str = "perfect", cancel_after: int | None = None):
        self.mode = mode
        self.cancel_after = cancel_after
        self.cancel_event: threading.Event | None = None
        self.calls = 0
        self.vocab: dict[str, int] = {}
        self.rev: dict[int, str] = {}

    def encode(self, text):
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
        return " ".join(self.rev.get(int(i), "?") for i in ids)

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
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
        found = _NEEDLE_RE.findall(text)
        answer = found[-1] if found else "unknown"
        if self.mode == "wrong":
            answer = "wrong-0000"
        elif self.mode == "midfail" and found and len(text) > 1000:
            # The length guard keeps the short dose-0 control
            # (where the needle is unavoidably at a mid
            # fraction of a tiny prompt) out of the failure
            # mode: midfail is a statement about bulk depth.
            at = text.find("code is")
            fraction = at / max(1, len(text))
            if 0.2 < fraction < 0.8:
                answer = "wrong-0000"
        boundary = self.encode("endmark")[0]
        if self.mode == "noboundary":
            filler = self.encode("zz")[0]
            out = list(self.encode(answer))
            return Generation(
                token_ids=tuple((out + [filler] * max_new_tokens)[:max_new_tokens]),
                reason="limit",
            )
        out = list(self.encode(answer)) + [boundary]
        out = (out + [boundary] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


class _ByteBackend:
    """Byte-level scripted backend: fast over the full grid."""

    fingerprint = "sha256:test-needle-byte"

    def encode(self, text):
        return tuple(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(int(i) % 256 for i in ids).decode("utf-8", "replace")

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        text = self.decode(list(input_ids))
        found = _NEEDLE_RE.findall(text)
        answer = found[-1] if found else "unknown"
        boundary = list(self.encode("endmark"))
        out = list(self.encode(answer)) + boundary
        return Generation(
            token_ids=tuple((out + boundary * max_new_tokens)[:max_new_tokens]),
            reason="limit",
        )


_MERGE_DIGRAPHS = ("Th", "he", "e ", ". ", " T", " W", "s ", "e.")


class _MergeBackend:
    """Greedy longest-match tokenizer with seam-crossing
    digraphs (``" T"`` spans a filler/needle seam, ``" W"`` the
    filler/question seam). This is the tokenizer class under
    which the marker-derived head anchor returned None or a
    biased offset; the stage's ``D`` measurement must still
    place needles exactly, or degrade cells to unmatchable --
    never accept a wrong offset."""

    fingerprint = "sha256:test-needle-merge"

    def __init__(self):
        vocab = {chr(c) for c in range(32, 127)} | set(_MERGE_DIGRAPHS) | {"§"}
        self.tokens = sorted(vocab, key=len, reverse=True)
        self.ids = {token: i + 1 for i, token in enumerate(self.tokens)}
        self.rev = {i: token for token, i in self.ids.items()}
        # First-char index so encode stays linear: at each
        # position only the tokens starting there are tried,
        # longest first (the same greedy match, faster).
        self.by_first: dict[str, list[str]] = {}
        for token in self.tokens:
            self.by_first.setdefault(token[0], []).append(token)

    def encode(self, text):
        ids = []
        at = 0
        while at < len(text):
            for token in self.by_first[text[at]]:
                if text.startswith(token, at):
                    ids.append(self.ids[token])
                    at += len(token)
                    break
            else:  # pragma: no cover -- vocab covers all printable ASCII
                raise ValueError("unencodable char")
        return tuple(ids)

    def decode(self, ids):
        return "".join(self.rev.get(int(i), "?") for i in ids)

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        text = self.decode(list(input_ids))
        found = _NEEDLE_RE.findall(text)
        answer = found[-1] if found else "unknown"
        boundary = self.encode("§")[0]
        out = list(self.encode(answer)) + [boundary]
        out = (out + [boundary] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


def _payload(backend=None, formats=FORMATS, seed=0, **stage_kwargs):
    backend = backend if backend is not None else _WordBackend()
    stage_kwargs.setdefault("max_context", 2048)
    stage = experiments.needle_stage(
        backend,
        formats,
        boundary=tuple(backend.encode("endmark")),
        seed=seed,
        **stage_kwargs,
    )
    return experiments.needle_analysis(stage, tuple(f.name for f in formats))


def _full_payload(seed=0):
    backend = _ByteBackend()
    stage = experiments.needle_stage(
        backend,
        FORMATS,
        boundary=tuple(backend.encode("endmark")),
        max_context=65536,
        seed=seed,
    )
    return experiments.needle_analysis(stage, tuple(f.name for f in FORMATS))


class TestProtocol(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(needle.DOSES, (1024, 4096))
        self.assertEqual(needle.DEPTHS, (0, 25, 50, 75, 100))
        self.assertEqual(needle.CASE_COUNT, 20)
        self.assertEqual(len(needle.CELL_KEYS), 15)
        self.assertEqual(needle.CELL_KEYS[0], "dose_0")
        self.assertEqual(needle.MAX_GROUND_RECORDS, 480)
        self.assertEqual(needle.GENERATION_BUDGET, 7200)
        self.assertEqual(needle.SIZING_ENCODE_BUDGET, 512)
        self.assertEqual(needle.MIN_HEADLINE_POPULATION, 6)

    def test_generation_deterministic_and_distinct(self):
        first = needle.generate_cases(7)
        self.assertEqual(first, needle.generate_cases(7))
        self.assertNotEqual(first, needle.generate_cases(8))
        values = [case["value"] for case in first]
        self.assertEqual(len(set(values)), 20)
        self.assertEqual(
            [case["fixture_id"] for case in first],
            [f"needle-{i:02d}" for i in range(20)],
        )
        for case in first:
            self.assertEqual(case["fact"], f"The {case['topic']} code is {case['value']}. ")
            self.assertEqual(case["question"], f"What is the {case['topic']} code?")
            self.assertEqual(case["expected"], case["value"])

    def test_generation_seed_must_be_int(self):
        with self.assertRaises(ValueError):
            needle.generate_cases("0")  # type: ignore[arg-type]

    def test_topics_and_words_absent_from_corpus(self):
        from model_protocol_lab.distraction import NEUTRAL_CORPUS

        corpus = "".join(NEUTRAL_CORPUS).casefold()
        for topic in needle.TOPICS:
            self.assertNotIn(topic.casefold(), corpus)
        for word in needle.WORDS:
            self.assertNotIn(word.casefold(), corpus)
        self.assertFalse(any(ch.isdigit() for ch in corpus))

    def test_protocol_binding(self):
        self.assertEqual(
            needle.protocol_fingerprint(3, "sha256:x"),
            needle.protocol_fingerprint(3, "sha256:x"),
        )
        self.assertNotEqual(
            needle.protocol_fingerprint(3, "sha256:x"),
            needle.protocol_fingerprint(3, "sha256:y"),
        )
        self.assertNotEqual(
            needle.protocol_fingerprint(3, "sha256:x"),
            needle.protocol_fingerprint(4, "sha256:x"),
        )
        protocol = needle.protocol_object(3, "sha256:x")
        self.assertEqual(protocol["ground_sha256"], needle.ground_sha256(3))
        self.assertEqual(protocol["sizing_rule_sha256"], needle.sizing_rule_sha256())
        self.assertEqual(protocol["screening_rule_sha256"], needle.screening_rule_sha256())
        self.assertEqual(protocol["case_material_sha256"], needle.case_material_sha256())

    def test_target_offsets(self):
        self.assertEqual(
            [needle.target_offset(1024, d) for d in needle.DEPTHS],
            [0, 256, 512, 768, 1024],
        )
        self.assertEqual(
            [needle.target_offset(4096, d) for d in needle.DEPTHS],
            [0, 1024, 2048, 3072, 4096],
        )

    def test_offset_measurement_head_merge_cancels(self):
        # No merges: head 9, needle starts 70, question starts
        # 50, prefix is (30, 1, 2) -- D reads the prefix length.
        q = (9, 50, 5, 6)
        z = (9, 70, 7, 8, 50, 5, 6)
        arm = (9, 30, 1, 2, 70, 7, 8, 50, 5, 6)
        comparator = (9, 30, 1, 2, 50, 5, 6)
        self.assertEqual(needle.offset_measurement(arm, comparator, z, q), 3)
        # Head merges with whatever follows it: with the
        # needle in Z (9+70 -> 90), with the question in Q
        # (9+50 -> 91), with the prefix in the arm and its
        # comparator (9+30 -> 92). LCP(Z, Q) collapses to 0
        # and LCP(arm, comparator) counts the merged head
        # token once -- D still reads the 3-token prefix.
        q = (91, 5, 6)
        z = (90, 7, 8, 5, 6)
        arm = (92, 1, 2, 70, 7, 8, 50, 5, 6)
        comparator = (92, 1, 2, 50, 5, 6)
        self.assertEqual(needle.offset_measurement(arm, comparator, z, q), 3)
        self.assertEqual(needle.lcp_len((1, 2, 3), (1, 2, 4)), 2)
        self.assertEqual(needle.lcp_len((), (1,)), 0)

    def test_filler_screened(self):
        self.assertTrue(needle.filler_screened("... Velvet-1234 ...", "velvet-1234", ""))
        self.assertTrue(needle.filler_screened("has END inside", "velvet-1234", "END"))
        self.assertFalse(needle.filler_screened("plain prose", "velvet-1234", "END"))
        self.assertFalse(needle.filler_screened("plain prose", "velvet-1234", ""))


class TestAggregatesPure(unittest.TestCase):
    """Hand-built records against the pure readouts."""

    def _record(self, fixture, verdicts):
        expected = f"{fixture}-value"
        digest = hashlib.sha256(expected.encode("utf-8")).hexdigest()
        wrong = hashlib.sha256(b"wrong").hexdigest()
        cells = {}
        details = {}
        for key in needle.CELL_KEYS:
            verdict = verdicts.get(key)
            if verdict is None:
                cells[key] = None
                details[key] = "arm_unmatchable"
                continue
            details[key] = ""
            if key == "dose_0":
                tokens, offsets = 20, [0]
            elif key.startswith("absent_"):
                tokens, offsets = 10 + int(key.split("_")[1]), []
            elif key.startswith("duplicate_"):
                dose = int(key.split("_")[1])
                tokens, offsets = 20 + dose + 10, [dose // 4, 3 * dose // 4]
            else:
                _, dose_s, _, depth_s = key.split("_")
                tokens = 20 + int(dose_s)
                offsets = [needle.target_offset(int(dose_s), int(depth_s))]
            cells[key] = {
                "verdict": verdict,
                "boundary_found": True,
                "boundary_text_match": True,
                "input_tokens": tokens,
                "answer_digest": digest if verdict else wrong,
                "needle_offsets": offsets,
            }
        return {
            "fixture": fixture,
            "topic": "chess club",
            "expected_digest": digest,
            "question_tokens": 10,
            "cells": cells,
            "cell_details": details,
        }

    def _all(self, verdict=True):
        return dict.fromkeys(needle.CELL_KEYS, verdict)

    def test_generation_collision_redraws_both_slots(self):
        # Seed 584's first draws (counter 0) collide: only 19
        # distinct values for 20 cases. The frozen redraw rule
        # must still deliver 20 pairwise-distinct values.
        raw = []
        for index in range(needle.CASE_COUNT):
            word = needle.WORDS[needle._derive(584, index, 0, 0) % len(needle.WORDS)]
            number = 1000 + needle._derive(584, index, 1, 0) % 9000
            raw.append(f"{word}-{number}")
        self.assertEqual(len(set(raw)), 19)
        cases = needle.generate_cases(584)
        self.assertEqual(len({case["value"] for case in cases}), needle.CASE_COUNT)

    def test_aggregates_tolerate_non_dict_cell(self):
        # A cell slot holding a non-dict is treated as missing
        # by every readout (the _cell guard), never crashed on;
        # the same holds when the cells mapping itself is junk.
        record = self._record("needle-00", {})
        record["cells"]["dose_0"] = "junk"
        control = needle.dose0_control([record])
        self.assertEqual(control["n_recorded"], 0)
        self.assertIsNone(control["rate"])
        record["cells"] = "junk"
        control = needle.dose0_control([record])
        self.assertEqual(control["n_recorded"], 0)

    def test_headline_population_empty_records(self):
        self.assertEqual(needle.headline_population({}, 1024), [])
        readouts = needle.headline_readouts({})
        self.assertEqual(readouts["1024"]["n_population"], 0)
        self.assertEqual(readouts["1024"]["curves"], {})

    def test_headline_population_requires_every_candidate(self):
        good = [self._record(f"needle-{i:02d}", self._all()) for i in range(8)]
        bad = [self._record(f"needle-{i:02d}", self._all()) for i in range(8)]
        bad[3]["cells"]["dose_0"]["verdict"] = False
        population = needle.headline_population({"a": good, "b": bad}, 1024)
        self.assertEqual(len(population), 7)
        self.assertNotIn("needle-03", population)

    def test_deficit_withheld_below_floor(self):
        records = [self._record(f"needle-{i:02d}", self._all()) for i in range(5)]
        readouts = needle.headline_readouts({"a": records})
        self.assertEqual(readouts["1024"]["n_population"], 5)
        self.assertIsNone(readouts["1024"]["deficit"]["a"])
        self.assertIsNone(readouts["1024"]["first_failure"]["a"])
        self.assertEqual(readouts["1024"]["curves"]["a"]["50"]["rate"], 1.0)

    def test_deficit_is_edge_mean_minus_middle(self):
        records = []
        for i in range(8):
            verdicts = self._all()
            if i < 4:
                verdicts["dose_1024_depth_50"] = False
            records.append(self._record(f"needle-{i:02d}", verdicts))
        readouts = needle.headline_readouts({"a": records})
        # edges 1.0, middle 0.5 -> deficit 0.5; flat elsewhere 0.
        self.assertEqual(readouts["1024"]["deficit"]["a"], 0.5)
        first = readouts["1024"]["first_failure"]["a"]
        self.assertEqual(first["counts"]["50"], 4)
        self.assertEqual(first["n_no_failure"], 4)

    def test_duplicate_independence_baseline(self):
        records = []
        for i in range(10):
            verdicts = self._all()
            verdicts["dose_1024_depth_25"] = i < 5  # r25 = 0.5
            verdicts["dose_1024_depth_75"] = i < 5  # r75 = 0.5
            verdicts["duplicate_1024"] = i < 6  # pass 0.6
            verdicts["dose_1024_depth_50"] = i >= 5  # mid fails for i<5
            records.append(self._record(f"needle-{i:02d}", verdicts))
        dup = needle.aggregate_candidate(records)["duplicate"]["1024"]
        self.assertEqual(dup["n_population"], 10)
        self.assertEqual(dup["independence_expected"], 0.75)
        self.assertAlmostEqual(dup["excess_over_independence"], -0.15)
        self.assertEqual(dup["n_mid_fail"], 5)
        self.assertEqual(dup["n_rescued"], 5)  # i<5: duplicate passes for i<6
        self.assertEqual(dup["rescue_rate"], 1.0)

    def test_cell_selection_and_flip(self):
        # Candidate a passes everything; b fails one dose-0 case
        # and one treated case on a different fixture, so the
        # treated selection over its completed set flips against
        # the restricted dose-0 baseline.
        a = [self._record(f"needle-{i:02d}", self._all()) for i in range(4)]
        b = [self._record(f"needle-{i:02d}", self._all()) for i in range(4)]
        b[0]["cells"]["dose_0"]["verdict"] = False
        b[1]["cells"]["dose_1024_depth_50"]["verdict"] = False
        by = {"a": a, "b": b}
        dose0 = needle.cell_selection(by, "dose_0")
        self.assertEqual(dose0["decision"], "selected")
        self.assertEqual(dose0["winner"], "a")
        self.assertFalse(dose0["flip"])
        treated = needle.cell_selection(by, "dose_1024_depth_50")
        self.assertEqual(treated["decision"], "selected")
        self.assertEqual(treated["winner"], "a")
        # Restricted baseline on the treated completed set (all
        # four fixtures): b fails dose-0 on needle-00, which is
        # in the set, so the baseline also selects a -> no flip.
        self.assertFalse(treated["flip"])

    def test_cell_selection_withheld_on_mismatched_sets(self):
        a = [self._record(f"needle-{i:02d}", self._all()) for i in range(3)]
        b = [self._record(f"needle-{i:02d}", self._all()) for i in range(3)]
        b[1]["cells"]["dose_1024_depth_50"] = None
        b[1]["cell_details"]["dose_1024_depth_50"] = "arm_screened"
        selection = needle.cell_selection({"a": a, "b": b}, "dose_1024_depth_50")
        self.assertTrue(selection["withheld"])
        self.assertEqual(selection["decision"], "unavailable")
        self.assertIsNone(selection["flip"])


class TestStageWordBackend(unittest.TestCase):
    def test_perfect_backend_default_context(self):
        payload = _payload()
        self.assertTrue(payload["available"])
        self.assertTrue(_valid_needle(payload))
        for name in ("nd-a", "nd-b"):
            entry = payload["candidates"][name]
            self.assertEqual(entry["dose0_control"]["rate"], 1.0)
            self.assertEqual(entry["false_recall"]["1024"]["rate"], 0.0)
            for record in entry["records"]:
                for dose in needle.DOSES:
                    for depth in needle.DEPTHS:
                        key = needle.cell_key_treated(dose, depth)
                        if dose == 1024:
                            self.assertIsNotNone(record["cells"][key])
                        else:
                            self.assertIsNone(record["cells"][key])
                            self.assertEqual(record["cell_details"][key], "arm_context_exceeded")
        self.assertEqual(payload["readouts"]["1024"]["n_population"], 20)
        self.assertEqual(payload["readouts"]["4096"]["n_population"], 0)
        self.assertEqual(payload["readouts"]["1024"]["deficit"]["nd-a"], 0.0)
        self.assertIsNone(payload["readouts"]["4096"]["deficit"]["nd-a"])
        self.assertEqual(payload["conditions"]["dose_0"]["decision"], "ambiguous")
        self.assertFalse(payload["conditions"]["dose_0"]["flip"])

    def test_midfail_backend_shows_deficit(self):
        payload = _payload(_WordBackend(mode="midfail"))
        self.assertTrue(_valid_needle(payload))
        deficit = payload["readouts"]["1024"]["deficit"]["nd-a"]
        self.assertIsNotNone(deficit)
        self.assertGreater(deficit, 0.0)
        curves = payload["readouts"]["1024"]["curves"]["nd-a"]
        self.assertGreater(curves["0"]["rate"], curves["50"]["rate"])
        first = payload["readouts"]["1024"]["first_failure"]["nd-a"]
        self.assertGreater(first["counts"]["25"] + first["counts"]["50"], 0)

    def test_wrong_backend_empties_headline(self):
        payload = _payload(_WordBackend(mode="wrong"))
        self.assertTrue(_valid_needle(payload))
        entry = payload["candidates"]["nd-a"]
        self.assertEqual(entry["dose0_control"]["rate"], 0.0)
        self.assertEqual(payload["readouts"]["1024"]["n_population"], 0)
        self.assertEqual(payload["conditions"]["dose_0"]["decision"], "no_candidate")

    def test_stage_rejects_bad_formats(self):
        backend = _WordBackend()
        with self.assertRaises(ValueError):
            experiments.needle_stage(backend, [], boundary=(1,), max_context=100)
        with self.assertRaises(ValueError):
            experiments.needle_stage(backend, (), boundary=(1,), max_context=100)

    def test_record_budget(self):
        backend = _WordBackend()
        formats = tuple(
            Format(name=f"f{i}", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>")
            for i in range(25)
        )
        stage = experiments.needle_stage(backend, formats, boundary=(1,), max_context=100, seed=0)
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "needle_record_budget_exceeded")

    def test_screen_failed_stage(self):
        # A boundary whose decoded text appears inside the
        # frozen generated text fails the whole stage closed
        # before any generation.
        backend = _WordBackend()
        stage = experiments.needle_stage(
            backend,
            FORMATS[:1],
            boundary=tuple(backend.encode("code")),
            max_context=8192,
            seed=0,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "needle_screen_failed")

    def test_degenerate_tokenizer_all_unmatchable(self):
        class _OneTokenBackend(_WordBackend):
            def encode(self, text):
                return (1,)

        backend = _OneTokenBackend()
        stage = experiments.needle_stage(
            backend, FORMATS[:1], boundary=(1,), max_context=8192, seed=0
        )
        self.assertTrue(stage["available"])
        for record in stage["candidates"]["nd-a"]["records"]:
            self.assertTrue(all(d == "arm_unmatchable" for d in record["cell_details"].values()))

    def test_cancel_unwinds_stage(self):
        backend = _WordBackend(cancel_after=1)
        event = threading.Event()
        backend.cancel_event = event
        stage = experiments.needle_stage(
            backend,
            FORMATS[:1],
            boundary=tuple(backend.encode("endmark")),
            max_context=2048,
            seed=0,
            cancel=event,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")

    def test_exploding_backend_marks_cells_failed(self):
        payload = _payload(_WordBackend(mode="explode"))
        # evaluate() surfaces the failure as an ineligible
        # report, so cells degrade to arm_evaluation_failed and
        # the payload stays valid and available.
        self.assertTrue(payload["available"])
        self.assertTrue(_valid_needle(payload))
        entry = payload["candidates"]["nd-a"]
        for record in entry["records"]:
            self.assertEqual(record["cell_details"]["dose_0"], "arm_evaluation_failed")


class TestStageFullGrid(unittest.TestCase):
    def test_byte_backend_full_grid(self):
        payload = _full_payload()
        self.assertTrue(payload["available"])
        self.assertTrue(_valid_needle(payload))
        for name in ("nd-a", "nd-b"):
            for record in payload["candidates"][name]["records"]:
                for key in needle.CELL_KEYS:
                    self.assertIsNotNone(record["cells"][key], key)
        self.assertEqual(payload["readouts"]["4096"]["n_population"], 20)
        self.assertEqual(payload["n_cases"], 20)
        self.assertEqual(payload["n_candidates"], 2)

    def test_merge_backend_places_needles(self):
        backend = _MergeBackend()
        stage = experiments.needle_stage(
            backend,
            FORMATS,
            boundary=tuple(backend.encode("§")),
            max_context=2600,
            seed=0,
        )
        self.assertTrue(stage["available"])
        payload = experiments.needle_analysis(stage, tuple(f.name for f in FORMATS))
        self.assertTrue(_valid_needle(payload))
        present = 0
        total = 0
        for entry in payload["candidates"].values():
            for record in entry["records"]:
                self.assertIsNotNone(record["cells"]["dose_0"])
                for key in needle.CELL_KEYS:
                    if key != "dose_0" and "1024" not in key:
                        continue  # 4096 cells exceed this run's context cap
                    total += 1
                    if record["cells"][key] is not None:
                        present += 1
        # Soundness is the guarantee (the payload above is
        # valid, so every accepted cell is exact); liveness
        # under this merge-dense tokenizer is bounded by the
        # step-skip mechanism documented in the design: the
        # delta walk steps +2 over some targets at seam
        # toggles, so a share of cells is honestly
        # unmatchable. A regression to the marker anchor's
        # failure mode would show up as invalid payloads or
        # near-zero placement, not this.
        self.assertGreater(present / total, 0.35)


class TestValidatorBattery(unittest.TestCase):
    """One break at a time against a coherent payload."""

    @classmethod
    def setUpClass(cls):
        cls.payload = _full_payload()

    def _broken(self, mutate):
        payload = copy.deepcopy(self.payload)
        mutate(payload)
        return payload

    def test_coherent_payload_valid(self):
        self.assertTrue(_valid_needle(self.payload))

    def test_breaks_rejected(self):
        def rec(p, name="nd-a", i=0):
            return p["candidates"][name]["records"][i]

        breaks = {
            "verdict_flip": lambda p: rec(p)["cells"]["dose_1024_depth_50"].update(verdict=False),
            "answer_digest_swap": lambda p: rec(p)["cells"]["dose_1024_depth_50"].update(
                answer_digest=rec(p, i=1)["expected_digest"]
            ),
            "treated_input_off_by_one": lambda p: rec(p)["cells"]["dose_1024_depth_50"].update(
                input_tokens=rec(p)["cells"]["dose_1024_depth_50"]["input_tokens"] + 1
            ),
            "dose0_input_off_by_one": lambda p: rec(p)["cells"]["dose_0"].update(
                input_tokens=rec(p)["cells"]["dose_0"]["input_tokens"] + 1
            ),
            "absent_input_off_by_one": lambda p: rec(p)["cells"]["absent_1024"].update(
                input_tokens=rec(p)["cells"]["absent_1024"]["input_tokens"] + 1
            ),
            "duplicate_input_off_by_one": lambda p: rec(p)["cells"]["duplicate_4096"].update(
                input_tokens=rec(p)["cells"]["duplicate_4096"]["input_tokens"] + 1
            ),
            "question_tokens_off_by_one": lambda p: rec(p).update(
                question_tokens=rec(p)["question_tokens"] + 1
            ),
            "offsets_wrong": lambda p: rec(p)["cells"]["dose_1024_depth_50"].update(
                needle_offsets=[511]
            ),
            "offsets_missing": lambda p: rec(p)["cells"]["dose_1024_depth_50"].update(
                needle_offsets=[]
            ),
            "topic_changed": lambda p: rec(p).update(topic="rowing team"),
            "expected_digest_changed": lambda p: rec(p).update(expected_digest="0" * 64),
            "record_order_swapped": lambda p: p["candidates"]["nd-a"]["records"].__setitem__(
                slice(0, 2),
                [rec(p, i=1), rec(p, i=0)],
            ),
            "record_dropped": lambda p: p["candidates"]["nd-a"]["records"].pop(),
            "record_extra_key": lambda p: rec(p).update(extra=1),
            "detail_on_present_cell": lambda p: rec(p)["cell_details"].update(
                dose_0="arm_screened"
            ),
            "missing_cell_blank_detail": lambda p: (
                rec(p)["cells"].update(dose_0=None),
                rec(p)["cell_details"].update(dose_0=""),
            ),
            "missing_cell_bogus_detail": lambda p: (
                rec(p)["cells"].update(dose_0=None),
                rec(p)["cell_details"].update(dose_0="bogus"),
            ),
            "deficit_tampered": lambda p: p["readouts"]["1024"]["deficit"].update({"nd-a": 0.5}),
            "curve_tampered": lambda p: p["readouts"]["1024"]["curves"]["nd-a"]["50"].update(
                n_pass=3
            ),
            "first_failure_tampered": lambda p: p["readouts"]["1024"]["first_failure"]["nd-a"][
                "counts"
            ].update({"50": 2}),
            "dose0_rate_tampered": lambda p: p["candidates"]["nd-a"]["dose0_control"].update(
                rate=0.5
            ),
            "false_recall_tampered": lambda p: p["candidates"]["nd-a"]["false_recall"][
                "1024"
            ].update(n_false_pass=1),
            "duplicate_excess_tampered": lambda p: p["candidates"]["nd-a"]["duplicate"][
                "1024"
            ].update(excess_over_independence=0.25),
            "condition_winner_tampered": lambda p: p["conditions"]["dose_0"].update(winner="nd-a"),
            "condition_flip_tampered": lambda p: p["conditions"]["dose_1024_depth_50"].update(
                flip=True
            ),
            "fingerprint_zeroed": lambda p: p["protocol"].update(fingerprint="0" * 64),
            "sizing_rule_tampered": lambda p: p["protocol"].update(sizing_rule_sha256="0" * 64),
            "n_cases_tampered": lambda p: p.update(n_cases=19),
            "entry_extra_key": lambda p: p["candidates"]["nd-a"].update(extra=1),
            "payload_extra_key": lambda p: p.update(extra=1),
        }
        for name, mutate in breaks.items():
            with self.subTest(case=name):
                self.assertFalse(_valid_needle(self._broken(mutate)), name)

    def test_unavailable_payload_invalid(self):
        self.assertFalse(_valid_needle({"available": False, "detail": "needle_unavailable"}))
        self.assertFalse(_valid_needle(None))
        self.assertFalse(_valid_needle({}))


class TestDiffNeedle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = _full_payload()

    def test_self_diff_unchanged(self):
        diff = diff_needle(self.payload, copy.deepcopy(self.payload))
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])
        self.assertTrue(diff["comparable"])

    def test_seed_change_is_generator_drift(self):
        other = _full_payload(seed=1)
        diff = diff_needle(self.payload, other)
        self.assertTrue(diff["generator_drift"])
        self.assertFalse(diff["comparable"])
        self.assertTrue(diff["changed"])

    def test_one_sided_availability(self):
        diff = diff_needle(self.payload, {"available": False})
        self.assertEqual(diff["available"], "old")
        self.assertFalse(diff["changed"])
        diff = diff_needle(None, None)
        self.assertEqual(diff["available"], "neither")

    def test_deficit_drift_reported(self):
        backend = _WordBackend(mode="midfail")
        stage = experiments.needle_stage(
            backend,
            FORMATS,
            boundary=tuple(backend.encode("endmark")),
            max_context=2048,
            seed=0,
        )
        other = experiments.needle_analysis(stage, tuple(f.name for f in FORMATS))
        base = _payload()
        diff = diff_needle(base, other)
        self.assertTrue(diff["changed"])
        metrics = {entry["metric"] for entry in diff["candidate_drift"]}
        self.assertIn("dose_1024_deficit", metrics)


class TestReportingAndCli(unittest.TestCase):
    def test_section_renders(self):
        payload = _full_payload()
        html = render_html({"needle": payload})
        self.assertIn('<section class="card" id="needle">', html)
        self.assertIn("Needle depth", html)
        self.assertIn("mid-depth deficit", html)
        self.assertNotIn("obelisk", html)  # no needle values rendered

    def test_section_absent_without_payload(self):
        html = render_html({})
        self.assertNotIn('<section class="card" id="needle">', html)

    def test_cli_helper_attaches_payload(self):
        backend = _ByteBackend()
        result: dict = {}
        cli._run_needle_probe(
            backend,
            FORMATS,
            result,
            boundary=tuple(backend.encode("endmark")),
            boundary_map=None,
            max_new_tokens=64,
            max_context=65536,
            seed=0,
            cancel=None,
        )
        self.assertTrue(result["needle"]["available"])
        self.assertTrue(_valid_needle(result["needle"]))

    def test_cli_helper_never_raises(self):
        class _Boom:
            fingerprint = "sha256:boom"

            def encode(self, text):
                raise RuntimeError("boom")

        result: dict = {}
        cli._run_needle_probe(
            _Boom(),
            FORMATS,
            result,
            boundary=(1,),
            boundary_map=None,
            max_new_tokens=64,
            max_context=2048,
            seed=0,
            cancel=None,
        )
        self.assertEqual(result["needle"], {"available": False, "detail": "needle_unavailable"})


if __name__ == "__main__":
    unittest.main()
