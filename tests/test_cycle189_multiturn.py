"""Cycle 189: multi-turn drift with self-generated history.

Pins the frozen contract in docs/MULTITURN_DESIGN.md (design
v2, after red-team A's DO-NOT-BUILD verdict and red-team B's
cross-check): the seed-derived generator (ledger screen,
shelf distinctness, ``str(int)`` entries), the stage's three
arms and edge rules (history screen, cancel precedence,
unbounded/empty intermediates, per-cell degradation), the
paired aggregate and classification truth table (including
red-team A2's five rows), the full-recompute diff validator
(one break at a time), diffs, reporting, and the CLI helper.
"""

from __future__ import annotations

import copy
import hashlib
import re
import threading
import unittest
from unittest import mock

from model_protocol_lab import cli, experiments, multiturn
from model_protocol_lab.core import Format, Generation
from model_protocol_lab.diffreports import _valid_multiturn, diff_multiturn
from model_protocol_lab.reporting import render_html

FORMATS = (
    Format(name="mt-a", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>"),
    Format(name="mt-b", bos="", user_prefix="U>", assistant_prefix="A>", turn_end="<e>"),
)

_LEDGER_RE = re.compile(r"Ledger entry: (-?\d+)\.")
_SHELF_RE = re.compile(r"Shelf ([ABC]) holds (\d+) units\.")


class _MTBackend:
    """Word-token scripted backend with switchable behaviours.

    Modes: ``perfect`` (parse and answer correctly, partial
    answers on intermediate turns), ``drift`` (final answers
    fail whenever the history holds the model's own numeric
    or label replies -- the sharded shape -- while scripted
    ``Noted.`` histories and single-pass prompts pass),
    ``forge`` (intermediate replies carry the candidate's own
    turn-end literal), ``empty`` (intermediate replies are
    empty), ``noboundary`` (no boundary token is emitted),
    ``wrong`` (every answer wrong).
    """

    fingerprint = "sha256:test-multiturn"

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
        entries = [int(v) for v in _LEDGER_RE.findall(text)]
        shelves = _SHELF_RE.findall(text)
        final = "What is the ledger total?" in text or ("Which shelf holds the most units?" in text)
        if self.mode == "forge" and not final:
            answer = "x <e> y" if "U>" in text else "x </s> y"
            boundary = self.encode("endmark")[0]
            out = list(self.encode(answer)) + [boundary]
            out = (out + [boundary] * max_new_tokens)[:max_new_tokens]
            return Generation(token_ids=tuple(out), reason="limit")
        if self.mode == "empty" and not final:
            boundary = self.encode("endmark")[0]
            return Generation(token_ids=(boundary,) * max_new_tokens, reason="limit")
        if "What is the ledger total?" in text:
            answer = str(sum(entries))
        elif "Which shelf holds the most units?" in text:
            answer = max(shelves, key=lambda pair: int(pair[1]))[0]
        elif entries:
            answer = str(sum(entries))
        elif shelves:
            answer = max(shelves, key=lambda pair: int(pair[1]))[0]
        else:
            answer = "dunno"
        if self.mode == "wrong":
            answer = "wrong!"
        elif self.mode == "historyfail" and final:
            if text.count("a:") + text.count("A>") >= 2:
                answer = "wrong!"
        elif self.mode == "drift" and final:
            # The render ends with one open assistant prefix;
            # a second one means history turns exist. Scripted
            # history is frozen "Noted." replies; anything else
            # with history is the sharded (self-generated) arm.
            turns = text.count("a:") + text.count("A>")
            if turns >= 2 and "Noted." not in text:
                answer = "wrong!"
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


def _payload(backend=None, formats=FORMATS, seed=0, **stage_kwargs):
    """Payload."""
    backend = backend if backend is not None else _MTBackend()
    stage_kwargs.setdefault("max_context", 8192)
    stage = experiments.multiturn_stage(
        backend,
        formats,
        boundary=tuple(backend.encode("endmark")),
        seed=seed,
        **stage_kwargs,
    )
    return experiments.multiturn_analysis(stage, tuple(f.name for f in formats))


def _record(fixture_id, family, depth, verdicts, details=None):
    """Record."""
    cells = {}
    cell_details = {}
    for arm in multiturn.ARMS:
        verdict = verdicts.get(arm)
        if verdict is None:
            cells[arm] = None
            cell_details[arm] = (details or {}).get(arm, "arm_screened")
        else:
            cells[arm] = {"verdict": verdict}
            cell_details[arm] = ""
    return {
        "fixture": fixture_id,
        "family": family,
        "depth": depth,
        "group": "multiturn",
        "shards": [],
        "values": [],
        "expected": "0",
        "cells": cells,
        "cell_details": cell_details,
    }


def _depth_records(depth, patterns):
    """Records at one depth from (full, sharded, scripted) verdict tuples."""
    records = []
    for index, pattern in enumerate(patterns):
        verdicts = dict(zip(multiturn.ARMS, pattern, strict=True))
        records.append(_record(f"mt-ledger-{depth}-{index:02d}", "ledger", depth, verdicts))
    return records


class TestGenerator(unittest.TestCase):
    """Test generator."""

    def test_constants(self):
        """Test constants."""
        self.assertEqual(multiturn.FAMILIES, ("ledger", "shelf"))
        self.assertEqual(multiturn.DEPTHS, (2, 3))
        self.assertEqual(multiturn.FIXTURES_PER_FAMILY_DEPTH, 12)
        self.assertEqual(multiturn.GENERATION_BUDGET, 5184)
        self.assertEqual(multiturn.MAX_GROUND_RECORDS, 1152)
        self.assertEqual(multiturn.ARMS, ("full", "sharded", "scripted"))
        self.assertEqual(multiturn.SCRIPTED_REPLY, "Noted.")
        # The budget is the forwards formula at the legal
        # candidate cap (24), not a round number.
        per_candidate = 24 * (4 + 5)
        self.assertEqual(per_candidate, 216)
        self.assertEqual(per_candidate * 24, multiturn.GENERATION_BUDGET)

    def test_fixture_set_shape(self):
        """Test fixture set shape."""
        fixtures = multiturn.generate_fixtures(0)
        self.assertEqual(len(fixtures), 48)
        ids = [f["fixture_id"] for f in fixtures]
        self.assertEqual(len(set(ids)), 48)
        for fixture in fixtures:
            self.assertEqual(len(fixture["shards"]), fixture["depth"])
            self.assertTrue(
                fixture["fixture_id"].startswith(f"mt-{fixture['family']}-{fixture['depth']}-")
            )

    def test_deterministic_and_seed_sensitive(self):
        """Test deterministic and seed sensitive."""
        self.assertEqual(multiturn.generate_fixtures(7), multiturn.generate_fixtures(7))
        self.assertNotEqual(multiturn.generate_fixtures(7), multiturn.generate_fixtures(8))
        self.assertNotEqual(multiturn.ground_sha256(7), multiturn.ground_sha256(8))
        self.assertEqual(
            multiturn.protocol_fingerprint(7, "sha256:x"),
            multiturn.protocol_fingerprint(7, "sha256:x"),
        )
        self.assertNotEqual(
            multiturn.protocol_fingerprint(7, "sha256:x"),
            multiturn.protocol_fingerprint(7, "sha256:y"),
        )

    def test_ledger_construction(self):
        """Test ledger construction."""
        for seed in range(30):
            for fixture in multiturn.generate_fixtures(seed):
                if fixture["family"] != "ledger":
                    continue
                values = fixture["values"]
                self.assertEqual(len(values), fixture["depth"])
                total = sum(values)
                self.assertEqual(fixture["expected"], str(total))
                # The entry-equality screen: the expected string
                # is never one of the stated entries (it would
                # be a substring of its own shard text).
                self.assertNotIn(str(total), [str(v) for v in values])
                for value in values:
                    self.assertTrue(1 <= abs(value) <= 999)
                text = multiturn.JOINER.join(fixture["shards"])
                for value in values:
                    self.assertIn(f"Ledger entry: {value}.", text)
                self.assertTrue(text.endswith("What is the ledger total?"))

    def test_shelf_construction(self):
        """Test shelf construction."""
        for seed in range(30):
            for fixture in multiturn.generate_fixtures(seed):
                if fixture["family"] != "shelf":
                    continue
                values = fixture["values"]
                self.assertEqual(len(set(values)), len(values))
                for value in values:
                    self.assertTrue(1 <= value <= 99)
                best = max(range(len(values)), key=lambda i: values[i])
                self.assertEqual(fixture["expected"], "ABC"[best])
                text = multiturn.JOINER.join(fixture["shards"])
                self.assertTrue(text.endswith("Answer with the shelf label."))

    def test_entries_are_str_int_form(self):
        """Test entries are str int form."""
        # No sign-prefix or zero-padded spellings: a "+42" style
        # against a str(int) expected would manufacture a form
        # trap (red-team B's correction to A8).
        for fixture in multiturn.generate_fixtures(3):
            for shard in fixture["shards"]:
                for match in re.findall(r"entry: (\S+)\.", shard):
                    self.assertEqual(match, str(int(match)))


class TestStage(unittest.TestCase):
    """Test stage."""

    def test_perfect_backend_all_pass(self):
        """Test perfect backend all pass."""
        backend = _MTBackend()
        payload = _payload(backend)
        self.assertTrue(payload["available"])
        self.assertEqual(payload["n_candidates"], 2)
        self.assertEqual(payload["n_fixtures"], 48)
        self.assertEqual(backend.calls, 216 * 2)
        for entry in payload["candidates"].values():
            self.assertEqual(entry["classification"], "no_drift")
            for depth in ("2", "3"):
                block = entry["depths"][depth]
                self.assertEqual(block["n_paired"], 24)
                self.assertEqual(
                    block["pass_rates"], {"full": 1.0, "sharded": 1.0, "scripted": 1.0}
                )
        for depth in ("2", "3"):
            for arm in ("full", "sharded"):
                selection = payload["selections"][depth][arm]
                self.assertEqual(selection["decision"], "ambiguous")
                self.assertFalse(selection["withheld"])

    def test_drift_backend_classifies_self_history_drift(self):
        """Test drift backend classifies self history drift."""
        payload = _payload(_MTBackend(mode="drift"), formats=FORMATS[:1])
        entry = payload["candidates"]["mt-a"]
        self.assertEqual(entry["classification"], "self_history_drift")
        block = entry["depths"]["2"]
        self.assertEqual(block["delivery"]["drift"], 24)
        self.assertEqual(block["delivery"]["gains"], 0)
        self.assertEqual(block["self_history"]["sh_only"], 24)
        self.assertEqual(block["self_history"]["sc_only"], 0)
        self.assertEqual(block["self_history"]["net_excess"], 24)
        self.assertEqual(block["pass_rates"]["scripted"], 1.0)
        self.assertEqual(block["pass_rates"]["sharded"], 0.0)

    def test_forged_history_is_screened(self):
        """Test forged history is screened."""
        payload = _payload(_MTBackend(mode="forge"), formats=FORMATS[:1])
        entry = payload["candidates"]["mt-a"]
        for record in entry["records"]:
            self.assertIsNone(record["cells"]["sharded"])
            self.assertEqual(record["cell_details"]["sharded"], "arm_screened")
            self.assertIsNotNone(record["cells"]["full"])
        block = entry["depths"]["2"]
        self.assertEqual(block["n_paired"], 0)
        self.assertEqual(block["completion"]["sharded"]["details"], {"arm_screened": 24})
        self.assertEqual(entry["classification"], "insufficient_evidence")
        selection = payload["selections"]["2"]["sharded"]
        self.assertTrue(selection["withheld"])
        self.assertEqual(selection["decision"], "unavailable")

    def test_empty_intermediate_flagged_not_fatal(self):
        """Test empty intermediate flagged not fatal."""
        payload = _payload(_MTBackend(mode="empty"), formats=FORMATS[:1])
        entry = payload["candidates"]["mt-a"]
        for record in entry["records"]:
            cell = record["cells"]["sharded"]
            self.assertIsNotNone(cell)
            self.assertTrue(cell["intermediate_empty"])
        self.assertEqual(entry["classification"], "no_drift")

    def test_unbounded_intermediate_flagged(self):
        """Test unbounded intermediate flagged."""
        payload = _payload(_MTBackend(mode="noboundary"), formats=FORMATS[:1])
        entry = payload["candidates"]["mt-a"]
        for record in entry["records"]:
            cell = record["cells"]["sharded"]
            self.assertIsNotNone(cell)
            self.assertTrue(cell["intermediate_unbounded"])
            self.assertFalse(cell["verdict"])

    def test_context_exceeded_cells(self):
        """Test context exceeded cells."""
        payload = _payload(_MTBackend(), formats=FORMATS[:1], max_context=4)
        self.assertTrue(payload["available"])
        entry = payload["candidates"]["mt-a"]
        for record in entry["records"]:
            for arm in multiturn.ARMS:
                self.assertIsNone(record["cells"][arm])
                self.assertEqual(record["cell_details"][arm], "arm_context_exceeded")
        self.assertEqual(entry["classification"], "insufficient_evidence")

    def test_cancel_pre_set_fails_closed(self):
        """Test cancel pre set fails closed."""
        event = threading.Event()
        event.set()
        payload = _payload(_MTBackend(), formats=FORMATS[:1], cancel=event)
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "cancelled")
        self.assertNotIn("candidates", payload)

    def test_cancel_mid_run_is_cancelled_not_evaluation_failure(self):
        """Test cancel mid run is cancelled not evaluation failure."""
        # S3 pin: a cancel observed after an evaluate returns is
        # a cancellation, never an arm_evaluation_failed cell.
        event = threading.Event()
        backend = _MTBackend(cancel_after=3)
        backend.cancel_event = event
        payload = _payload(backend, formats=FORMATS[:1], cancel=event)
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "cancelled")

    def test_stage_rejects_bad_arguments(self):
        """Test stage rejects bad arguments."""
        backend = _MTBackend()
        with self.assertRaises(ValueError):
            experiments.multiturn_stage(backend, [], boundary=(1,), max_context=100)
        with self.assertRaises(ValueError):
            experiments.multiturn_stage(backend, (), boundary=(1,), max_context=100)


class TestAggregateClassify(unittest.TestCase):
    """Test aggregate classify."""

    @staticmethod
    def _classify(patterns_by_depth):
        """Classify."""
        records = []
        for depth, patterns in patterns_by_depth.items():
            records.extend(_depth_records(depth, patterns))
        return multiturn.aggregate_candidate(records)

    def test_no_discordance_is_no_drift(self):
        """Test no discordance is no drift."""
        patterns = [(True, True, True)] * 12
        out = self._classify({2: patterns, 3: patterns})
        self.assertEqual(out["classification"], "no_drift")

    def test_single_drift_pair_is_below_threshold(self):
        """Test single drift pair is below threshold."""
        # A2 row 1: one sharded-only failure routes to
        # below_threshold, never to self_history_drift.
        patterns = [(True, True, True)] * 11 + [(True, False, True)]
        out = self._classify({2: patterns, 3: [(True, True, True)] * 12})
        self.assertEqual(out["classification"], "below_threshold")
        block = out["depths"]["2"]
        self.assertEqual(block["delivery"]["drift"], 1)
        self.assertEqual(block["self_history"]["sh_only"], 1)
        self.assertEqual(block["self_history"]["sc_only"], 0)

    def test_scripted_discordance_is_framing_only(self):
        """Test scripted discordance is framing only."""
        # A2 row 2: the scripted control discordant while the
        # sharded arm is below its own bar is a framing effect.
        patterns = [(True, True, True)] * 9 + [(True, True, False)] * 3
        patterns[0] = (True, False, True)
        out = self._classify({2: patterns, 3: [(True, True, True)] * 12})
        self.assertEqual(out["classification"], "framing_only")
        block = out["depths"]["2"]
        self.assertEqual(block["scripted_contrast"]["drift"], 3)

    def test_drift_dominant_is_self_history_drift(self):
        """Test drift dominant is self history drift."""
        patterns = [(True, True, True)] * 7 + [(True, False, True)] * 3
        patterns.append((True, True, False))  # scripted-only failure
        patterns.append((False, True, True))  # full fails: not a full-passer
        out = self._classify({2: patterns, 3: [(True, True, True)] * 12})
        self.assertEqual(out["classification"], "self_history_drift")
        block = out["depths"]["2"]
        self.assertEqual(block["self_history"]["sh_only"], 3)
        self.assertEqual(block["self_history"]["sc_only"], 1)
        self.assertEqual(block["self_history"]["n_full_pass"], 11)
        self.assertEqual(block["self_history"]["net_excess"], 2)

    def test_balanced_discordance_is_framing_only(self):
        """Test balanced discordance is framing only."""
        patterns = [(True, True, True)] * 8 + [(True, False, True)] * 2
        patterns += [(True, True, False)] * 2
        out = self._classify({2: patterns, 3: [(True, True, True)] * 12})
        self.assertEqual(out["classification"], "framing_only")

    def test_below_floor_is_insufficient_evidence(self):
        """Test below floor is insufficient evidence."""
        patterns = [(True, True, True)] * 5 + [(True, None, True)] * 7
        out = self._classify({2: patterns, 3: list(patterns)})
        self.assertEqual(out["classification"], "insufficient_evidence")
        self.assertEqual(out["depths"]["2"]["n_paired"], 5)

    def test_mcnemar_diagnostic_present(self):
        """Test mcnemar diagnostic present."""
        patterns = [(True, True, True)] * 9 + [(True, False, True)] * 3
        out = self._classify({2: patterns, 3: [(True, True, True)] * 12})
        block = out["depths"]["2"]
        self.assertIsNotNone(block["delivery"]["mcnemar_p"])
        self.assertTrue(0.0 < block["delivery"]["mcnemar_p"] <= 1.0)
        self.assertEqual(block["self_history"]["net_excess"], 3)


class TestValidatorBattery(unittest.TestCase):
    """One break at a time against the full-recompute validator."""

    @classmethod
    def setUpClass(cls):
        """Set up the test fixture."""
        cls.payload = _payload(_MTBackend(), formats=FORMATS[:1])

    def _broken(self, mutate):
        """Broken."""
        payload = copy.deepcopy(self.payload)
        mutate(payload)
        return payload

    def test_valid_payload_accepted(self):
        """Test valid payload accepted."""
        self.assertTrue(_valid_multiturn(self.payload))

    def test_unavailable_payload_rejected(self):
        """Test unavailable payload rejected."""
        self.assertFalse(_valid_multiturn({"available": False, "detail": "cancelled"}))

    def test_verdict_flipped(self):
        """Test verdict flipped."""

        def mutate(payload):
            """Mutate."""
            record = payload["candidates"]["mt-a"]["records"][0]
            record["cells"]["full"]["verdict"] = False

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_digest_forged(self):
        """Test digest forged."""

        def mutate(payload):
            """Mutate."""
            record = payload["candidates"]["mt-a"]["records"][0]
            record["cells"]["full"]["answer_digest"] = hashlib.sha256(b"forged").hexdigest()

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_record_values_forged(self):
        """Test record values forged.

        The generator mirror regenerates the set from the
        seed; a stored-values forgery cannot validate (the
        numform F2 lesson).
        """

        def mutate(payload):
            """Mutate."""
            record = payload["candidates"]["mt-a"]["records"][0]
            record["values"][0] += 1

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_classification_tampered(self):
        """Test classification tampered."""

        def mutate(payload):
            """Mutate."""
            payload["candidates"]["mt-a"]["classification"] = "self_history_drift"

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_paired_count_tampered(self):
        """Test paired count tampered."""

        def mutate(payload):
            """Mutate."""
            payload["candidates"]["mt-a"]["depths"]["2"]["n_paired"] += 1

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_pass_rate_tampered(self):
        """Test pass rate tampered."""

        def mutate(payload):
            """Mutate."""
            payload["candidates"]["mt-a"]["depths"]["2"]["pass_rates"]["full"] = 0.5

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_self_history_pair_tampered(self):
        """Test self history pair tampered."""

        def mutate(payload):
            """Mutate."""
            payload["candidates"]["mt-a"]["depths"]["2"]["self_history"]["sh_only"] = 2

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_record_dropped(self):
        """Test record dropped."""

        def mutate(payload):
            """Mutate."""
            payload["candidates"]["mt-a"]["records"].pop()

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_record_extra_key(self):
        """Test record extra key."""

        def mutate(payload):
            """Mutate."""
            payload["candidates"]["mt-a"]["records"][0]["extra"] = 1

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_detail_vocabulary_enforced(self):
        """Test detail vocabulary enforced."""

        def mutate(payload):
            """Mutate."""
            record = payload["candidates"]["mt-a"]["records"][0]
            record["cells"]["full"] = None
            record["cell_details"]["full"] = "arm_melted"

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_cell_with_detail_rejected(self):
        """Test cell with detail rejected."""

        def mutate(payload):
            """Mutate."""
            record = payload["candidates"]["mt-a"]["records"][0]
            record["cell_details"]["full"] = "arm_screened"

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_turns_truncated(self):
        """Test turns truncated."""

        def mutate(payload):
            """Mutate."""
            record = payload["candidates"]["mt-a"]["records"][0]
            record["cells"]["sharded"]["turns"].pop()

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_turns_last_entry_mismatch(self):
        """Test turns last entry mismatch."""

        def mutate(payload):
            """Mutate."""
            record = payload["candidates"]["mt-a"]["records"][0]
            record["cells"]["sharded"]["turns"][-1]["input_tokens"] += 1

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_intermediate_flag_flipped(self):
        """Test intermediate flag flipped."""

        def mutate(payload):
            """Mutate."""
            record = payload["candidates"]["mt-a"]["records"][0]
            record["cells"]["sharded"]["intermediate_unbounded"] = True

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_protocol_fingerprint_zeroed(self):
        """Test protocol fingerprint zeroed."""

        def mutate(payload):
            """Mutate."""
            payload["protocol"]["fingerprint"] = "0" * 64

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_protocol_seed_changed(self):
        """Test protocol seed changed."""

        def mutate(payload):
            """Mutate."""
            payload["protocol"]["seed"] = 99

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_selection_winner_tampered(self):
        """Test selection winner tampered."""

        def mutate(payload):
            """Mutate."""
            payload["selections"]["2"]["full"]["winner"] = "mt-b"

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_counts_tampered(self):
        """Test counts tampered."""

        def mutate(payload):
            """Mutate."""
            payload["n_candidates"] = 7

        self.assertFalse(_valid_multiturn(self._broken(mutate)))


class TestDiff(unittest.TestCase):
    """Test diff."""

    def test_self_diff_unchanged(self):
        """Test self diff unchanged."""
        payload = _payload(_MTBackend(), formats=FORMATS[:1])
        diff = diff_multiturn(payload, payload)
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])
        self.assertTrue(diff["comparable"])

    def test_foreign_backend_not_comparable(self):
        """Test foreign backend not comparable."""
        payload = _payload(_MTBackend(), formats=FORMATS[:1])
        other_backend = _MTBackend()
        other_backend.fingerprint = "sha256:other"
        other = _payload(other_backend, formats=FORMATS[:1])
        diff = diff_multiturn(payload, other)
        self.assertTrue(diff["backend_drift"])
        self.assertTrue(diff["protocol_drift"])
        self.assertFalse(diff["comparable"])
        self.assertTrue(diff["changed"])

    def test_classification_drift_detected(self):
        """Test classification drift detected."""
        calm = _payload(_MTBackend(), formats=FORMATS[:1])
        stormy = _payload(_MTBackend(mode="drift"), formats=FORMATS[:1])
        diff = diff_multiturn(calm, stormy)
        self.assertTrue(diff["changed"])
        metrics = {entry["metric"] for entry in diff["classification_drift"]}
        self.assertIn("classification", metrics)
        rate_metrics = {entry["metric"] for entry in diff["depth_drift"]}
        self.assertIn("depth_2_sharded_rate", rate_metrics)

    def test_absent_sides(self):
        """Test absent sides."""
        payload = _payload(_MTBackend(), formats=FORMATS[:1])
        self.assertEqual(diff_multiturn(None, payload)["available"], "new")
        self.assertEqual(diff_multiturn(payload, None)["available"], "old")
        self.assertEqual(diff_multiturn(None, None)["available"], "neither")


class TestGapCoverage(unittest.TestCase):
    """Branches the main battery reaches only sideways: the
    seed type guard, defensive cell access, both-fail
    self-history outcomes, in-turn cancel, evaluation
    failure, the record budget, the stage screen, analysis
    over an unavailable candidate, and mismatched completed
    sets across candidates.
    """

    def test_seed_must_be_int(self):
        """Test seed must be int."""
        with self.assertRaises(ValueError):
            multiturn.generate_fixtures("0")  # type: ignore[arg-type]

    def test_pair_counts_skip_incomplete_records(self):
        """Test pair counts skip incomplete records."""
        record = _record("mt-ledger-2-00", "ledger", 2, {"full": True})
        record["cells"] = None
        counts = multiturn._pair_counts([record], "full", "sharded")
        self.assertEqual(counts["n_paired"], 0)
        out = multiturn.aggregate_candidate([record])
        self.assertEqual(out["depths"]["2"]["n_paired"], 0)

    def test_both_history_arms_fail(self):
        """Test both history arms fail."""
        payload = _payload(_MTBackend(mode="historyfail"), formats=FORMATS[:1])
        entry = payload["candidates"]["mt-a"]
        block = entry["depths"]["2"]
        self.assertEqual(block["self_history"]["both_fail"], 24)
        self.assertEqual(block["self_history"]["net_excess"], 0)
        self.assertTrue(_valid_multiturn(payload))

    def test_cancel_set_during_encode_unwinds_stage(self):
        """Test cancel set during encode unwinds stage."""
        event = threading.Event()

        class _EncodeCancelBackend(_MTBackend):
            """Encode cancel backend."""

            def encode(self, text):
                """Encode."""
                event.set()
                return super().encode(text)

        payload = _payload(_EncodeCancelBackend(), formats=FORMATS[:1], cancel=event)
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "cancelled")

    def test_evaluation_failed_cells(self):
        """Test evaluation failed cells."""
        payload = _payload(_MTBackend(mode="explode"), formats=FORMATS[:1])
        entry = payload["candidates"]["mt-a"]
        for record in entry["records"]:
            for arm in multiturn.ARMS:
                self.assertIsNone(record["cells"][arm])
                self.assertEqual(record["cell_details"][arm], "arm_evaluation_failed")
        self.assertEqual(entry["classification"], "insufficient_evidence")

    def test_record_budget_exceeded(self):
        """Test record budget exceeded."""
        formats = tuple(
            Format(
                name=f"mt-{i}",
                bos="",
                user_prefix="u:",
                assistant_prefix="a:",
                turn_end="</s>",
            )
            for i in range(25)
        )
        backend = _MTBackend()
        stage = experiments.multiturn_stage(
            backend,
            formats,
            boundary=tuple(backend.encode("endmark")),
            max_context=8192,
            seed=0,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "multiturn_record_budget_exceeded")

    def test_stage_screen_failure_fails_closed(self):
        """Test stage screen failure fails closed."""
        # A boundary whose decoded text appears inside the
        # frozen template text would let a generation forge
        # protocol structure; the stage fails closed before
        # any generation.
        backend = _MTBackend()
        stage = experiments.multiturn_stage(
            backend,
            FORMATS[:1],
            boundary=tuple(backend.encode("Ledger")),
            max_context=8192,
            seed=0,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "multiturn_screen_failed")

    def test_analysis_over_unavailable_candidate(self):
        """Test analysis over unavailable candidate."""
        backend = _MTBackend()
        stage = experiments.multiturn_stage(
            backend,
            FORMATS[:1],
            boundary=tuple(backend.encode("endmark")),
            max_context=8192,
            seed=0,
        )
        stage["candidates"] = {"mt-a": {"status": "unavailable", "detail": "gone", "records": []}}
        payload = experiments.multiturn_analysis(stage, ("mt-a",))
        self.assertTrue(payload["available"])
        self.assertEqual(payload["candidates"]["mt-a"]["status"], "unavailable")
        self.assertTrue(payload["selections"]["2"]["full"]["withheld"])

    def test_mismatched_completed_sets_withheld(self):
        """Test mismatched completed sets withheld."""
        # One candidate complete, the other fully screened:
        # the completed sets differ across candidates, which
        # is its own withholding branch in the analysis (and
        # in the mirror). Merged by hand because a single
        # stage run shares one backend across formats.
        calm = _payload(_MTBackend(), formats=FORMATS[:1])
        screened = _payload(_MTBackend(mode="forge"), formats=FORMATS[1:])
        merged_stage = {
            "available": True,
            "protocol": calm["protocol"],
            "candidates": {
                **calm["candidates"],
                **screened["candidates"],
            },
        }
        merged = experiments.multiturn_analysis(merged_stage, ("mt-a", "mt-b"))
        self.assertTrue(merged["selections"]["2"]["sharded"]["withheld"])
        self.assertEqual(merged["candidates"]["mt-a"]["depths"]["2"]["n_paired"], 24)
        self.assertEqual(merged["candidates"]["mt-b"]["depths"]["2"]["n_paired"], 0)


class TestMirrorBatteryAdditions(unittest.TestCase):
    """Mirror-side reject branches: mutations that pass the
    producer-shaped checks and must die inside the
    diffreports mirror's own cell/recompute guards.
    """

    @classmethod
    def setUpClass(cls):
        """Set up the test fixture."""
        cls.payload = _payload(_MTBackend(), formats=FORMATS[:1])

    def _broken(self, mutate):
        """Broken."""
        payload = copy.deepcopy(self.payload)
        mutate(payload)
        return payload

    def test_cell_input_tokens_negative(self):
        """Test cell input tokens negative."""

        def mutate(payload):
            """Mutate."""
            record = payload["candidates"]["mt-a"]["records"][0]
            record["cells"]["full"]["input_tokens"] = -1

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_cell_boundary_text_match_wrong_type(self):
        """Test cell boundary text match wrong type."""

        def mutate(payload):
            """Mutate."""
            record = payload["candidates"]["mt-a"]["records"][0]
            record["cells"]["full"]["boundary_text_match"] = "yes"

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_cell_verdict_not_bool(self):
        """Test cell verdict not bool."""

        def mutate(payload):
            """Mutate."""
            record = payload["candidates"]["mt-a"]["records"][0]
            record["cells"]["full"]["verdict"] = 1

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_cell_digest_not_hex(self):
        """Test cell digest not hex."""

        def mutate(payload):
            """Mutate."""
            record = payload["candidates"]["mt-a"]["records"][0]
            record["cells"]["full"]["answer_digest"] = "z" * 64

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_turn_entry_boundary_not_bool(self):
        """Test turn entry boundary not bool."""

        def mutate(payload):
            """Mutate."""
            record = payload["candidates"]["mt-a"]["records"][0]
            record["cells"]["sharded"]["turns"][0]["boundary_found"] = "x"

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_candidates_empty(self):
        """Test candidates empty."""

        def mutate(payload):
            """Mutate."""
            payload["candidates"] = {}

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_entry_status_unavailable_rejected(self):
        """Test entry status unavailable rejected."""

        def mutate(payload):
            """Mutate."""
            entry = payload["candidates"]["mt-a"]
            entry["status"] = "unavailable"
            entry["records"] = []

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_record_cells_not_dict(self):
        """Test record cells not dict."""

        def mutate(payload):
            """Mutate."""
            payload["candidates"]["mt-a"]["records"][0]["cells"] = []

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_protocol_seed_missing(self):
        """Test protocol seed missing."""

        def mutate(payload):
            """Mutate."""
            del payload["protocol"]["seed"]

        self.assertFalse(_valid_multiturn(self._broken(mutate)))

    def test_records_byte_cap_engages(self):
        """Test records byte cap engages."""
        # Every field is individually legal (the int stays
        # under the str-conversion limit, turns agree with
        # their cells, aggregates ignore input_tokens), so
        # only the records byte cap can reject this payload:
        # 24 candidates x 48 records of ~26 KB of digits each
        # is far past 12 MiB of ground records.
        payload = copy.deepcopy(self.payload)
        giant = 10**4000
        entry = payload["candidates"]["mt-a"]
        for record in entry["records"]:
            for arm in multiturn.ARMS:
                cell = record["cells"][arm]
                cell["input_tokens"] = giant
                if arm == "sharded":
                    for turn in cell["turns"]:
                        turn["input_tokens"] = giant
        payload["candidates"] = {f"cand-{i:02d}": copy.deepcopy(entry) for i in range(24)}
        payload["n_candidates"] = 24
        self.assertFalse(_valid_multiturn(payload))


class TestReportingAndCli(unittest.TestCase):
    """Test reporting and cli."""

    def test_section_renders(self):
        """Test section renders."""
        payload = _payload(_MTBackend(mode="drift"), formats=FORMATS[:1])
        html = render_html({"decision": "selected", "multiturn": payload})
        self.assertIn("Multi-turn drift", html)
        self.assertIn('id="multiturn"', html)
        self.assertIn("self_history_drift", html)

    def test_section_absent_without_payload(self):
        """Test section absent without payload."""
        html = render_html({"decision": "selected"})
        self.assertNotIn('id="multiturn"', html)

    def test_section_defensive_shapes(self):
        """Test section defensive shapes."""
        # Hostile payload shapes degrade to rows or nothing,
        # never to an exception (the renderer's guards are
        # the boundary).
        html = render_html({"multiturn": {"available": True, "candidates": "junk"}})
        self.assertNotIn('id="multiturn"', html)
        payload = {
            "available": True,
            "candidates": {
                "gone": {"status": "unavailable"},
                "junk": "not-a-dict",
                "bare": {"status": "available", "classification": "no_drift"},
                "odd": {
                    "status": "available",
                    "classification": "no_drift",
                    "depths": {"2": {"pass_rates": "nope", "delivery": {}}},
                },
            },
        }
        html = render_html({"decision": "selected", "multiturn": payload})
        self.assertIn('id="multiturn"', html)
        self.assertIn("unavailable", html)

    def test_cli_helper_attaches_payload(self):
        """Test cli helper attaches payload."""
        backend = _MTBackend()
        result: dict = {"decision": "selected"}
        cli._run_multiturn_probe(
            backend,
            FORMATS[:1],
            result,
            boundary=tuple(backend.encode("endmark")),
            boundary_map=None,
            max_new_tokens=64,
            max_context=8192,
            seed=0,
            cancel=None,
        )
        self.assertTrue(result["multiturn"]["available"])
        self.assertEqual(result["multiturn"]["n_fixtures"], 48)

    def test_cli_helper_never_raises(self):
        """Test cli helper never raises."""
        result: dict = {"decision": "selected"}
        with mock.patch.object(experiments, "multiturn_stage", side_effect=RuntimeError("boom")):
            cli._run_multiturn_probe(
                _MTBackend(),
                FORMATS[:1],
                result,
                boundary=(1,),
                boundary_map=None,
                max_new_tokens=8,
                max_context=100,
                seed=0,
                cancel=None,
            )
        self.assertEqual(
            result["multiturn"],
            {"available": False, "detail": "multiturn_unavailable"},
        )


if __name__ == "__main__":
    unittest.main()
