"""Cycle 173: determinism self-audit (``verify --determinism``).

The audit re-executes a recorded run's search observations against a
backend and compares replayed token IDs with the recorded ones. The
pre-registered falsifiable claims (docs/DETERMINISM_AUDIT_DESIGN.md):

- F1: a deterministic backend reproduces every recorded completed
  search cell token-exactly -> verdict "deterministic" + certificate.
- F2: a call-count-dependent backend diverges, and first_divergence
  matches an independent oracle.
- F3: audit seeds equal core.evaluate's seeds for the same artifact
  (seed-recording double; expectations from an inline hashlib oracle,
  never from the shared helper -- that would test a copy against
  itself).
- F4: fingerprint mismatch -> zero generate calls.
- F5: hostile artifacts yield fixed-vocabulary output, never a raise.
- F6: one flipped recorded token diverges at exactly that index; one
  truncated recorded output is a length skip, never a divergence.
- F7: an aborted search earns no certificate even when every recorded
  completed cell reproduces.

Red-team findings pinned here: recorded-length validation before
blame (A1), certificate binding via cells_digest (A2), vocabulary
validation (A3), the attempted-cell budget (A4/B2), error cells map
to an unavailable check, never a failed one (A6), partial evidence
blocks the certificate (A8), byte-exact fixture IDs (A9), and no
replay memoisation -- duplicate cells are re-executed individually,
because a call-count backend is exactly what the audit must catch
(B1). Three tempdir mutation gates pin the comparison, the
fingerprint gate, and the length invariant.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest import mock

from model_protocol_lab import determinism, verify
from model_protocol_lab.catalog import CATALOG
from model_protocol_lab.core import (
    Fixture,
    Generation,
    Message,
    evaluate,
    request_seed_for,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    import importlib.util as _ilu

    HAS_HF = all(
        _ilu.find_spec(name) is not None
        for name in ("torch", "transformers", "safetensors", "tokenizers")
    )
except Exception:  # noqa: BLE001 -- availability probe only
    HAS_HF = False


def _pure_output(input_ids: tuple[int, ...], seed: int, cap: int) -> tuple[int, ...]:
    """The deterministic doubles' generation rule (a pure function)."""
    return tuple((input_ids[i % len(input_ids)] * 31 + seed % 97 + i * 7) % 50 for i in range(cap))


class _PureBackend:
    """Deterministic double: output depends only on (input_ids, seed)."""

    vocab_size = 1000

    def __init__(self, fingerprint: str = "testfp") -> None:
        self.fingerprint = fingerprint
        self.calls: list[tuple[tuple[int, ...], int, int]] = []

    @staticmethod
    def encode(text: str) -> tuple[int, ...]:
        """Encode."""
        return tuple(text.encode("utf-8"))

    @staticmethod
    def decode(token_ids: tuple[int, ...]) -> str:
        """Decode."""
        return bytes(t % 256 for t in token_ids).decode("utf-8", errors="replace")

    def generate(
        self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> Generation:
        """Generate."""
        self.calls.append((tuple(input_ids), max_new_tokens, seed))
        return Generation(_pure_output(tuple(input_ids), seed, max_new_tokens), "limit")


class _FlakyBackend(_PureBackend):
    """Call-count-dependent double: correct on call 1, shifted after."""

    def generate(
        self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> Generation:
        """Generate."""
        result = super().generate(
            input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel
        )
        if len(self.calls) == 1:
            return result
        shifted = list(result.token_ids)
        shifted[2] = (shifted[2] + 1) % 50
        return Generation(tuple(shifted), "limit")


class _RaisingBackend(_PureBackend):
    """Raising backend."""

    @staticmethod
    def generate(input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        # type: ignore[no-untyped-def]
        raise RuntimeError("boom -- private backend text")


class _InterruptingBackend(_PureBackend):
    """Interrupting backend."""

    @staticmethod
    def generate(input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        # type: ignore[no-untyped-def]
        raise KeyboardInterrupt


class _ReasonBackend(_PureBackend):
    """Returns a scripted Generation (or non-Generation) per mode."""

    def __init__(self, mode: str, fingerprint: str = "testfp") -> None:
        super().__init__(fingerprint)
        self.mode = mode

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        # type: ignore[no-untyped-def]
        self.calls.append((tuple(input_ids), max_new_tokens, seed))
        if self.mode == "error_allowlisted":
            return Generation((), "error", "invalid_seed")
        if self.mode == "error_arbitrary":
            return Generation((), "error", "weird backend prose")
        if self.mode == "error_context":
            return Generation((), "error", "context_limit_exceeded")
        if self.mode == "cancelled":
            return Generation((), "cancelled")
        if self.mode == "short_limit":
            return Generation((1, 2), "limit")
        if self.mode == "not_a_generation":
            return {"token_ids": (1, 2, 3, 4), "reason": "limit"}
        raise AssertionError(self.mode)


def _obs(
    fixture: str = "f1",
    input_ids: tuple[int, ...] = (1, 2, 3),
    output_ids: tuple[int, ...] = (4, 5, 6, 7),
    status: str = "completed",
) -> dict:
    """Obs."""
    return {
        "candidate": "cand-a",
        "fixture": fixture,
        "input_ids": list(input_ids),
        "output_ids": list(output_ids),
        "answer": "",
        "exact_match": False,
        "boundary_found": False,
        "boundary_text_match": None,
        "raw_reason": "limit" if status == "completed" else None,
        "error": None,
        "status": status,
        "input_text": "",
    }


def _artifact(
    entries: list[dict],
    *,
    seed: int = 7,
    cap: int = 4,
    fingerprint: str = "testfp",
    decision: str = "no_candidate",
) -> dict:
    """Artifact."""
    return {
        "schema_version": 1,
        "seed": seed,
        "max_new_tokens": cap,
        "backend_fingerprint": fingerprint,
        "decision": decision,
        "search": entries,
        "held_out": None,
        "selected_candidate": None,
        "boundary_token_ids": [9],
        "failure_taxonomy": {},
    }


def _entry(
    observations: list[dict] | None,
    *,
    name: str = "cand-a",
    eligible: bool = True,
    planned: int = 2,
) -> dict:
    """Entry."""
    entry: dict = {
        "format": {"name": name},
        "summary": {
            "aggregate_eligible": eligible,
            "planned": planned,
            "exact_and_boundary_passes": 1,
            "status": "completed",
        },
    }
    if observations is not None:
        entry["observations"] = observations
    return entry


def _recorded_artifact(backend: _PureBackend, *, seed: int = 7, cap: int = 4) -> dict:
    """An artifact whose recorded outputs are the double's own outputs."""
    fixtures = (("f1", (1, 2, 3)), ("f2", (9, 8)))
    observations = []
    for fixture, input_ids in fixtures:
        request_seed = int.from_bytes(
            hashlib.sha256(f"{seed}\0{fixture}".encode()).digest()[:4], "big"
        )
        observations.append(
            _obs(
                fixture=fixture,
                input_ids=input_ids,
                output_ids=_pure_output(input_ids, request_seed, cap),
            )
        )
    return _artifact([_entry(observations)], seed=seed, cap=cap)


class TestCompareTokenIds(unittest.TestCase):
    """Test compare token ids."""

    def test_compare_battery_against_oracle(self) -> None:
        """Test compare battery against oracle."""
        cases = [
            ((1, 2, 3), (1, 2, 3), True, None, 3),
            ((1, 2, 3), (1, 9, 3), False, 1, 1),
            ((1, 2, 3), (1, 2), False, 2, 2),
            ((1, 2), (1, 2, 3), False, 2, 2),
            ((5,), (6,), False, 0, 0),
            ((), (), True, None, 0),
            ((), (1,), False, 0, 0),
        ]
        for recorded, replayed, match, divergence, prefix in cases:
            with self.subTest(recorded=recorded, replayed=replayed):
                result = determinism.compare_token_ids(recorded, replayed)
                self.assertEqual(result["match"], match)
                self.assertEqual(result["first_divergence"], divergence)
                self.assertEqual(result["shared_prefix"], prefix)
                self.assertEqual(result["recorded_length"], len(recorded))
                self.assertEqual(result["replayed_length"], len(replayed))


class TestRequestSeedOracle(unittest.TestCase):
    """Test request seed oracle."""

    def test_helper_matches_independent_oracle(self) -> None:
        """Test helper matches independent oracle."""
        # F3's expectation: the seed formula recomputed inline from
        # hashlib -- never via request_seed_for itself.
        ids = ["f1", "café", "café", "a\0b", "\ud800x", "", "x" * 500]
        for seed in (0, 7, 2**64 - 1):
            for case_id in ids:
                expected = int.from_bytes(
                    hashlib.sha256(
                        f"{seed}\0{case_id}".encode("utf-8", errors="surrogatepass")
                    ).digest()[:4],
                    "big",
                )
                self.assertEqual(request_seed_for(seed, case_id), expected)

    def test_surfacenoise_derivation_drift_guard(self) -> None:
        """Test surfacenoise derivation drift guard."""
        # B3: surfacenoise carries a pre-existing private copy of the
        # derivation; pin the two equal so any drift is loud.
        from model_protocol_lab import surfacenoise

        for case_id in ("f1", "café", "café", "\ud800x", ""):
            self.assertEqual(
                surfacenoise._request_seed(7, case_id),  # noqa: SLF001
                request_seed_for(7, case_id),
            )


class TestPrecheck(unittest.TestCase):
    """Test precheck."""

    def test_header_problems(self) -> None:
        """Test header problems."""
        good = _recorded_artifact(_PureBackend())
        self.assertIsNone(determinism.precheck(good))
        self.assertEqual(determinism.precheck(None), "artifact_malformed")
        self.assertEqual(determinism.precheck([1]), "artifact_malformed")
        for mutate, code in [
            ({"schema_version": 2}, "unsupported_schema_version"),
            ({"schema_version": None}, "unsupported_schema_version"),
            ({"schema_version": True}, "unsupported_schema_version"),
            ({"seed": True}, "artifact_malformed"),
            ({"seed": -1}, "artifact_malformed"),
            ({"seed": 2**64}, "artifact_malformed"),
            ({"seed": "7"}, "artifact_malformed"),
            ({"max_new_tokens": 0}, "artifact_malformed"),
            ({"max_new_tokens": 4097}, "artifact_malformed"),
            ({"max_new_tokens": True}, "artifact_malformed"),
            ({"backend_fingerprint": ""}, "artifact_malformed"),
            ({"backend_fingerprint": 9}, "artifact_malformed"),
            ({"search": {}}, "artifact_malformed"),
        ]:
            bad = dict(good)
            bad.update(mutate)
            self.assertEqual(determinism.precheck(bad), code, mutate)

    def test_stripped_and_empty(self) -> None:
        """Test stripped and empty."""
        stripped = _artifact([_entry(None)])
        self.assertEqual(determinism.precheck(stripped), "observations_stripped")
        empty = _artifact([_entry([])])
        self.assertEqual(determinism.precheck(empty), "no_replayable_cells")
        not_completed = _artifact([_entry([_obs(status="failed")])])
        self.assertEqual(determinism.precheck(not_completed), "no_replayable_cells")
        no_entries = _artifact([])
        self.assertEqual(determinism.precheck(no_entries), "no_replayable_cells")


class TestAuditDeterministic(unittest.TestCase):
    """Test audit deterministic."""

    def test_f1_certificate_and_digest(self) -> None:
        """Test f1 certificate and digest."""
        artifact = _recorded_artifact(_PureBackend())
        audit = determinism.audit_determinism(
            artifact, _PureBackend(), replay={"device": "cpu", "max_context": 2048}
        )
        self.assertTrue(audit["available"])
        self.assertEqual(audit["verdict"], "deterministic")
        self.assertEqual(audit["n_cells"], 2)
        self.assertEqual(audit["n_attempted"], 2)
        self.assertEqual(audit["n_reproduced"], 2)
        self.assertEqual(audit["n_diverged"], 0)
        certificate = audit["certificate"]
        self.assertIsNotNone(certificate)
        self.assertTrue(certificate["unsigned"])
        self.assertEqual(certificate["issuer"], "self")
        self.assertEqual(certificate["backend_fingerprint"], "testfp")
        self.assertEqual(certificate["seed"], 7)
        self.assertEqual(certificate["max_new_tokens"], 4)
        self.assertEqual(certificate["replay"], {"device": "cpu", "max_context": 2048})
        # The digest is recomputed independently from the recorded cells.
        records = [
            {
                "candidate": "cand-a",
                "fixture": obs["fixture"],
                "input_ids": obs["input_ids"],
                "output_ids": obs["output_ids"],
            }
            for obs in artifact["search"][0]["observations"]
        ]
        payload = json.dumps(records, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        self.assertEqual(
            certificate["cells_digest"],
            hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        )

    def test_held_out_is_never_replayed(self) -> None:
        """Test held out is never replayed."""
        artifact = _recorded_artifact(_PureBackend())
        artifact["held_out"] = {
            "summary": {"aggregate_eligible": True},
            "observations": [_obs(fixture="h1", output_ids=(1, 1, 1, 1))],
        }
        backend = _PureBackend()
        audit = determinism.audit_determinism(artifact, backend)
        self.assertEqual(audit["verdict"], "deterministic")
        self.assertEqual(audit["n_cells"], 2)
        self.assertEqual(len(backend.calls), 2)

    def test_duplicate_cells_are_reexecuted_individually(self) -> None:
        """Test duplicate cells are reexecuted individually."""
        # B1: no memoisation -- two entries recording the same
        # (fixture, input) cell cost two generate calls.
        artifact = _recorded_artifact(_PureBackend())
        artifact["search"].append(
            _entry(list(artifact["search"][0]["observations"]), name="cand-b")
        )
        backend = _PureBackend()
        audit = determinism.audit_determinism(artifact, backend)
        self.assertEqual(audit["n_cells"], 4)
        self.assertEqual(len(backend.calls), 4)
        self.assertEqual(audit["verdict"], "deterministic")


class TestAuditEndToEnd(unittest.TestCase):
    """Test audit end to end."""

    def test_f3_seeds_match_evaluate_and_oracle(self) -> None:
        """Test f3 seeds match evaluate and oracle."""
        recorder = _PureBackend()
        fmt = CATALOG["chatml"]
        cases = (
            Fixture(
                id="fix-one",
                group="g",
                expected="x",
                messages=(Message("user", "Say x."),),
            ),
            Fixture(
                id="fix-two",
                group="g",
                expected="y",
                messages=(Message("user", "Say y."),),
            ),
        )
        report = evaluate(
            recorder,
            fmt,
            cases,
            boundary=(999,),
            max_new_tokens=6,
            seed=11,
            cancel=Event(),
        )
        self.assertEqual(report.status, "completed")
        entry = {
            "format": {"name": fmt.name},
            "summary": {
                "aggregate_eligible": True,
                "planned": 2,
                "exact_and_boundary_passes": 0,
                "status": "completed",
            },
            "observations": [asdict(o) for o in report.observations],
        }
        artifact = _artifact([entry], seed=11, cap=6, decision="no_candidate")
        artifact = json.loads(json.dumps(artifact))  # stored form: lists
        replay_backend = _PureBackend()
        audit = determinism.audit_determinism(artifact, replay_backend)
        self.assertEqual(audit["verdict"], "deterministic")
        evaluate_seeds = [call[2] for call in recorder.calls]
        audit_seeds = [call[2] for call in replay_backend.calls]
        self.assertEqual(audit_seeds, evaluate_seeds)
        oracle = [
            int.from_bytes(hashlib.sha256(f"11\0{case.id}".encode()).digest()[:4], "big")
            for case in cases
        ]
        self.assertEqual(audit_seeds, oracle)


class TestAuditDivergence(unittest.TestCase):
    """Test audit divergence."""

    def test_f6_flipped_token_diverges_at_exact_index(self) -> None:
        """Test f6 flipped token diverges at exact index."""
        artifact = _recorded_artifact(_PureBackend())
        artifact["search"][0]["observations"][0]["output_ids"][2] += 1
        audit = determinism.audit_determinism(artifact, _PureBackend())
        self.assertEqual(audit["verdict"], "diverged")
        self.assertIsNone(audit["certificate"])
        cell = audit["cells"][0]
        self.assertEqual(cell["status"], "diverged")
        self.assertEqual(cell["first_divergence"], 2)
        self.assertEqual(cell["shared_prefix"], 2)
        self.assertEqual(audit["cells"][1]["status"], "reproduced")

    def test_f2_flaky_backend_divergence_matches_oracle(self) -> None:
        """Test f2 flaky backend divergence matches oracle."""
        artifact = _recorded_artifact(_PureBackend())
        audit = determinism.audit_determinism(artifact, _FlakyBackend())
        self.assertEqual(audit["verdict"], "diverged")
        self.assertEqual(audit["cells"][0]["status"], "reproduced")
        cell = audit["cells"][1]
        self.assertEqual(cell["status"], "diverged")
        self.assertEqual(cell["first_divergence"], 2)

    def test_f6_truncated_output_is_a_length_skip_not_divergence(self) -> None:
        """Test f6 truncated output is a length skip not divergence."""
        artifact = _recorded_artifact(_PureBackend())
        artifact["search"][0]["observations"][0]["output_ids"] = [1, 2, 3]
        audit = determinism.audit_determinism(artifact, _PureBackend())
        cell = audit["cells"][0]
        self.assertEqual(cell["status"], "skipped")
        self.assertEqual(cell["detail"], "recorded_length_inconsistent")
        self.assertEqual(audit["n_diverged"], 0)
        # A blocking skip caps the verdict: no certificate.
        self.assertEqual(audit["verdict"], "inconclusive")
        self.assertIsNone(audit["certificate"])


class TestAuditGates(unittest.TestCase):
    """Test audit gates."""

    def test_f4_fingerprint_mismatch_makes_zero_calls(self) -> None:
        """Test f4 fingerprint mismatch makes zero calls."""
        artifact = _recorded_artifact(_PureBackend())
        backend = _PureBackend(fingerprint="otherfp")
        audit = determinism.audit_determinism(artifact, backend)
        self.assertFalse(audit["available"])
        self.assertEqual(audit["detail"], "backend_fingerprint_mismatch")
        self.assertEqual(audit["replay_backend_fingerprint"], "otherfp")
        self.assertEqual(len(backend.calls), 0)

    def test_missing_fingerprint_attribute_is_a_mismatch(self) -> None:
        """Test missing fingerprint attribute is a mismatch."""
        artifact = _recorded_artifact(_PureBackend())
        backend = _PureBackend()
        del backend.fingerprint
        audit = determinism.audit_determinism(artifact, backend)
        self.assertFalse(audit["available"])
        self.assertEqual(audit["detail"], "backend_fingerprint_mismatch")
        self.assertIsNone(audit["replay_backend_fingerprint"])

    def test_out_of_vocab_ids_are_skips_not_blame(self) -> None:
        """Test out of vocab ids are skips not blame."""
        artifact = _recorded_artifact(_PureBackend())
        artifact["search"][0]["observations"][0]["output_ids"][1] = 5000
        artifact["search"][0]["observations"][1]["input_ids"] = [1, 5000]
        audit = determinism.audit_determinism(artifact, _PureBackend())
        for cell in audit["cells"]:
            self.assertEqual(cell["status"], "skipped")
            self.assertEqual(cell["detail"], "recorded_ids_out_of_vocab")
        self.assertEqual(audit["verdict"], "inconclusive")

    def test_backend_without_vocab_skips_the_vocab_check(self) -> None:
        """Test backend without vocab skips the vocab check."""

        class NoVocab(_PureBackend):
            """No vocab."""

            vocab_size = None  # type: ignore[assignment]

        artifact = _recorded_artifact(_PureBackend())
        artifact["search"][0]["observations"][0]["output_ids"][1] = 5000
        backend = NoVocab()
        audit = determinism.audit_determinism(artifact, backend)
        # No bound is knowable, so the cell is attempted and diverges:
        # the audit reports what it could measure, not a guess.
        self.assertEqual(audit["cells"][0]["status"], "diverged")

    def test_vocab_method_and_hostile_vocab(self) -> None:
        """Test vocab method and hostile vocab."""

        class MethodVocab(_PureBackend):
            """Method vocab."""

            @staticmethod
            def vocab_size():
                """Vocab size."""
                # type: ignore[no-untyped-def]
                return 1000

        class HostileVocab(_PureBackend):
            """Hostile vocab."""

            @property
            def vocab_size(self):  # type: ignore[no-untyped-def]
                raise RuntimeError("nope")

        artifact = _recorded_artifact(_PureBackend())
        self.assertEqual(
            determinism.audit_determinism(artifact, MethodVocab())["verdict"],
            "deterministic",
        )
        self.assertEqual(
            determinism.audit_determinism(artifact, HostileVocab())["verdict"],
            "deterministic",
        )

    def test_f7_aborted_search_earns_no_certificate(self) -> None:
        """Test f7 aborted search earns no certificate."""
        artifact = _recorded_artifact(_PureBackend())
        artifact["decision"] = "cancelled"
        audit = determinism.audit_determinism(artifact, _PureBackend())
        self.assertEqual(audit["n_reproduced"], 2)
        self.assertTrue(audit["search_incomplete"])
        self.assertEqual(audit["verdict"], "inconclusive")
        self.assertIsNone(audit["certificate"])

    def test_non_eligible_entry_blocks_certificate(self) -> None:
        """Test non eligible entry blocks certificate."""
        artifact = _recorded_artifact(_PureBackend())
        artifact["search"][0]["summary"]["aggregate_eligible"] = False
        artifact["decision"] = "failed"
        audit = determinism.audit_determinism(artifact, _PureBackend())
        self.assertEqual(audit["verdict"], "inconclusive")
        self.assertIsNone(audit["certificate"])

    def test_partial_strip_blocks_certificate_but_replays_the_rest(self) -> None:
        """Test partial strip blocks certificate but replays the rest."""
        artifact = _recorded_artifact(_PureBackend())
        artifact["search"].append(_entry(None, name="cand-b"))
        audit = determinism.audit_determinism(artifact, _PureBackend())
        self.assertEqual(audit["n_entries_stripped"], 1)
        self.assertEqual(audit["n_reproduced"], 2)
        self.assertEqual(audit["verdict"], "inconclusive")
        self.assertIsNone(audit["certificate"])

    def test_not_completed_cells_are_scope_not_gaps(self) -> None:
        """Test not completed cells are scope not gaps."""
        artifact = _recorded_artifact(_PureBackend())
        artifact["search"][0]["observations"].append(
            _obs(fixture="f3", status="failed", output_ids=())
        )
        audit = determinism.audit_determinism(artifact, _PureBackend())
        self.assertEqual(audit["verdict"], "deterministic")
        self.assertEqual(audit["n_cells"], 3)
        self.assertEqual(audit["certificate"]["n_not_completed_in_original"], 1)
        skipped = [c for c in audit["cells"] if c["detail"] == "observation_not_completed"]
        self.assertEqual(len(skipped), 1)


class TestAuditErrors(unittest.TestCase):
    """Test audit errors."""

    def test_raising_backend_is_an_error_cell_not_a_crash(self) -> None:
        """Test raising backend is an error cell not a crash."""
        artifact = _recorded_artifact(_PureBackend())
        audit = determinism.audit_determinism(artifact, _RaisingBackend())
        self.assertEqual(audit["n_error"], 2)
        self.assertEqual(audit["verdict"], "inconclusive")
        for cell in audit["cells"]:
            self.assertEqual(cell["status"], "error")
            self.assertEqual(cell["detail"], "generate_failed")
        # Backend prose never reaches the report.
        self.assertNotIn("boom", json.dumps(audit))

    def test_keyboard_interrupt_propagates(self) -> None:
        """Test keyboard interrupt propagates."""
        artifact = _recorded_artifact(_PureBackend())
        with self.assertRaises(KeyboardInterrupt):
            determinism.audit_determinism(artifact, _InterruptingBackend())

    def test_reason_error_classification(self) -> None:
        """Test reason error classification."""
        artifact = _recorded_artifact(_PureBackend())
        cases = [
            ("error_allowlisted", "error", "invalid_seed"),
            ("error_arbitrary", "error", "backend_reported_error"),
            ("error_context", "skipped", "replay_context_limit_exceeded"),
            ("cancelled", "error", "backend_reported_cancelled"),
            ("short_limit", "error", "invalid_generation"),
            ("not_a_generation", "error", "invalid_generation"),
        ]
        for mode, status, detail in cases:
            with self.subTest(mode=mode):
                audit = determinism.audit_determinism(artifact, _ReasonBackend(mode))
                self.assertEqual(audit["cells"][0]["status"], status)
                self.assertEqual(audit["cells"][0]["detail"], detail)
                self.assertEqual(audit["verdict"], "inconclusive")
        self.assertNotIn(
            "weird backend prose",
            json.dumps(determinism.audit_determinism(artifact, _ReasonBackend("error_arbitrary"))),
        )

    def test_budget_blocks_certificate(self) -> None:
        """Test budget blocks certificate."""
        artifact = _recorded_artifact(_PureBackend())
        backend = _PureBackend()
        with mock.patch.object(determinism, "MAX_AUDIT_CELLS", 1):
            audit = determinism.audit_determinism(artifact, backend)
        self.assertEqual(audit["n_attempted"], 1)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(audit["cells"][1]["detail"], "audit_cell_budget_exceeded")
        self.assertEqual(audit["verdict"], "inconclusive")
        self.assertIsNone(audit["certificate"])


class TestAuditHostile(unittest.TestCase):
    """Test audit hostile."""

    def test_f5_hostile_battery_never_raises(self) -> None:
        """Test f5 hostile battery never raises."""
        backend = _PureBackend()
        base = _recorded_artifact(_PureBackend())

        def variant(mutate) -> dict:
            """Variant."""
            # type: ignore[no-untyped-def]
            import copy

            data = copy.deepcopy(base)
            mutate(data)
            return data

        cases = [
            variant(lambda d: d["search"][0].update({"observations": "nope"})),
            variant(lambda d: d["search"][0].update({"observations": [42]})),
            variant(lambda d: d["search"][0]["observations"][0].update({"input_ids": "abc"})),
            variant(lambda d: d["search"][0]["observations"][0].update({"input_ids": [1, -2, 3]})),
            variant(
                lambda d: d["search"][0]["observations"][0].update({"input_ids": [1, True, 3]})
            ),
            variant(lambda d: d["search"][0]["observations"][0].update({"fixture": 9})),
            variant(lambda d: d["search"].append("not-a-dict")),
            variant(lambda d: d["search"][0].pop("summary")),
        ]
        for data in cases:
            audit = determinism.audit_determinism(data, backend)
            self.assertIn(audit["verdict"], ("deterministic", "diverged", "inconclusive", None))
            json.dumps(audit, ensure_ascii=True)  # must stay serializable

    def test_lone_surrogate_fixture_id_replays_and_certifies(self) -> None:
        """Test lone surrogate fixture id replays and certifies."""
        # Byte-exact IDs (A9): a lone-surrogate fixture id derives its
        # seed via surrogatepass and the certificate digest encodes it
        # via ensure_ascii escapes -- neither path may raise.
        seed = 3
        fixture = "\ud800fx"
        request_seed = request_seed_for(seed, fixture)
        artifact = _artifact(
            [_entry([_obs(fixture=fixture, output_ids=_pure_output((1, 2, 3), request_seed, 4))])],
            seed=seed,
        )
        audit = determinism.audit_determinism(artifact, _PureBackend())
        self.assertEqual(audit["verdict"], "deterministic")
        json.dumps(audit, ensure_ascii=True)


def _verify_args(**overrides) -> argparse.Namespace:
    """Verify args."""
    # type: ignore[no-untyped-def]
    values = {
        "results": None,
        "bundle": None,
        "expect_digest": None,
        "determinism": False,
        "model": None,
        "device": "cpu",
        "max_context": 2048,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _run_verify(args: argparse.Namespace) -> tuple[int, dict | None, str]:
    """Run verify."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = verify.run_verify(args, None)
    text = out.getvalue()
    return code, (json.loads(text) if text.strip() else None), err.getvalue()


def _wiring_artifact() -> dict:
    """A results dict plain verify accepts AND the audit can replay."""
    fixtures = (("f1", (1, 2, 3)), ("f2", (9, 8)))
    entries = []
    for name in ("cand-a", "cand-b"):
        observations = []
        for fixture, input_ids in fixtures:
            request_seed = int.from_bytes(
                hashlib.sha256(f"7\0{fixture}".encode()).digest()[:4], "big"
            )
            observations.append(
                _obs(
                    fixture=fixture,
                    input_ids=input_ids,
                    output_ids=_pure_output(input_ids, request_seed, 4),
                )
            )
        entries.append(_entry(observations, name=name))
    return _artifact(entries)


class TestVerifyWiring(unittest.TestCase):
    """Test verify wiring."""

    @staticmethod
    def _write(tmp: str, data: dict) -> Path:
        """Write."""
        path = Path(tmp) / "results.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_plain_verify_output_has_no_determinism_section(self) -> None:
        """Test plain verify output has no determinism section."""
        with TemporaryDirectory() as tmp:
            path = self._write(tmp, _wiring_artifact())
            code, payload, _err = _run_verify(_verify_args(results=path))
        self.assertEqual(code, 0)
        self.assertNotIn("determinism", payload)
        self.assertNotIn("determinism_replay", [c["id"] for c in payload["checks"]])

    def test_flag_matrix_errors(self) -> None:
        """Test flag matrix errors."""
        with TemporaryDirectory() as tmp:
            path = self._write(tmp, _wiring_artifact())
            cases = [
                _verify_args(results=path, determinism=True),  # no --model
                _verify_args(results=path, model=Path(tmp)),  # no --determinism
                _verify_args(results=None, bundle=path, determinism=True, model=Path(tmp)),
                _verify_args(
                    results=path,
                    determinism=True,
                    model=Path(tmp),
                    expect_digest="a" * 64,
                ),
                _verify_args(determinism=True, model=Path(tmp)),  # no results
            ]
            for args in cases:
                with self.subTest(args=args):
                    code, payload, err = _run_verify(args)
                    self.assertEqual(code, 1)
                    self.assertIsNone(payload)
                    self.assertIn("verify:", err)

    def test_precheck_failure_never_constructs_a_backend(self) -> None:
        """Test precheck failure never constructs a backend."""
        stripped = _wiring_artifact()
        for entry in stripped["search"]:
            del entry["observations"]
        with TemporaryDirectory() as tmp:
            path = self._write(tmp, stripped)
            args = _verify_args(results=path, determinism=True, model=Path("/nonexistent-ckpt"))
            code, payload, _err = _run_verify(args)
        # Construction of a backend for /nonexistent-ckpt would exit 1;
        # the precheck path reports unavailable instead.
        self.assertEqual(code, 0)
        audit = payload["determinism"]
        self.assertFalse(audit["available"])
        self.assertEqual(audit["detail"], "observations_stripped")
        check = next(c for c in payload["checks"] if c["id"] == "determinism_replay")
        self.assertEqual(check["status"], "unavailable")

    def test_full_determinism_run_via_verify(self) -> None:
        """Test full determinism run via verify."""
        with TemporaryDirectory() as tmp:
            path = self._write(tmp, _wiring_artifact())
            args = _verify_args(results=path, determinism=True, model=Path(tmp))
            with mock.patch(
                "model_protocol_lab.backends.LocalHFBackend",
                lambda *a, **k: _PureBackend(),
            ):
                code, payload, _err = _run_verify(args)
        self.assertEqual(code, 0)
        audit = payload["determinism"]
        self.assertEqual(audit["verdict"], "deterministic")
        self.assertIsNotNone(audit["certificate"])
        self.assertEqual(audit["certificate"]["replay"], {"device": "cpu", "max_context": 2048})
        check = next(c for c in payload["checks"] if c["id"] == "determinism_replay")
        self.assertEqual(check["status"], "passed")

    def test_diverged_run_fails_verify_with_finding_framing(self) -> None:
        """Test diverged run fails verify with finding framing."""
        data = _wiring_artifact()
        data["search"][0]["observations"][0]["output_ids"][1] += 1
        with TemporaryDirectory() as tmp:
            path = self._write(tmp, data)
            args = _verify_args(results=path, determinism=True, model=Path(tmp))
            with mock.patch(
                "model_protocol_lab.backends.LocalHFBackend",
                lambda *a, **k: _PureBackend(),
            ):
                code, payload, _err = _run_verify(args)
        self.assertEqual(code, 2)
        self.assertEqual(payload["verdict"], "failed")
        check = next(c for c in payload["checks"] if c["id"] == "determinism_replay")
        self.assertEqual(check["status"], "failed")
        self.assertIn("backend-determinism finding", check["detail"])

    def test_error_cells_map_to_an_unavailable_check_not_a_failure(self) -> None:
        """Test error cells map to an unavailable check not a failure."""
        # A6 (verify doctrine): a backend that errors during
        # re-execution means the check could not be performed -- the
        # run still exits 0 when nothing else failed, and the JSON
        # determinism section carries the actual result.
        with TemporaryDirectory() as tmp:
            path = self._write(tmp, _wiring_artifact())
            args = _verify_args(results=path, determinism=True, model=Path(tmp))
            with mock.patch(
                "model_protocol_lab.backends.LocalHFBackend",
                lambda *a, **k: _RaisingBackend(),
            ):
                code, payload, _err = _run_verify(args)
        self.assertEqual(code, 0)
        self.assertEqual(payload["determinism"]["verdict"], "inconclusive")
        check = next(c for c in payload["checks"] if c["id"] == "determinism_replay")
        self.assertEqual(check["status"], "unavailable")
        self.assertIn("determinism not established", check["detail"])

    def test_incomplete_search_detail_names_the_gap(self) -> None:
        """Test incomplete search detail names the gap."""
        data = _wiring_artifact()
        data["decision"] = "cancelled"
        with TemporaryDirectory() as tmp:
            path = self._write(tmp, data)
            args = _verify_args(results=path, determinism=True, model=Path(tmp))
            with mock.patch(
                "model_protocol_lab.backends.LocalHFBackend",
                lambda *a, **k: _PureBackend(),
            ):
                code, payload, _err = _run_verify(args)
        self.assertEqual(code, 0)
        check = next(c for c in payload["checks"] if c["id"] == "determinism_replay")
        self.assertEqual(check["status"], "unavailable")
        self.assertIn("search incomplete", check["detail"])

    def test_backend_construction_failure_is_fixed_vocabulary(self) -> None:
        """Test backend construction failure is fixed vocabulary."""

        def explode(*a, **k):
            """Explode."""
            # type: ignore[no-untyped-def]
            raise RuntimeError("local_backend_initialization_failed")

        with TemporaryDirectory() as tmp:
            path = self._write(tmp, _wiring_artifact())
            args = _verify_args(results=path, determinism=True, model=Path(tmp))
            with mock.patch("model_protocol_lab.backends.LocalHFBackend", explode):
                code, payload, err = _run_verify(args)
        self.assertEqual(code, 1)
        self.assertIsNone(payload)
        self.assertIn("could not be initialized", err)
        self.assertNotIn("Traceback", err)


def _exec_variant(original: str, mutant: str, tag: str):  # type: ignore[no-untyped-def]
    """Exec determinism.py with one anchored mutation, package-relative.

    The mutant module is loaded under the real package namespace so its
    ``from .core import ...`` resolves exactly as the shipped module's
    does (the cycle-169/170 tempdir-gate pattern).
    """
    source = (REPO_ROOT / "model_protocol_lab" / "determinism.py").read_text(encoding="utf-8")
    assert source.count(original) == 1, f"gate anchor not unique: {original!r}"
    name = f"model_protocol_lab.determinism_mut_{tag}"
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "model_protocol_lab"
    sys.modules[name] = module
    try:
        exec(  # noqa: S102 -- the gate's whole point: run a mutated copy of our own module
            compile(source.replace(original, mutant), f"<mutant {tag}>", "exec"),
            module.__dict__,
        )
    finally:
        del sys.modules[name]
    return module


class TestMutationGates(unittest.TestCase):
    """Test mutation gates."""

    def test_gate_comparison_off_by_one_is_killed(self) -> None:
        """Test gate comparison off by one is killed."""
        mutant = _exec_variant(
            '"first_divergence": None if match else shared,',
            '"first_divergence": None if match else shared + 1,',
            "cmp",
        )
        killed = mutant.compare_token_ids((1, 2, 3), (1, 9, 3))["first_divergence"] != 1
        self.assertTrue(killed, "comparison mutant survived")
        self.assertEqual(determinism.compare_token_ids((1, 2, 3), (1, 9, 3))["first_divergence"], 1)

    def test_gate_fingerprint_gate_removed_is_killed(self) -> None:
        """Test gate fingerprint gate removed is killed."""
        mutant = _exec_variant(
            "if type(replay_fingerprint) is not str or replay_fingerprint != recorded_fingerprint:",
            "if False:",
            "fp",
        )
        artifact = _recorded_artifact(_PureBackend())
        backend = _PureBackend(fingerprint="otherfp")
        mutant.audit_determinism(artifact, backend)
        self.assertGreater(len(backend.calls), 0, "fingerprint-gate mutant survived")
        clean = _PureBackend(fingerprint="otherfp")
        determinism.audit_determinism(artifact, clean)
        self.assertEqual(len(clean.calls), 0)

    def test_gate_length_invariant_removed_is_killed(self) -> None:
        """Test gate length invariant removed is killed."""
        mutant = _exec_variant(
            "if len(output_ids) != cap:",
            "if False:",
            "len",
        )
        artifact = _recorded_artifact(_PureBackend())
        artifact["search"][0]["observations"][0]["output_ids"] = [1, 2, 3]
        audit = mutant.audit_determinism(artifact, _PureBackend())
        killed = audit["cells"][0]["status"] == "diverged"
        self.assertTrue(killed, "length-invariant mutant survived")
        clean = determinism.audit_determinism(artifact, _PureBackend())
        self.assertEqual(clean["cells"][0]["detail"], "recorded_length_inconsistent")


@unittest.skipUnless(HAS_HF, "hf extra not installed")
class TestRealBackendEndToEnd(unittest.TestCase):
    """Test real backend end to end."""

    def test_local_hf_backend_reproduces_its_own_run(self) -> None:
        """Test local hf backend reproduces its own run."""
        # Reality check with the shipped backend on a tiny checkpoint:
        # evaluate once, then audit with a FRESH backend instance over
        # the same checkpoint. CPU fp32 greedy must reproduce exactly.
        from model_protocol_lab.backends import LocalHFBackend
        from tests.test_adapter_integration import build_tiny_gpt2

        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "tiny"
            build_tiny_gpt2(model_dir)
            # The tiny checkpoint declares a 64-position window.
            backend = LocalHFBackend(model_dir, max_context=64)
            fmt = CATALOG["chatml"]
            cases = (
                Fixture(
                    id="real-one",
                    group="g",
                    expected="x",
                    messages=(Message("user", "Say x."),),
                ),
            )
            report = evaluate(
                backend,
                fmt,
                cases,
                boundary=(999,),
                max_new_tokens=6,
                seed=5,
                cancel=Event(),
            )
            self.assertEqual(report.status, "completed")
            entry = {
                "format": {"name": fmt.name},
                "summary": {
                    "aggregate_eligible": True,
                    "planned": 1,
                    "exact_and_boundary_passes": 0,
                    "status": "completed",
                },
                "observations": [asdict(o) for o in report.observations],
            }
            artifact = _artifact(
                [entry],
                seed=5,
                cap=6,
                fingerprint=backend.fingerprint,
                decision="no_candidate",
            )
            artifact = json.loads(json.dumps(artifact))
            replay_backend = LocalHFBackend(model_dir, max_context=64)
            audit = determinism.audit_determinism(artifact, replay_backend)
        self.assertEqual(audit["verdict"], "deterministic")
        self.assertIsNotNone(audit["certificate"])


if __name__ == "__main__":
    unittest.main()
