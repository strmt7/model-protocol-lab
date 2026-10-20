"""Cycle 134 audit: prompt-boundary canonicality (research backlog R2).

For every stored rendered prompt x greedy completion, re-encoding the
concatenated text must reproduce ``prompt_ids + completion_ids``. A
mismatch is the Prompt Boundary Problem (PBP): the tokenizer re-segments
the text at the prompt/completion seam, so the ids the model saw are not
what the stored text would reproduce (ByteSampler, arXiv 2506.14123; the
companion measurement study, arXiv 2601.23223v1, shows 60-95.6% next-token
degradation at unexpected boundaries). No evaluation harness reports
boundary-canonicality as a template-quality metric; this cycle adds it as
a first-class, pure, post-hoc measurement: ``boundary_audit`` in
``results.json`` (overall + per-candidate + per-family PBP rates), an HTML
"Prompt-boundary canonicality" section, and a ``diff-reports`` comparison.

Design notes:
- The check needs the *rendered prompt string*, so ``core.Observation``
  now stores ``input_text`` (decoding ``input_ids`` is not a substitute:
  re-decoding is itself canonicalizing). Threaded through every
  ``evaluate()`` record path; defaults to "" on pre-render failures.
- ``check_boundary`` takes ``encode``/``decode`` callables, so the core is
  fully testable without any model or tokenizer. Backend encode/decode
  failures degrade a check to "unusable", never a crash.
- A zero rate renders as a "certificate" line in the HTML report; the
  section is honest about what it measures (text->ids reproducibility,
  not model quality).

Tests pin: the pure check (identity, planted seam-merge PBP, divergence
index, length mismatch, unusable paths), rate aggregation, the family
rollup, end-to-end ``input_text`` recording via ``evaluate()``, the HTML
section, the diff-reports comparison, and a committed 4-mutant kill gate.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from threading import Event

from model_protocol_lab.boundary_audit import (
    BoundaryCheck,
    audit_search_observations,
    boundary_problem_rate,
    check_boundary,
)
from model_protocol_lab.core import Fixture, Format, Generation, Message, evaluate
from model_protocol_lab.diffreports import diff_boundary_audit, diff_reports
from model_protocol_lab.reporting import render_html

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _identity_encode(text: str) -> tuple[int, ...]:
    return tuple(map(ord, text))


def _identity_decode(ids) -> str:
    return "".join(map(chr, ids))


def _seam_merging_encode(text: str) -> tuple[int, ...]:
    """Toy tokenizer that merges 'b'+'c' across the prompt/completion seam."""
    if text == "ab" + "cd":
        return (97, 200, 100)
    return tuple(map(ord, text))


class TestCheckBoundary(unittest.TestCase):
    def test_clean_identity(self) -> None:
        check = check_boundary("ab", (97, 98), (99, 100), _identity_encode, _identity_decode)
        self.assertEqual(check, BoundaryCheck(False, None, True))

    def test_empty_completion_clean(self) -> None:
        check = check_boundary("ab", (97, 98), (), _identity_encode, _identity_decode)
        self.assertFalse(check.has_problem)
        self.assertTrue(check.usable)

    def test_planted_seam_merge_detected_with_index(self) -> None:
        check = check_boundary("ab", (97, 98), (99, 100), _seam_merging_encode, _identity_decode)
        self.assertTrue(check.has_problem)
        self.assertTrue(check.usable)
        self.assertEqual(check.first_divergence_index, 1)

    def test_length_mismatch_diverges_at_shorter_end(self) -> None:
        def short_encode(text: str) -> tuple[int, ...]:
            return tuple(map(ord, text))[:2]

        check = check_boundary("ab", (97, 98), (99, 100), short_encode, _identity_decode)
        self.assertTrue(check.has_problem)
        self.assertEqual(check.first_divergence_index, 2)

    def test_non_string_prompt_unusable(self) -> None:
        check = check_boundary(None, (97,), (98,), _identity_encode, _identity_decode)  # type: ignore[arg-type]
        self.assertFalse(check.usable)
        self.assertFalse(check.has_problem)

    def test_encode_failure_unusable(self) -> None:
        def boom(text: str) -> tuple[int, ...]:
            raise RuntimeError("encode exploded")

        check = check_boundary("ab", (97, 98), (99, 100), boom, _identity_decode)
        self.assertFalse(check.usable)

    def test_decode_failure_unusable(self) -> None:
        def boom(ids) -> str:
            raise RuntimeError("decode exploded")

        check = check_boundary("ab", (97, 98), (99, 100), _identity_encode, boom)
        self.assertFalse(check.usable)

    def test_decode_non_string_unusable(self) -> None:
        check = check_boundary("ab", (97, 98), (99, 100), _identity_encode, lambda ids: 42)  # type: ignore[return-value]
        self.assertFalse(check.usable)

    def test_non_integer_ids_unusable(self) -> None:
        check = check_boundary("ab", ("x",), (99,), _identity_encode, _identity_decode)  # type: ignore[arg-type]
        self.assertFalse(check.usable)


class TestBoundaryProblemRate(unittest.TestCase):
    def test_all_clean(self) -> None:
        checks = [BoundaryCheck(False, None, True)] * 4
        self.assertEqual(boundary_problem_rate(checks), (0, 4, 0.0))

    def test_exact_fraction(self) -> None:
        checks = [BoundaryCheck(True, 1, True), BoundaryCheck(False, None, True)]
        self.assertEqual(boundary_problem_rate(checks), (1, 2, 0.5))

    def test_unusable_excluded(self) -> None:
        checks = [BoundaryCheck(False, None, False), BoundaryCheck(True, 0, True)]
        self.assertEqual(boundary_problem_rate(checks), (1, 1, 1.0))

    def test_no_usable_none_rate(self) -> None:
        self.assertEqual(boundary_problem_rate([]), (0, 0, None))
        self.assertEqual(boundary_problem_rate([BoundaryCheck(False, None, False)]), (0, 0, None))


class TestAuditSearchObservations(unittest.TestCase):
    def _entries(self):
        return [
            {
                "format": {"name": "chatml/x"},
                "observations": [
                    {"input_text": "ab", "input_ids": (97, 98), "output_ids": (99, 100)},
                    {"input_text": "ab", "input_ids": (97, 98), "output_ids": (99, 100)},
                ],
            },
            {
                "format": {"name": "chatml/y"},
                "observations": [
                    {"input_text": "", "input_ids": (), "output_ids": ()},  # unusable
                ],
            },
            {"format": {"name": "llama3"}, "observations": []},
            "not-a-dict",
        ]

    def test_rates_and_rollup(self) -> None:
        audit = audit_search_observations(self._entries(), _identity_encode, _identity_decode)
        self.assertTrue(audit["available"])
        self.assertEqual(audit["n_observations"], 3)
        self.assertEqual(audit["n_usable"], 2)
        self.assertEqual(audit["n_unusable"], 1)
        self.assertEqual(audit["overall_rate"], 0.0)
        self.assertEqual(audit["per_candidate"]["chatml/x"]["rate"], 0.0)
        self.assertEqual(audit["per_candidate"]["chatml/x"]["n_usable"], 2)
        # Family rollup merges chatml/x and chatml/y.
        self.assertEqual(audit["per_family"]["chatml"]["n_usable"], 2)
        self.assertIn("llama3", audit["per_family"])

    def test_planted_problem_shows_in_rate(self) -> None:
        entries = [
            {
                "format": {"name": "chatml"},
                "observations": [
                    {"input_text": "ab", "input_ids": (97, 98), "output_ids": (99, 100)},
                ],
            }
        ]
        audit = audit_search_observations(entries, _seam_merging_encode, _identity_decode)
        self.assertEqual(audit["overall_rate"], 1.0)
        self.assertEqual(audit["n_problem"], 1)
        # Mutation-gate pin: the per-candidate rate must reflect the
        # planted problem too (the `"rate": 0.0` mutant survived on the
        # overall rate alone).
        self.assertEqual(audit["per_candidate"]["chatml"]["rate"], 1.0)
        self.assertEqual(audit["per_candidate"]["chatml"]["n_problem"], 1)

    def test_empty_entries_unavailable(self) -> None:
        audit = audit_search_observations([], _identity_encode, _identity_decode)
        self.assertFalse(audit["available"])
        self.assertIsNone(audit["overall_rate"])

    def test_non_list_input_degrades(self) -> None:
        audit = audit_search_observations(None, _identity_encode, _identity_decode)  # type: ignore[arg-type]
        self.assertFalse(audit["available"])

    def test_encode_failure_marks_unusable_not_problem(self) -> None:
        def boom(text: str) -> tuple[int, ...]:
            raise RuntimeError("nope")

        entries = [
            {
                "format": {"name": "chatml"},
                "observations": [
                    {"input_text": "ab", "input_ids": (97, 98), "output_ids": (99, 100)},
                ],
            }
        ]
        audit = audit_search_observations(entries, boom, _identity_decode)
        self.assertFalse(audit["available"])
        self.assertEqual(audit["n_usable"], 0)

    def test_malformed_entries_skipped_fail_closed(self) -> None:
        # Defensive `continue` guards: entries with a non-str name, a
        # non-list observations field, or non-dict observations are
        # skipped without raising, and contribute no candidates.
        entries = [
            {"format": {"name": 123}, "observations": []},
            {"format": {"name": ""}, "observations": []},
            {"format": {"name": "chatml"}, "observations": "not-a-list"},
            {
                "format": {"name": "llama3"},
                "observations": [
                    "not-a-dict",
                    {"input_text": "ab", "input_ids": (97, 98), "output_ids": (99, 100)},
                ],
            },
        ]
        audit = audit_search_observations(entries, _identity_encode, _identity_decode)
        # Only llama3 survives, with the single well-formed observation.
        self.assertIn("llama3", audit["per_candidate"])
        self.assertNotIn(123, audit["per_candidate"])
        self.assertEqual(audit["per_candidate"]["llama3"]["n_observations"], 1)


class IdentityBackend:
    """Minimal canonical backend: ord/chr, perfectly boundary-canonical."""

    fingerprint = "identity-test-backend"

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(map(ord, text))

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(map(chr, token_ids))

    def generate(self, input_ids, *, max_new_tokens: int, seed: int, cancel: Event) -> Generation:
        return Generation((65,), "limit")


def _fixture() -> Fixture:
    return Fixture(
        id="f1",
        messages=(Message(role="user", content="say A"),),
        expected="A",
        group="g1",
    )


def _format() -> Format:
    return Format(
        name="chatml",
        bos="",
        user_prefix="<u>",
        assistant_prefix="<a>",
        turn_end="",
        repeat_bos=False,
        user_turn_end="",
        assistant_turn_end="",
        open_suffix="",
    )


class TestInputTextRecording(unittest.TestCase):
    def test_evaluate_records_rendered_prompt_text(self) -> None:
        fmt = _format()
        report = evaluate(IdentityBackend(), fmt, (_fixture(),), boundary=(65,), max_new_tokens=8)
        self.assertEqual(len(report.observations), 1)
        obs = report.observations[0]
        self.assertEqual(obs.input_text, fmt.render(_fixture().messages))
        self.assertEqual(obs.input_ids, tuple(map(ord, obs.input_text)))

    def test_observation_input_text_defaults_empty(self) -> None:
        from model_protocol_lab.core import Observation

        obs = Observation(
            candidate="c",
            fixture="f",
            input_ids=(1,),
            output_ids=(2,),
            answer="a",
            exact_match=True,
            boundary_found=True,
            boundary_text_match=None,
            raw_reason=None,
            error=None,
            status="completed",
        )
        self.assertEqual(obs.input_text, "")

    def test_audit_over_evaluate_output_is_canonical(self) -> None:
        fmt = _format()
        report = evaluate(IdentityBackend(), fmt, (_fixture(),), boundary=(65,), max_new_tokens=8)
        entries = [
            {
                "format": {"name": fmt.name},
                "observations": [asdict(o) for o in report.observations],
            }
        ]
        backend = IdentityBackend()
        audit = audit_search_observations(entries, backend.encode, backend.decode)
        self.assertTrue(audit["available"])
        self.assertEqual(audit["overall_rate"], 0.0)


class TestHtmlSection(unittest.TestCase):
    def _result(self, per_family):
        return {
            "decision": "select_best",
            "selected_candidate": "chatml",
            "boundary_audit": {
                "available": True,
                "n_observations": 4,
                "n_usable": 4,
                "n_unusable": 0,
                "n_problem": 0,
                "overall_rate": 0.0,
                "per_candidate": {},
                "per_family": per_family,
            },
        }

    def test_section_renders_certificate(self) -> None:
        html = render_html(self._result({"chatml": {"n_usable": 4, "n_problem": 0, "rate": 0.0}}))
        self.assertIn("Prompt-boundary canonicality", html)
        self.assertIn("certificate", html)
        self.assertIn("chatml", html)

    def test_section_renders_nonzero_rate(self) -> None:
        html = render_html(self._result({"llama3": {"n_usable": 4, "n_problem": 1, "rate": 0.25}}))
        self.assertIn("Prompt-boundary canonicality", html)
        self.assertIn("25.0%", html)
        self.assertIn("non-canonical", html)

    def test_section_absent_when_unavailable(self) -> None:
        html = render_html({"decision": "failed"})
        self.assertNotIn("Prompt-boundary canonicality", html)


class TestDiffBoundaryAudit(unittest.TestCase):
    def _audit(self, rate, families=None):
        chatml_entry = {"n_usable": 4, "n_problem": int(rate * 4), "rate": rate}
        return {
            "available": True,
            "n_observations": 4,
            "n_usable": 4,
            "n_unusable": 0,
            "n_problem": int(rate * 4),
            "overall_rate": rate,
            "per_candidate": {},
            "per_family": families if families is not None else {"chatml": chatml_entry},
        }

    def test_identical_audits(self) -> None:
        diff = diff_boundary_audit(self._audit(0.0), self._audit(0.0))
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])
        self.assertEqual(diff["family_drift"], [])

    def test_rate_drift_detected(self) -> None:
        diff = diff_boundary_audit(self._audit(0.0), self._audit(0.5))
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["overall_rate"]["changed"])
        self.assertEqual(len(diff["family_drift"]), 1)
        self.assertEqual(diff["family_drift"][0]["family"], "chatml")

    def test_one_sided(self) -> None:
        diff = diff_boundary_audit(self._audit(0.0), {"available": False})
        self.assertEqual(diff["available"], "old")
        self.assertFalse(diff["changed"])

    def test_neither(self) -> None:
        diff = diff_boundary_audit(None, None)
        self.assertEqual(diff["available"], "neither")

    def test_new_family_appears_as_drift(self) -> None:
        old = self._audit(0.0, {"chatml": {"n_usable": 2, "n_problem": 0, "rate": 0.0}})
        new = self._audit(
            0.25,
            {
                "chatml": {"n_usable": 2, "n_problem": 0, "rate": 0.0},
                "llama3": {"n_usable": 2, "n_problem": 1, "rate": 0.5},
            },
        )
        diff = diff_boundary_audit(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["family_drift"][0]["family"], "llama3")

    def test_family_with_no_usable_observations_is_not_drift(self) -> None:
        old = self._audit(0.0, {"chatml": {"n_usable": 2, "n_problem": 0, "rate": 0.0}})
        new = self._audit(
            0.0,
            {
                "chatml": {"n_usable": 2, "n_problem": 0, "rate": 0.0},
                "llama3": {"n_usable": 0, "n_problem": 0, "rate": None},
            },
        )
        diff = diff_boundary_audit(old, new)
        self.assertFalse(diff["changed"])
        self.assertEqual(diff["family_drift"], [])


class TestDiffReportsWiring(unittest.TestCase):
    def _report(self, rate):
        return {
            "decision": "select_best",
            "selected_candidate": "chatml",
            "search": [],
            "boundary_audit": {
                "available": True,
                "n_observations": 2,
                "n_usable": 2,
                "n_unusable": 0,
                "n_problem": int(rate * 2),
                "overall_rate": rate,
                "per_candidate": {},
                "per_family": {"chatml": {"n_usable": 2, "n_problem": int(rate * 2), "rate": rate}},
            },
        }

    def test_summary_flags_and_payload_key(self) -> None:
        diff = diff_reports(self._report(0.0), self._report(0.5))
        self.assertIn("boundary_audit", diff)
        self.assertTrue(diff["summary"]["boundary_audit_changed"])
        self.assertFalse(diff["summary"]["boundary_audit_asymmetric"])
        self.assertFalse(diff["summary"]["identical"])

    def test_identical_when_audits_match(self) -> None:
        diff = diff_reports(self._report(0.0), self._report(0.0))
        self.assertFalse(diff["summary"]["boundary_audit_changed"])
        self.assertTrue(diff["summary"]["identical"])


class TestMutationGate(unittest.TestCase):
    SOURCE = _REPO_ROOT / "model_protocol_lab" / "boundary_audit.py"

    # Mutant anchors, kept as constants so the (anchor, mutant) pairs stay
    # under the line-length limit without changing the matched text.
    _CLEAN_RETURN_8 = (
        "        return BoundaryCheck(has_problem=False, first_divergence_index=None, usable=True)"
    )
    _PBP_RETURN_12 = (
        "            return BoundaryCheck(has_problem=True, first_divergence_index=i, usable=True)"
    )
    _CLEAN_RETURN_12 = (
        "            return BoundaryCheck(has_problem=False, "
        "first_divergence_index=None, usable=True)"
    )

    MUTANTS = [
        # PBP never reported: joint equality forced true.
        (
            "    if joint == expected:\n" + _CLEAN_RETURN_8,
            "    if True:\n" + _CLEAN_RETURN_8,
        ),
        # Detection inverted: divergences reported clean.
        (
            _PBP_RETURN_12,
            _CLEAN_RETURN_12,
        ),
        # Rate pinned to zero regardless of problems.
        (
            '            "rate": rate,\n',
            '            "rate": 0.0,\n',
        ),
        # Unusable checks counted as problems.
        (
            "    usable = [c for c in checks if c.usable]",
            "    usable = list(checks)",
        ),
    ]

    def test_mutants_are_killed(self) -> None:
        source = self.SOURCE.read_text()
        killed = []
        survived = []
        for original, mutant in self.MUTANTS:
            self.assertIn(original, source, f"mutant anchor missing: {original!r}")
            self.SOURCE.write_text(source.replace(original, mutant, 1))
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        __file__,
                        "-q",
                        "-p",
                        "no:cacheprovider",
                        # Exclude the gate itself from the subprocess run:
                        # without this, the subprocess re-runs this very
                        # test, stacking one more mutant per nesting level
                        # (cycle-125's 22-site anchor fanned out to 184
                        # processes). -k is used instead of --deselect
                        # because __file__ is absolute under unittest
                        # discovery while pytest normalizes node IDs to
                        # relative paths, so an absolute --deselect never
                        # matches. The kill verdict must come from the
                        # audit tests alone.
                        "-k",
                        "not TestMutationGate",
                    ],
                    capture_output=True,
                    text=True,
                    cwd=_REPO_ROOT,
                )
            finally:
                self.SOURCE.write_text(source)
                # Invalidate the bytecode cache: a same-size mutant can
                # leave a .pyc whose (mtime, size) still validates
                # against the reverted source, poisoning later imports
                # with mutant bytecode (see the cycle-135 gate fix).
                for _pyc in self.SOURCE.parent.glob("__pycache__/boundary_audit.*.pyc"):
                    _pyc.unlink(missing_ok=True)
            if proc.returncode != 0:
                killed.append(mutant)
            else:
                survived.append(mutant)
        self.assertEqual(survived, [], f"surviving mutants: {survived}")
        self.assertEqual(len(killed), len(self.MUTANTS))


if __name__ == "__main__":
    unittest.main()
