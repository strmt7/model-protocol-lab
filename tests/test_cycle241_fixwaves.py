"""Cycle 241: combined bug-hunt waves 4+5+6 fix regressions.

Every test pins a confirmed finding from the three parent-verified
bug-hunt waves (``~/workspace/agents_out/bughunt4``, ``bughunt5``,
``bughunt6``) as adjudicated by the wave-4 fix brief. One family:
the diff/report mirrors consumed a stored payload value before
type-checking it, so a malformed-but-loadable payload crashed the
differ/renderer with a raw traceback (or was admitted by a widened
grammar) instead of degrading to the frozen ``available``
vocabulary. Each hostile case below raised (or was admitted) on
the pre-fix code and degrades on the fixed code; each control pins
that honest payloads behave exactly as before. Grammar specs are
the modules' own documented grammars: the numform mirror's
``[0-9]`` discipline (``_numform_parse_value`` docstring), the
confidence producer's frozen line convention (a carried line is an
element of ``split("\\n")``, hence newline-free), the SHA-256
grammar's own text (exactly 64 lowercase hex characters, the
convention verify.py and the diffreports mirrors already apply via
``re.fullmatch``), and ``planted_answer``'s documented contract
(whitespace-padded shapes are out-of-grammar and return None).
"""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

from model_protocol_lab import diffreports, pack, reporting
from model_protocol_lab.confidence import protocol_fingerprint
from model_protocol_lab.distraction import planted_answer
from tests import test_cycle146_token_healing as _token_healing_factories
from tests import test_cycle166_template_robustness as _template_robustness_factories
from tests import test_cycle183_repetition as _repetition_factories
from tests.test_cycle156_wscanonical import _ws_payload
from tests.test_cycle184_distraction import EXPECTED_BY_MARKER
from tests.test_cycle184_distraction import RULES as _DISTRACTION_RULES
from tests.test_cycle184_distraction import _payload as _distraction_payload
from tests.test_cycle184_distraction import _run_stage as _distraction_stage
from tests.test_cycle184_distraction import _WordBackend as _DistractionBackend
from tests.test_cycle185_leading import _fab_payload, _vrecords
from tests.test_cycle186_numform import _base_payload as _numform_payload
from tests.test_cycle188_dateline import _base_payload as _dateline_payload
from tests.test_cycle189_multiturn import _payload as _multiturn_payload
from tests.test_cycle199_confidence import _stage_payload as _confidence_payload
from tests.test_cycle205_fieldorder import _payload as _field_order_payload

REPO_ROOT = Path(__file__).resolve().parent.parent


def _sample_results() -> dict:
    """Load the committed sample results payload."""
    path = REPO_ROOT / "examples" / "sample-report" / "results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _dg(text: str) -> str:
    """SHA-256 hex digest of ``text``."""
    return hashlib.sha256(text.encode()).hexdigest()


class NumformIsdigitGuardTests(unittest.TestCase):
    """Wave-4 F1: the numform mirror's value_id construction gate.

    Spec: the mirror's grammars are deliberately ``[0-9]`` (never
    ``\\d`` and never a Unicode-lenient digit test), because the
    producer accepts ASCII ids only. A No-category digit string
    (superscripts, circled digits) satisfies ``str.isdigit()`` but
    makes ``int()`` raise; an Nd string (Arabic-Indic, fullwidth)
    passes both and admits an id the producer never emits.
    """

    @staticmethod
    def _record(value_id: str) -> dict:
        """A record whose value_id check runs before all others."""
        return {
            "task": "copy",
            "value_id": value_id,
            "form": "plain",
            "band": 0,
            "stratum": "random",
            "operands": [123],
            "expected_value": 123,
            "expected_answer": "123",
            "form_identical": True,
            "expected_answer_tokens": 1,
            "answer_value": None,
            "answer_text": None,
            "cell": None,
            "detail": "arm_evaluation_failed",
        }

    def test_ascii_id_constructs(self):
        """An honest ASCII id still constructs its tuple."""
        built = diffreports._numform_record_construction(self._record("nf-copy-07"))
        self.assertIsInstance(built, tuple)

    def test_no_category_digits_rejected_not_raised(self):
        """Superscript/circled ids degrade to None, never ValueError."""
        for value_id in ("nf-copy-²³", "nf-copy-①②"):
            with self.subTest(value_id=value_id):
                self.assertIsNone(diffreports._numform_record_construction(self._record(value_id)))

    def test_nd_digits_rejected_at_construction(self):
        """Nd ids the producer never emits are rejected at the gate."""
        for value_id in ("nf-copy-١٢", "nf-copy-１２"):
            with self.subTest(value_id=value_id):
                self.assertIsNone(diffreports._numform_record_construction(self._record(value_id)))

    def test_diff_degrades_on_hostile_value_id(self):
        """A genuine payload carrying a hostile id diffs as one-sided."""
        genuine = _numform_payload()
        mutated = copy.deepcopy(genuine)
        name = next(iter(mutated["candidates"]))
        mutated["candidates"][name]["records"][0]["value_id"] = "nf-copy-²³"
        self.assertEqual(diffreports.diff_numform(mutated, genuine)["available"], "new")
        self.assertEqual(diffreports.diff_numform(genuine, mutated)["available"], "old")


class ReportingDistractionSortTests(unittest.TestCase):
    """Wave-4 F2: the distraction section's curve-level sort key.

    Spec: honest payloads carry ASCII dose levels; the sort key's
    fallback (0) exists for non-digit keys, so a Unicode-digit key
    must take the fallback, not raise inside ``int()``.
    """

    @staticmethod
    def _payload(level: str) -> dict:
        """A minimal distraction section with one curve level."""
        return {
            "distraction": {
                "available": True,
                "candidates": {
                    "cand": {
                        "status": "available",
                        "curves": {
                            "neutral_before": {level: {"n_pass": 1, "n_paired": 2, "rate": 0.5}}
                        },
                        "first_flip": {},
                        "capture": {},
                    }
                },
            }
        }

    def test_ascii_level_renders(self):
        """The ASCII control renders its curve cell."""
        out = reporting._distraction_section(self._payload("3"))
        self.assertIn("tok:", out)

    def test_unicode_digit_levels_render(self):
        """Superscript/circled level keys render instead of raising."""
        for level in ("²", "①"):
            with self.subTest(level=level):
                out = reporting._distraction_section(self._payload(level))
                self.assertIn("tok:", out)

    def test_mixed_levels_render(self):
        """A curve mixing ASCII and Unicode-digit keys sorts safely."""
        payload = self._payload("3")
        curves = payload["distraction"]["candidates"]["cand"]["curves"]
        curves["neutral_before"]["10"] = {"n_pass": 2, "n_paired": 2, "rate": 1.0}
        curves["neutral_before"]["²"] = {"n_pass": 0, "n_paired": 2, "rate": 0.0}
        curves["neutral_before"]["x"] = {"n_pass": 1, "n_paired": 2, "rate": 0.5}
        out = reporting._distraction_section(payload)
        self.assertIn("tok:", out)


class ConfidenceGrammarFullmatchTests(unittest.TestCase):
    """Wave-4 F3: the confidence mirror grammar uses fullmatch.

    Spec: the producer's frozen line convention -- a carried ok
    line is an element of ``split("\\n")`` and therefore
    newline-free; the grammar is the bijection with the confidence
    scalar. A stored ``final_line`` of ``"Confidence: 42\\n"`` is a
    state no answer can produce and must not validate.
    """

    _EXPECTED = _dg("the expected span")

    @classmethod
    def _ok_cell(cls, line: str, confidence: int = 42) -> dict:
        """A self-consistent ok cell carrying ``line``."""
        return {
            "answer_digest": _dg("full answer text"),
            "boundary_found": True,
            "code": "ok",
            "confidence": confidence,
            "final_line": line,
            "final_line_length": len(line),
            "input_tokens": 10,
            "span_digest": cls._EXPECTED,
            "verdict": True,
        }

    @classmethod
    def _malformed_cell(cls, line: str) -> dict:
        """A self-consistent malformed_line cell carrying ``line``."""
        return {
            "answer_digest": cls._EXPECTED,
            "boundary_found": True,
            "code": "malformed_line",
            "confidence": None,
            "final_line": line,
            "final_line_length": len(line),
            "input_tokens": 10,
            "span_digest": cls._EXPECTED,
            "verdict": True,
        }

    def test_clean_line_validates(self):
        """The honest grammar line still validates."""
        self.assertTrue(
            diffreports._confidence_cell_ok(self._ok_cell("Confidence: 42"), self._EXPECTED)
        )

    def test_trailing_newline_line_rejected(self):
        """The forged newline-carrying ok cell no longer validates."""
        self.assertFalse(
            diffreports._confidence_cell_ok(self._ok_cell("Confidence: 42\n"), self._EXPECTED)
        )

    def test_trailing_space_and_cr_rejected(self):
        """Trailing-space and carriage-return controls stay rejected."""
        for line in ("Confidence: 42 ", "Confidence: 42\r", "Confidence: 42\n\n"):
            with self.subTest(line=line):
                self.assertFalse(
                    diffreports._confidence_cell_ok(self._ok_cell(line), self._EXPECTED)
                )

    def test_malformed_exclusion_still_fires(self):
        """A fully carried true grammar line is not malformed_line."""
        self.assertFalse(
            diffreports._confidence_cell_ok(self._malformed_cell("Confidence: 42"), self._EXPECTED)
        )

    def test_malformed_genuine_line_admitted(self):
        """A genuinely malformed carried line still classifies."""
        self.assertTrue(
            diffreports._confidence_cell_ok(
                self._malformed_cell("Confidence: forty-two"), self._EXPECTED
            )
        )

    def test_malformed_newline_line_classifiable(self):
        """A newline-carrying line is malformed, not grammar-shaped.

        Pins the fix brief's branch note: under fullmatch the
        malformed branch's exclusion ("a fully carried *grammar*
        line is not malformed") no longer treats ``"Confidence:
        42\\n"`` as grammar-shaped, so the cell classifies as
        malformed_line -- the spec-correct reading, since the line
        attests no scalar and verdicts stay digest-bound.
        """
        self.assertTrue(
            diffreports._confidence_cell_ok(
                self._malformed_cell("Confidence: 42\n"), self._EXPECTED
            )
        )


class PackManifestGrammarTests(unittest.TestCase):
    """Wave-4 fix brief 2.2 / wave-5 F4: pack's SHA-256 screen.

    Spec: a digest is exactly 64 lowercase hex characters. Each
    assertion kills its own call-site mutant -- restoring
    ``.match()`` at the file-entry site or the catalog site
    individually must fail the corresponding test.
    """

    @staticmethod
    def _manifest(file_sha: str, catalog_sha: str) -> dict:
        """A minimal format-1 manifest with the two digest fields."""
        return {
            "format": pack.FORMAT_ID,
            "files": [
                {
                    "path": pack.RESULTS_NAME,
                    "sha256": file_sha,
                    "size": 3,
                    "role": "results",
                }
            ],
            "context": {
                "package_version": "0.2.0",
                "catalog_sha256": catalog_sha,
                "backend_fingerprint": "fp",
            },
        }

    def test_clean_manifest_valid(self):
        """The honest 64-hex manifest validates (returns None)."""
        self.assertIsNone(pack._valid_manifest(self._manifest("a" * 64, "b" * 64)))

    def test_file_sha_trailing_newline_rejected(self):
        """A newline-carrying file sha names manifest_sha256_invalid."""
        self.assertEqual(
            pack._valid_manifest(self._manifest("a" * 64 + "\n", "b" * 64)),
            "manifest_sha256_invalid",
        )

    def test_file_sha_trailing_space_rejected(self):
        """A space-carrying file sha names manifest_sha256_invalid."""
        self.assertEqual(
            pack._valid_manifest(self._manifest("a" * 64 + " ", "b" * 64)),
            "manifest_sha256_invalid",
        )

    def test_catalog_sha_trailing_newline_rejected(self):
        """A newline-carrying catalog sha names manifest_context_invalid."""
        self.assertEqual(
            pack._valid_manifest(self._manifest("a" * 64, "b" * 64 + "\n")),
            "manifest_context_invalid",
        )

    def test_catalog_sha_trailing_space_rejected(self):
        """A space-carrying catalog sha names manifest_context_invalid."""
        self.assertEqual(
            pack._valid_manifest(self._manifest("a" * 64, "b" * 64 + " ")),
            "manifest_context_invalid",
        )

    def test_short_sha_still_rejected(self):
        """The 63-hex control is rejected as before."""
        self.assertEqual(
            pack._valid_manifest(self._manifest("a" * 63, "b" * 64)),
            "manifest_sha256_invalid",
        )


class WsCanonicalFragileGateTests(unittest.TestCase):
    """Wave-5 F1: the ws_canonical validator gates fragile_candidates.

    Spec: the validator's own contract -- compared fields are
    type-checked "so the renderers never hit a TypeError on a
    hostile payload... A malformed side degrades to asymmetric
    instead of raising." ``fragile_candidates`` reaches ``set()``
    in the differ, so its entries must be strings, exactly as the
    sibling ``mutations`` list already requires.
    """

    def test_genuine_payload_valid(self):
        """The factory payload validates and self-diffs as both."""
        payload = _ws_payload(robust=False)
        self.assertTrue(diffreports._valid_ws_canonical(payload))
        self.assertEqual(
            diffreports.diff_ws_canonical(payload, copy.deepcopy(payload))["available"],
            "both",
        )

    def test_unhashable_entries_invalidate(self):
        """List/dict fragile entries invalidate the side, never raise."""
        genuine = _ws_payload(robust=False)
        for leaf in ([], {}, ["nested"], {"a": 1}):
            with self.subTest(leaf=leaf):
                mutated = copy.deepcopy(genuine)
                mutated["fragile_candidates"][0] = leaf
                self.assertFalse(diffreports._valid_ws_canonical(mutated))
                self.assertEqual(
                    diffreports.diff_ws_canonical(mutated, genuine)["available"], "new"
                )
                self.assertEqual(
                    diffreports.diff_ws_canonical(genuine, mutated)["available"], "old"
                )

    def test_non_list_field_invalidates(self):
        """A bare-string fragile_candidates field is malformed too."""
        mutated = _ws_payload(robust=False)
        mutated["fragile_candidates"] = "alpha"
        self.assertFalse(diffreports._valid_ws_canonical(mutated))

    def test_absent_field_stays_valid(self):
        """A payload without the field validates, as the differ tolerates."""
        payload = _ws_payload(robust=False)
        del payload["fragile_candidates"]
        self.assertTrue(diffreports._valid_ws_canonical(payload))

    def test_public_diff_reports_degrades(self):
        """The public entry point reports the section one-sided."""
        genuine = _ws_payload(robust=False)
        mutated = copy.deepcopy(genuine)
        mutated["fragile_candidates"][0] = []
        old = _sample_results()
        old["ws_canonical"] = genuine
        new = _sample_results()
        new["ws_canonical"] = mutated
        out = diffreports.diff_reports(old, new)
        self.assertEqual(out["ws_canonical"]["available"], "old")


class FrontierElementGateTests(unittest.TestCase):
    """Wave-5 F2: the frontier renderer gates span/step elements.

    Spec: the sibling helper for the same data, ``safe_budget_floor``,
    re-validates every span element's type and documents "Never
    raises"; the section's own attribution comment says rendering
    never raises. Span and step elements are budgets (ints) in
    every honest payload.
    """

    def test_baseline_renders(self):
        """The committed sample renders its frontier section."""
        out = reporting.render_html(_sample_results())
        self.assertIn('id="token-budget-frontier"', out)

    def test_span_lo_hostile_values_render(self):
        """None/string/list span lows are skipped, never compared."""
        for hostile in (None, "²³", []):
            with self.subTest(hostile=hostile):
                payload = _sample_results()
                payload["budget_frontier"]["flip_ranges"][0][0] = hostile
                out = reporting.render_html(payload)
                self.assertIsInstance(out, str)

    def test_span_hi_hostile_value_renders(self):
        """A non-int span high is skipped, never rendered as a budget."""
        payload = _sample_results()
        payload["budget_frontier"]["flip_ranges"][0][1] = "x"
        self.assertIsInstance(reporting.render_html(payload), str)

    def test_step_budget_string_renders(self):
        """A string decision-step budget is skipped, never compared."""
        payload = _sample_results()
        payload["budget_frontier"]["decision_steps"][0][0] = "x"
        self.assertIsInstance(reporting.render_html(payload), str)

    def test_attribution_span_unhashable_renders(self):
        """An unhashable attribution span element is never a dict key."""
        for span in ([[], 5], [5, []]):
            with self.subTest(span=span):
                payload = _sample_results()
                payload["budget_frontier"]["flip_attribution"] = [{"span": span, "crossings": []}]
                self.assertIsInstance(reporting.render_html(payload), str)


class PortfolioMemberGateTests(unittest.TestCase):
    """Wave-5 F3: the portfolio renderer gates member names.

    Spec: every neighbouring value in the section is defensively
    coerced and rendering never raises; a stored member name is
    used as a ``per_member`` dict key, so a non-string member must
    skip the lookup instead of raising ``TypeError: unhashable``.
    """

    def test_baseline_renders(self):
        """The committed sample renders its portfolio block."""
        out = reporting.render_html(_sample_results())
        self.assertIn("Greedy covering portfolio", out)

    def test_unhashable_members_render(self):
        """List/dict member names render instead of raising."""
        for hostile in ([], {}, ["x"]):
            with self.subTest(hostile=hostile):
                payload = _sample_results()
                payload["portfolio"]["portfolio"]["members"][0] = hostile
                self.assertIsInstance(reporting.render_html(payload), str)


class TokenHealingFalsificationGateTests(unittest.TestCase):
    """Wave-6 F1: the token_healing validator gates falsification.

    Spec: the validator's contract -- compared fields are
    type-checked "so the renderers never hit a TypeError on a
    hostile payload". The differ reads ``falsification`` via
    ``.get``, so a present value must be a dict (the producer
    always emits one); falsy non-dicts are malformed shapes of the
    same field and degrade the same way.
    """

    def test_genuine_payload_valid(self):
        """The factory payload validates and self-diffs as both."""
        payload = _token_healing_factories.DiffTokenHealingTests._payload()
        self.assertTrue(diffreports._valid_token_healing(payload))
        self.assertEqual(
            diffreports.diff_token_healing(payload, copy.deepcopy(payload))["available"],
            "both",
        )

    def test_truthy_non_dict_falsification_invalidates(self):
        """Truthy non-dict falsification degrades instead of raising."""
        genuine = _token_healing_factories.DiffTokenHealingTests._payload()
        for hostile in ("x", [[]], ["a"], 1.5, 7):
            with self.subTest(hostile=hostile):
                mutated = copy.deepcopy(genuine)
                mutated["falsification"] = hostile
                self.assertFalse(diffreports._valid_token_healing(mutated))
                self.assertEqual(
                    diffreports.diff_token_healing(mutated, genuine)["available"], "new"
                )

    def test_absent_and_empty_dict_stay_valid(self):
        """Absent and empty-dict falsification keep the old behavior."""
        genuine = _token_healing_factories.DiffTokenHealingTests._payload()
        for shape in (None, {}):
            with self.subTest(shape=shape):
                mutated = copy.deepcopy(genuine)
                mutated["falsification"] = shape
                self.assertTrue(diffreports._valid_token_healing(mutated))

    def test_public_diff_reports_degrades(self):
        """The public entry point reports the section one-sided."""
        genuine = _token_healing_factories.DiffTokenHealingTests._payload()
        mutated = copy.deepcopy(genuine)
        mutated["falsification"] = "x"
        old = _sample_results()
        old["token_healing"] = mutated
        new = _sample_results()
        new["token_healing"] = genuine
        out = diffreports.diff_reports(old, new)
        self.assertEqual(out["token_healing"]["available"], "new")


class TemplateRobustnessOperatorsGateTests(unittest.TestCase):
    """Wave-6 F2: the template_robustness validator gates operators.

    Spec: the validator's contract -- candidate fields are checked
    "so the renderers never hit a TypeError on a hostile payload".
    The differ's candidate view reads the ``operators`` map and
    each operator entry via ``.get``, so both must be dicts when
    present (the producer emits a dict of dicts).
    """

    def test_genuine_payload_valid(self):
        """The factory payload validates and self-diffs as both."""
        payload = _template_robustness_factories.DiffTests._payload()
        self.assertTrue(diffreports._valid_template_robustness(payload))
        self.assertEqual(
            diffreports.diff_template_robustness(payload, copy.deepcopy(payload))["available"],
            "both",
        )

    def test_non_dict_operators_invalidates(self):
        """A truthy non-dict operators map degrades, never raises."""
        genuine = _template_robustness_factories.DiffTests._payload()
        for hostile in ("x", [[]], ["a"], 7, 1.5):
            with self.subTest(hostile=hostile):
                mutated = copy.deepcopy(genuine)
                mutated["candidates"]["tmpl-a"]["operators"] = hostile
                self.assertFalse(diffreports._valid_template_robustness(mutated))
                self.assertEqual(
                    diffreports.diff_template_robustness(mutated, genuine)["available"],
                    "new",
                )

    def test_non_dict_operator_entry_invalidates(self):
        """A truthy non-dict operator entry degrades, never raises."""
        genuine = _template_robustness_factories.DiffTests._payload()
        for hostile in ("x", [[]], 7):
            with self.subTest(hostile=hostile):
                mutated = copy.deepcopy(genuine)
                mutated["candidates"]["tmpl-a"]["operators"]["case_flip"] = hostile
                self.assertFalse(diffreports._valid_template_robustness(mutated))
                self.assertEqual(
                    diffreports.diff_template_robustness(mutated, genuine)["available"],
                    "new",
                )

    def test_absent_operators_stays_valid(self):
        """A report without an operators map keeps the old behavior."""
        mutated = _template_robustness_factories.DiffTests._payload()
        mutated["candidates"]["tmpl-a"]["operators"] = None
        self.assertTrue(diffreports._valid_template_robustness(mutated))

    def test_missing_operator_entries_stay_valid(self):
        """A partial operators map keeps the old behavior."""
        mutated = _template_robustness_factories.DiffTests._payload()
        mutated["candidates"]["tmpl-a"]["operators"] = {"case_flip": {"preservation_rate": 1.0}}
        self.assertTrue(diffreports._valid_template_robustness(mutated))


def _set_path(obj: dict, path: tuple, value) -> None:
    """Set ``value`` at the nested ``path`` of ``obj``."""
    cur = obj
    for step in path[:-1]:
        cur = cur[step]
    cur[path[-1]] = value


class UnhashableOperandTests(unittest.TestCase):
    """Wave-6 F3: operands are type-gated before hashing operations.

    Spec: the mirrors' pervasive discipline -- a stored value is
    type-checked before it is consumed. A list or dict operand fed
    to a frozenset/set membership test, a dict membership test, or
    a dict ``.get`` key raises ``TypeError: unhashable type``; the
    operand must instead degrade exactly like a wrong string
    (the section recompute returns None, the side is unavailable).
    Every case below asserts the hostile verdict equals the
    wrong-string control verdict for the same leaf.
    """

    def _assert_site_degrades(self, genuine, differ, path) -> None:
        """Hostile []/{} and a wrong string degrade identically."""
        for hostile in ([], {}):
            with self.subTest(path=path, hostile=hostile):
                mutated = copy.deepcopy(genuine)
                _set_path(mutated, path, hostile)
                self.assertEqual(differ(mutated, genuine)["available"], "new")
        control = copy.deepcopy(genuine)
        _set_path(control, path, "bogus-value")
        self.assertEqual(differ(control, genuine)["available"], "new")

    def test_repetition_arm_details(self):
        """Repetition arm detail operand is gated before membership."""
        genuine = _repetition_factories.TestReporting._payload()
        self.assertEqual(
            diffreports.diff_repetition(genuine, copy.deepcopy(genuine))["available"],
            "both",
        )
        self._assert_site_degrades(
            genuine,
            diffreports.diff_repetition,
            ("candidates", "cand-a", "records", 0, "arm_details", "x2"),
        )

    def test_leading_cell_details(self):
        """Leading cell detail operand is gated before membership."""
        genuine = _fab_payload(_vrecords())
        self.assertEqual(
            diffreports.diff_leading(genuine, copy.deepcopy(genuine))["available"], "both"
        )
        self._assert_site_degrades(
            genuine,
            diffreports.diff_leading,
            ("candidates", "cand-a", "records", 0, "cell_details", "wrong_first_person"),
        )

    def test_distraction_cell_details(self):
        """Distraction cell detail operand is gated before membership."""
        backend = _DistractionBackend(_DISTRACTION_RULES, EXPECTED_BY_MARKER)
        genuine = _distraction_payload(backend, _distraction_stage(backend))
        self.assertEqual(
            diffreports.diff_distraction(genuine, copy.deepcopy(genuine))["available"],
            "both",
        )
        self._assert_site_degrades(
            genuine,
            diffreports.diff_distraction,
            ("candidates", "dst-a", "records", 0, "cell_details", "neutral_64_after"),
        )

    def test_dateline_cell_details(self):
        """Dateline cell detail operand is gated before membership."""
        genuine = _dateline_payload()
        self.assertEqual(
            diffreports.diff_dateline(genuine, copy.deepcopy(genuine))["available"], "both"
        )
        self._assert_site_degrades(
            genuine,
            diffreports.diff_dateline,
            ("candidates", "dt-a", "records", 0, "cell_details", "plus_year_long"),
        )

    def test_field_order_detail(self):
        """Field-order record detail never reaches a hashing test.

        The section's detail constant now has the tuple shape of
        its density sibling (identical vocabulary), so membership
        scans by equality and an unhashable operand degrades like
        a wrong string.
        """
        genuine = _field_order_payload()
        self.assertEqual(
            diffreports.diff_field_order(genuine, copy.deepcopy(genuine))["available"],
            "both",
        )
        self._assert_site_degrades(
            genuine,
            diffreports.diff_field_order,
            ("candidates", "fo-a", "records", 0, "detail"),
        )

    def test_confidence_elicited_code(self):
        """Confidence elicited code operand is gated before membership."""
        genuine = _confidence_payload()
        self.assertEqual(
            diffreports.diff_confidence(genuine, copy.deepcopy(genuine))["available"],
            "both",
        )
        self._assert_site_degrades(
            genuine,
            diffreports.diff_confidence,
            ("records", 0, "elicited", "code"),
        )

    def test_confidence_protocol_screens(self):
        """Confidence screen-class operand is gated before membership."""
        genuine = _confidence_payload()
        for hostile in ([], {}):
            with self.subTest(hostile=hostile):
                mutated = copy.deepcopy(genuine)
                mutated["protocol"]["screens"] = {"probe-fixture": hostile}
                self.assertEqual(diffreports.diff_confidence(mutated, genuine)["available"], "new")
        control = copy.deepcopy(genuine)
        control["protocol"]["screens"] = {"probe-fixture": "bogus-value"}
        self.assertEqual(diffreports.diff_confidence(control, genuine)["available"], "new")

    def test_confidence_protocol_screens_genuine_entry(self):
        """A genuine screen-class entry stays valid (fingerprint redone)."""
        genuine = _confidence_payload()
        genuine["protocol"]["screens"] = {"probe-fixture": "expected_blank"}
        genuine["protocol"]["fingerprint"] = protocol_fingerprint(genuine["protocol"])
        self.assertEqual(
            diffreports.diff_confidence(genuine, copy.deepcopy(genuine))["available"],
            "both",
        )

    def test_confidence_control_case(self):
        """Confidence control case operand is gated before dict lookup."""
        genuine = _confidence_payload()
        self._assert_site_degrades(
            genuine,
            diffreports.diff_confidence,
            ("control_records", 0, "case"),
        )

    def test_multiturn_fixture(self):
        """Multiturn fixture operand is gated before the dict lookup."""
        genuine = _multiturn_payload()
        self.assertEqual(
            diffreports.diff_multiturn(genuine, copy.deepcopy(genuine))["available"],
            "both",
        )
        self._assert_site_degrades(
            genuine,
            diffreports.diff_multiturn,
            ("candidates", "mt-b", "records", 0, "fixture"),
        )

    def test_public_diff_reports_degrades(self):
        """The public entry point reports the section one-sided."""
        genuine = _confidence_payload()
        mutated = copy.deepcopy(genuine)
        mutated["records"][0]["elicited"]["code"] = []
        old = _sample_results()
        old["confidence"] = mutated
        new = _sample_results()
        new["confidence"] = genuine
        out = diffreports.diff_reports(old, new)
        self.assertEqual(out["confidence"]["available"], "new")


class DatelineAnchorValidationTests(unittest.TestCase):
    """Wave-6 F4: the dateline anchor branch validates before subscripting.

    Spec: ``_leading_cell_consistent`` is the check that
    type-verifies cells; the treated control-key branch compares a
    cell against its anchor before that check runs, so the branch
    itself must require dict cells whose ``input_tokens`` passes
    the same ``_leading_int_ok`` discipline before subscripting.
    A malformed cell or anchor makes the recompute return None.
    """

    def test_hostile_nodate_cells_degrade(self):
        """Non-dict and key-missing nodate cells degrade, never raise."""
        genuine = _dateline_payload()
        for hostile in (0, "x", [[]], {}, {"verdict": True}):
            with self.subTest(hostile=hostile):
                mutated = copy.deepcopy(genuine)
                mutated["candidates"]["dt-a"]["records"][0]["cells"]["nodate_long"] = hostile
                self.assertEqual(diffreports.diff_dateline(mutated, genuine)["available"], "new")

    def test_non_dict_anchor_degrades(self):
        """A non-dict anchor cell degrades instead of subscripting."""
        genuine = _dateline_payload()
        mutated = copy.deepcopy(genuine)
        mutated["candidates"]["dt-a"]["records"][0]["cells"]["anchor_long"] = 0
        self.assertEqual(diffreports.diff_dateline(mutated, genuine)["available"], "new")


class PlantShapeFullmatchTests(unittest.TestCase):
    """Named decision: ``planted_answer`` uses fullmatch semantics.

    Spec: the function's own documented contract -- anything
    outside the frozen shape, whitespace-padded included, returns
    None. The pre-fix ``.match()`` admitted exactly one trailing
    newline (``planted_answer("42\\n") == "43"``), contradicting
    the contract. The conversion is behavior-identical on every
    shipped fixture (the cycle's census found no shipped
    ``expected`` ending in a newline), so no case digest moves.
    """

    def test_clean_shapes_still_plant(self):
        """The docstring's own examples plant exactly as documented."""
        self.assertEqual(planted_answer("42"), "43")
        self.assertEqual(planted_answer("9,999"), "10,000")
        self.assertEqual(planted_answer("-0"), "-1")
        self.assertEqual(planted_answer("1e3"), "2e3")
        self.assertEqual(planted_answer(".5"), ".6")

    def test_trailing_newline_is_out_of_grammar(self):
        """A single trailing newline now returns None, per contract."""
        self.assertIsNone(planted_answer("42\n"))
        self.assertIsNone(planted_answer("9,999\n"))

    def test_other_whitespace_stays_out_of_grammar(self):
        """Trailing-space and CRLF controls were and stay None."""
        self.assertIsNone(planted_answer("42 "))
        self.assertIsNone(planted_answer("42\r\n"))


if __name__ == "__main__":
    unittest.main()
