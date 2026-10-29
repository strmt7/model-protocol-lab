"""Cycle 43: second canary evidence-leak sweep of every public output path.

Cycle 22 found and fixed two REAL evidence leaks (raw generated text and
token-ID subsequences reaching public JSON/HTML with include_evidence=False).
Cycle 24 swept the other surfaces and came back clean. Since then new code
has landed (debug REPL transcript/.save, catalog changes, diffbackends
docstrings/divergence section, scoring.py, experiments.py logprob/stop-matrix
additions, reporting.py sections, backends.py, fixtures.py audit sidecar,
datasets.py, diffreports.py fixture-set drift, cli.py, core.py), so this
file re-sweeps everything with a fresh canary: ``CANARY43-LEAKSWEEP-QZ7``.

Methodology: the canary is planted as fixture message content, expected
answers, generated text, and token-ID arrays inside a synthetic result
dict; every public output path is then exercised -- ``save_report`` JSON +
HTML with include_evidence=False AND True, ``diff-reports``, ``diff-backends``
(function and CLI), ``mpl debug`` REPL (incl. ``.save``, ``.read``/``--script``
and the transcript), ``mpl doctor`` (incl. CLI and the portability matrix),
CLI ``inspect`` / ``fixtures build``, ``stop_matrix`` / ``budget_frontier`` /
logprob interpretation, and the committed ``examples/sample-report/``.
Each assertion checks the canary in four forms -- raw text, the
JSON-escaped form, the ord-probe token-ID array, and the bracketless
token-ID comma sequence (both ID forms whitespace-stripped, so the canary
is caught however the serializer wraps it, including embedded inside a
longer ID array). Every ``assertNoCanary`` is paired with a guard proving
the scan is not vacuous (the input provably decodes to the canary, or the
output provably rendered the canary-carrying section).

The canary was chosen with no HTML-significant characters (no ``<>&"'``),
so absence of the raw canary in HTML is a meaningful negative: ``escape()``
would leave it byte-identical. It is pure ASCII, so the JSON-escaped form
equals the raw form -- checked explicitly anyway.

Per-surface verdicts (each pinned below):

- ``save_report`` JSON with include_evidence=False: CLEAN. ``_public_view``
  strips ``observations`` and ``_strip_raw_diagnostics`` strips
  ``hand_ids``/``template_ids`` (prompt-reconstructing token IDs),
  ``stop_texts``/``stop_text`` (decoded expected answers) and replaces
  free-text divergence ``detail`` (raw generations) with the fixed
  ``[withheld without --include-evidence]`` marker. include_evidence=True
  keeps all of it by design (proven present -- the barrier, not absence,
  is what the False-path tests measure).
- ``save_report`` HTML: CLEAN. ``render_html`` runs on the *enriched*
  result (not the public view), so this independently pins the HTML-side
  withholding: conformance renders char spans only, the divergence section
  withholds stop text/detail, and no section embeds observations.
- ``divergence_section_data`` (real pipeline): its ``stop_texts`` output is
  a genuine raw-evidence carrier and is stripped by the barrier.
- ``mpl diff-reports``: CLEAN. New since cycle 24: ``config_drift`` now
  carries ``fixtures:search``/``fixtures:held_out`` ID lists; pinned to be
  exactly the sorted fixture-ID lists (identifiers, never content).
- ``mpl diff-backends``: CLEAN / BY DESIGN. Takes no dataset input; with
  canary-free operator args the output is canary-free, and with canary
  args the canary appears only as an echo of the operator's own
  ``--prompt-text``/``--output-text``/``--stops`` (like any CLI echoing
  its argv).
- ``mpl debug`` REPL: BY DESIGN where the operator's own content shows
  (``.render``, ``.tokens``, ``.say`` echo, ``.save`` transcript,
  ``.read``/``--script``, ``.step`` decoding the operator's own typed ID
  stream); CLEAN where it must stay content-free (``.audit`` findings,
  ``.fixtures`` ID list). USAGE.md section 10 documents the contract,
  including "do not paste ``.save`` transcripts into public issues".
- ``mpl doctor`` (function + CLI): CLEAN, including the new portability
  matrix table and ``template_conformance`` fixed-vocabulary details.
- CLI ``inspect`` / ``fixtures build``: ``inspect`` prints counts only
  (CLEAN); ``fixtures build`` writes the operator-authored fixture file
  (canary present BY DESIGN -- it is the operator's own dataset) but the
  ``.audit.json`` sidecar carries findings only (CLEAN).
- ``stop_matrix`` / ``budget_frontier`` / logprob interpretation: CLEAN --
  fixed vocabularies, counts and fixture IDs only.
- ``examples/sample-report/``: CLEAN (re-pinned).

Sensitivity controls (mutation testing, run 2026-10-15 on a /tmp copy of
the package via /tmp/cycle43_mutants/run_mutations.py -- the repo itself
was never mutated; PYTHONPATH shadowing made the test process and its CLI
subprocesses exercise the mutated package):

- M1 ``_strip_raw_diagnostics`` replaced with a no-op: 3 barrier tests
  fail (``test_public_view_strips_every_carrier``,
  ``test_save_report_json_and_html_false_path``,
  ``test_real_divergence_stripped_from_public_json``) -- the sweep catches
  a gutted stripper.
- M2 ``_backend_divergence_section`` renders the finding's ``stop_text``
  instead of the withheld paragraph: 3 tests fail (``..._false_path``,
  ``test_save_report_true_path_keeps_canary`` -- the HTML withholds even
  in evidence mode -- and
  ``test_real_divergence_stripped_from_public_json``) -- the sweep catches
  an HTML-side regression. Note the third failure is the *real* pipeline:
  with MAX_NEW_TOKENS=32 the genuine ``divergence_section_data`` fires
  KEPT_VS_STRIPPED findings whose ``stop_text`` is the canary, so the
  mutant leaks real findings, not just grafted ones.
- M3 ``_public_view`` stops popping ``observations``: the same 3 tests as
  M1 fail -- the sweep catches the cycle-22 leak class reintroduced. (An
  early draft with max_new_tokens=16 caught M3 only via the schema
  assertion, because the truncated generation carried just a canary
  prefix; raising generations to emit the full canary plus scanning the
  bracketless ID sequence made the content scan catch it too.)
- Control (unmutated copy): the full file passes (34 tests + 11 subtests).

Red-team notes (adversarial pass, no leak found):

- ``render_html`` deliberately bypasses ``_public_view`` (it renders the
  enriched dict); every HTML section was audited for free-text renders --
  the only free-text renders are ``escape()``d fixed-vocabulary details
  (conformance ``detail`` from doctor's fixed vocabulary, stop-matrix
  ``observation_incomplete``, logprob ``interpretation.text`` from a fixed
  template set, ``_failure_diagnosis`` codes from the fixed
  ``_FAILURE_GUIDANCE`` vocabulary). Each is pinned canary-free.
- ``Observation.error`` / ``raw_reason`` are allowlisted fixed vocabularies
  in core.py (non-allowlisted backend text collapses to
  ``backend_reported_error``); ``answer`` is the raw carrier and lives
  inside ``observations`` (stripped).
- Fixture IDs are operator-chosen identifiers and are echoed in reports by
  design (outcome grid, decisive lists, drift ID lists); the canary lives
  in message/expected/generated content, never in IDs, and drift entries
  are schema-pinned to ``{field, old, new}`` with ID lists.
- Base64/URL encodings of the canary are not emitted by any surface (all
  serializers are plain JSON/HTML text); the three checked forms cover
  every encoding the codebase produces.

Deterministic: no network, no model downloads, no weights loaded anywhere
(the only tokenizer built is a tiny BPE trained locally on synthetic text).
"""

from __future__ import annotations

import io
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from threading import Event

from model_protocol_lab.catalog import candidates
from model_protocol_lab.core import Generation
from model_protocol_lab.datasets import parse_dataset
from model_protocol_lab.debug import DebugShell
from model_protocol_lab.diffbackends import diff_backends, divergence_section_data
from model_protocol_lab.diffreports import diff_reports, render_summary
from model_protocol_lab.doctor import doctor, examine_dataset, render_portability_table
from model_protocol_lab.experiments import (
    _interpret_logprob_candidate,
    budget_frontier,
    run_experiment,
    stop_matrix,
)
from model_protocol_lab.reporting import _public_view, save_report

REPO_ROOT = Path(__file__).resolve().parent.parent

CANARY = "CANARY43-LEAKSWEEP-QZ7"
# The probe backend encodes text as code points. CANARY_ID_ARRAY is the
# exact standalone array; CANARY_ID_SEQ is the bracketless comma sequence,
# which also catches the canary embedded inside a longer ID array (e.g.
# input_ids, where the prompt's framing IDs surround the canary's).
CANARY_ID_SEQ = ",".join(str(ord(char)) for char in CANARY)
CANARY_ID_ARRAY = "[" + CANARY_ID_SEQ + "]"
# Generations must be long enough to emit the whole 22-char canary, or an
# observations leak would carry only a truncated prefix and the content
# scan could not catch it (mutation M3 proved this).
MAX_NEW_TOKENS = 32
# The fixed replacement _strip_raw_diagnostics writes over free-text detail.
WITHHELD = "[withheld without --include-evidence]"


def _no_ws(text: str) -> str:
    """No ws."""
    return "".join(text.split())


# skipcq: PTC-W0046 - shared base: subclasses carry the test methods; the base runs none by design
class LeakAsserts(unittest.TestCase):
    """Shared canary scanners; not a test case itself (no test_ methods)."""

    def assertNoCanary(self, text: str, surface: str) -> None:  # noqa: N802 -- unittest-style name
        """The canary appears in none of its leak forms in ``text``."""
        self.assertNotIn(CANARY, text, f"raw canary leaked into {surface}")
        self.assertNotIn(
            json.dumps(CANARY)[1:-1],
            text,
            f"escaped canary leaked into {surface}",
        )
        squashed = _no_ws(text)
        self.assertNotIn(
            CANARY_ID_ARRAY,
            squashed,
            f"canary token-ID array leaked into {surface}",
        )
        self.assertNotIn(
            CANARY_ID_SEQ,
            squashed,
            f"canary token-ID sequence leaked into {surface}",
        )

    def assertCanaryPresent(self, text: str, surface: str) -> None:
        """Guard: the scan is not vacuous -- the canary IS in ``text``."""
        self.assertIn(
            CANARY,
            text,
            f"expected canary in {surface} (vacuity guard)",
        )


def _write_minimal_safetensors(path: Path) -> None:
    """Structurally valid minimal safetensors (mirrors tests/test_doctor)."""
    header = {
        "__metadata__": {},
        "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
    }
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"\x00" * 4)


def _write_checkpoint(root: Path) -> Path:
    """Local checkpoint dir with a tiny BPE trained on synthetic text."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer

    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "n_positions": 1024}), encoding="utf-8"
    )
    _write_minimal_safetensors(root / "model.safetensors")
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.train_from_iterator(
        ["hello world foo bar"],
        BpeTrainer(special_tokens=["[UNK]", "[BOS]", "[EOS]"], vocab_size=64),
    )
    tok.save(str(root / "tokenizer.json"))
    return root


def _hostile_document() -> dict:
    """A fixture document whose user content (not ids) carries the canary."""
    return {
        "search": [
            {
                "id": "canary-search-1",
                "group": "canary-g1",
                "messages": [{"role": "user", "content": f"Say the phrase {CANARY} out loud"}],
                "expected": f"the answer is {CANARY}",
            },
            {
                "id": "canary-search-2",
                "group": "canary-g1",
                "messages": [{"role": "user", "content": f"Repeat {CANARY} twice"}],
                "expected": "something else entirely",
            },
        ],
        "held_out": [
            {
                "id": "canary-held-1",
                "group": "canary-g2",
                "messages": [{"role": "user", "content": f"held-out says {CANARY}"}],
                "expected": f"held answer {CANARY}",
            },
        ],
    }


def _dataset_path(directory: Path, document: dict | None = None) -> Path:
    """Dataset path."""
    path = directory / "dataset.json"
    path.write_text(json.dumps(document or _hostile_document()), encoding="utf-8")
    return path


class _CanaryBackend:
    """Ord-based probe backend; generations echo the canary plus boundary."""

    fingerprint = "canary43-probe-backend-v1"

    @staticmethod
    def encode(text: str) -> tuple[int, ...]:
        """Encode."""
        return tuple(map(ord, text))

    @staticmethod
    def decode(token_ids: tuple[int, ...]) -> str:
        """Decode."""
        return "".join(map(chr, token_ids))

    @staticmethod
    def generate(
        input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> Generation:
        """Generate."""
        text = (CANARY + "|").ljust(max_new_tokens)[:max_new_tokens]
        return Generation(tuple(map(ord, text)), "limit")


def _canary_ids() -> list[int]:
    """Canary ids."""
    return [ord(char) for char in CANARY]


def _graft_carriers(result: dict) -> dict:
    """Attach every known raw-evidence carrier, all holding the canary.

    ``template_conformance`` hand/template ID arrays decode to the prompt;
    ``backend_divergences`` stop_texts/detail carry decoded answers and raw
    generations. The schema mirrors exactly what ``doctor`` and
    ``divergence_section_data`` emit, so the barrier is tested against the
    real shapes.
    """
    canary_ids = _canary_ids()
    result["template_conformance"] = {
        "status": "ok",
        "diffs": [
            {
                "format": "llama3",
                "fixture": "canary-search-1",
                "status": "ok",
                "hand_tokens": len(canary_ids),
                "template_tokens": len(canary_ids) + 2,
                "diff": [
                    {
                        "hand_span": [0, len(canary_ids)],
                        "template_span": [0, len(canary_ids) + 2],
                        "hand_ids": canary_ids,
                        "template_ids": canary_ids,
                    }
                ],
            }
        ],
    }
    result["backend_divergences"] = {
        "verdict": "diverge",
        "pairs_verdict": "HF vs vLLM vs SGLang stop semantics: 0/1 agree; 1 divergence",
        "per_fixture": [
            {
                "fixture": "canary-search-1",
                "verdict": "diverge",
                "divergences": [
                    {
                        "class": "KEPT_VS_STRIPPED",
                        "stop_text": CANARY,
                        "detail": f"raw generated text before the stop: {CANARY} tail",
                        "severity": "high",
                    }
                ],
            }
        ],
        "matrix": {"KEPT_VS_STRIPPED": {"hf_stop_string": 1}},
        "severities": {"KEPT_VS_STRIPPED": "high"},
        "stop_texts": [CANARY],
        "eos_id": None,
        "min_tokens": 0,
        "observations_truncated": False,
        "observations_total": 1,
        "observations_processed": 1,
    }
    return result


def _graft_diagnostics(result: dict, split) -> dict:
    """Attach the real diagnostic sections (stop matrix, budget frontier,
    logprob diagnostic) so the HTML render path is swept too. All three are
    aggregates over the hostile run: fixture IDs, counts and fixed
    vocabularies only.
    """
    backend = _CanaryBackend()
    entry = result["search"][0]
    observations = tuple(entry["observations"])
    result["stop_matrix"] = {
        "candidate": "llama3",
        "semantics": [
            {
                "id": "native",
                "label": "Native (lab)",
                "description": "The lab's own behavior.",
            },
            {
                "id": "hf_stop_string",
                "label": "HF StopStringCriteria",
                "description": "Emulated HF stop.",
            },
        ],
        "search": stop_matrix(
            backend, "llama3", split.search, observations, boundary=(124,), eos_id=None
        ),
        "held_out": None,
    }
    result["budget_frontier"] = budget_frontier(
        result["search"],
        boundary=(124,),
        max_new_tokens=MAX_NEW_TOKENS,
        native_decision=result["decision"],
        native_winner=result.get("selected_candidate"),
    )
    hostile_scores = [
        {
            "fixture": "canary-search-1",
            "native_pass": False,
            "status": "ok",
            "reason": None,
            "total_logprob": -1.0,
            "mean_logprob": -0.05,
            "scored_token_count": 20,
            "target_byte_length": 22,
        }
    ]
    result["logprob_diagnostic"] = {
        "available": True,
        "detail": None,
        "method": "teacher-forced",
        "thresholds": {
            "derailed_mean_logprob_at_least": -0.5,
            "unknown_mean_logprob_below": -3.0,
        },
        "candidates": [
            {
                "name": "llama3",
                "fixture_scores": hostile_scores,
                "aggregate": {
                    "native_passes": 0,
                    "native_failures": 1,
                    "scored_token_count": 20,
                    "total_logprob": -1.0,
                    "mean_logprob": -0.05,
                },
                "interpretation": _interpret_logprob_candidate(hostile_scores),
            }
        ],
    }
    return result


def _hostile_result() -> dict:
    """A real run_experiment result over the hostile document, carriers grafted."""
    split = parse_dataset(json.dumps(_hostile_document()))
    result = run_experiment(
        _CanaryBackend(),
        candidates("llama3", mutate=False)[:1],
        split,
        boundary=(124,),
        max_new_tokens=MAX_NEW_TOKENS,
        seed=7,
    )
    _graft_carriers(result)
    _graft_diagnostics(result, split)
    return result


def _walk_keys(value: object, seen: set[str]) -> None:
    """Walk keys."""
    if isinstance(value, dict):
        for key, item in value.items():
            seen.add(str(key))
            _walk_keys(item, seen)
    elif isinstance(value, list):
        for item in value:
            _walk_keys(item, seen)


class PublicViewBarrier(LeakAsserts):
    """The _public_view / _strip_raw_diagnostics barrier over every carrier."""

    def test_public_view_strips_every_carrier(self) -> None:
        """Test public view strips every carrier."""
        result = _hostile_result()
        public = _public_view(result, include_evidence=False)
        text = json.dumps(public, ensure_ascii=True)
        self.assertNoCanary(text, "public results.json (include_evidence=False)")
        self.assertFalse(public["raw_evidence_included"])
        # Observations are gone from both stages...
        for run in public["search"]:
            self.assertNotIn("observations", run)
        held = public.get("held_out")
        if type(held) is dict:
            self.assertNotIn("observations", held)
        # ...prompt-reconstructing token-ID arrays are gone...
        diffs = public["template_conformance"]["diffs"]
        for entry in diffs:
            for segment in entry["diff"]:
                with self.subTest(segment=segment):
                    self.assertNotIn("hand_ids", segment)
                    self.assertNotIn("template_ids", segment)
                    # Spans (positions, not content) stay on the record.
                    self.assertIn("hand_span", segment)
        # ...decoded stop texts are gone...
        divergences = public["backend_divergences"]
        self.assertNotIn("stop_texts", divergences)
        for entry in divergences["per_fixture"]:
            for finding in entry["divergences"]:
                with self.subTest(finding=finding):
                    self.assertNotIn("stop_text", finding)
                    # ...and free-text detail is replaced, never dropped
                    # silently (the fixed class/severity stay).
                    self.assertEqual(finding["detail"], WITHHELD)
                    self.assertEqual(finding["class"], "KEPT_VS_STRIPPED")
        # The fixed-vocabulary aggregates survive the strip.
        self.assertEqual(divergences["matrix"], {"KEPT_VS_STRIPPED": {"hf_stop_string": 1}})

    def test_public_view_true_keeps_evidence_by_design(self) -> None:
        """Non-vacuity: with include_evidence=True the canary IS present."""
        result = _hostile_result()
        public = _public_view(result, include_evidence=True)
        text = json.dumps(public, ensure_ascii=True)
        self.assertCanaryPresent(text, "public results.json (include_evidence=True)")
        self.assertTrue(public["raw_evidence_included"])
        self.assertIn("observations", public["search"][0])
        self.assertIn("stop_texts", public["backend_divergences"])

    def test_public_view_does_not_mutate_caller_dict(self) -> None:
        """Test public view does not mutate caller dict."""
        result = _hostile_result()
        _public_view(result, include_evidence=False)
        self.assertIn("observations", result["search"][0])
        self.assertIn("stop_texts", result["backend_divergences"])

    def test_save_report_json_and_html_false_path(self) -> None:
        """Test save report json and html false path."""
        with tempfile.TemporaryDirectory() as tmp:
            json_path, html_path = save_report(
                _hostile_result(), Path(tmp) / "report", include_evidence=False
            )
            json_text = json_path.read_text(encoding="utf-8")
            html_text = html_path.read_text(encoding="utf-8")
        self.assertNoCanary(json_text, "results.json (include_evidence=False)")
        self.assertNoCanary(html_text, "report.html (include_evidence=False)")
        # Non-vacuous: the evidence-carrying sections did render, just
        # without content -- fixture ids appear in the outcome grid, the
        # divergence section shows its withheld paragraph, the conformance
        # section shows its divergent-segment badge, and the diagnostic
        # sections (stop matrix, frontier, logprob) rendered their tables.
        self.assertIn("canary-search-1", html_text)
        self.assertIn("[stop text and diagnostic detail withheld", html_text)
        self.assertIn("divergent segment(s)", html_text)
        self.assertIn("Stop-behavior isolation matrix", html_text)
        self.assertIn("Token-budget frontier", html_text)
        self.assertIn("Log-likelihood diagnostic", html_text)
        self.assertIn("canary-search-1", json_text)

    def test_save_report_true_path_keeps_canary(self) -> None:
        """Test save report true path keeps canary."""
        with tempfile.TemporaryDirectory() as tmp:
            json_path, html_path = save_report(
                _hostile_result(), Path(tmp) / "report", include_evidence=True
            )
            json_text = json_path.read_text(encoding="utf-8")
            html_text = html_path.read_text(encoding="utf-8")
        self.assertCanaryPresent(json_text, "results.json (include_evidence=True)")
        # The HTML view withholds raw evidence even in evidence mode: the
        # flag only affects the JSON.
        self.assertNoCanary(html_text, "report.html (include_evidence=True)")


class EndToEndDivergencePipeline(LeakAsserts):
    """The real divergence_section_data output through the real barrier."""

    @staticmethod
    def _result_with_real_divergence() -> tuple[dict, dict]:
        """Result with real divergence."""
        split = parse_dataset(json.dumps(_hostile_document()))
        result = run_experiment(
            _CanaryBackend(),
            candidates("llama3", mutate=False)[:1],
            split,
            boundary=(124,),
            max_new_tokens=MAX_NEW_TOKENS,
            seed=7,
        )
        observations = result["search"][0]["observations"]
        section = divergence_section_data(
            observations=observations,
            stop_texts=(CANARY,),
            eos_id=None,
            min_tokens=0,
            backend=_CanaryBackend(),
        )
        assert section is not None
        result["backend_divergences"] = section
        return result, section

    def test_real_divergence_section_carries_canary(self) -> None:
        """Guard: the real pipeline's stop_texts genuinely hold the canary."""
        _, section = self._result_with_real_divergence()
        self.assertEqual(section["stop_texts"], [CANARY])

    def test_real_divergence_stripped_from_public_json(self) -> None:
        """Test real divergence stripped from public json."""
        result, _ = self._result_with_real_divergence()
        with tempfile.TemporaryDirectory() as tmp:
            json_path, html_path = save_report(result, Path(tmp) / "report", include_evidence=False)
            json_text = json_path.read_text(encoding="utf-8")
            html_text = html_path.read_text(encoding="utf-8")
        self.assertNoCanary(json_text, "results.json with real divergence section")
        self.assertNoCanary(html_text, "report.html with real divergence section")
        self.assertNotIn("stop_texts", json.loads(json_text)["backend_divergences"])

    def test_real_divergence_kept_with_evidence(self) -> None:
        """Test real divergence kept with evidence."""
        result, _ = self._result_with_real_divergence()
        with tempfile.TemporaryDirectory() as tmp:
            json_path, _ = save_report(result, Path(tmp) / "report", include_evidence=True)
            json_text = json_path.read_text(encoding="utf-8")
        self.assertCanaryPresent(json_text, "results.json (include_evidence=True)")


def _run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run cli."""
    return subprocess.run(
        [sys.executable, "-m", "model_protocol_lab", *args],
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


class _OrdBackend:
    """Decode-only backend adapter over code-point IDs."""

    @staticmethod
    def decode(token_ids: tuple[int, ...]) -> str:
        """Decode."""
        return "".join(map(chr, token_ids))


class DiffBackendsSweep43(LeakAsserts):
    """diff-backends takes no dataset input; only operator args are echoed."""

    def test_diff_backends_clean_inputs_clean_output(self) -> None:
        """Test diff backends clean inputs clean output."""
        result = diff_backends(
            prompt_ids=(1, 2, 3),
            output_ids=(4, 5),
            stop_texts=("stop!",),
            eos_id=None,
            min_tokens=0,
            backend=_OrdBackend(),
        )
        self.assertNoCanary(json.dumps(result, ensure_ascii=True), "diff-backends")

    def test_diff_backends_echoes_only_operator_supplied_text(self) -> None:
        """By design: the operator's own --stop texts are echoed back (the
        tool has no dataset input for content to come from).
        """
        result = diff_backends(
            prompt_ids=tuple(map(ord, f"prompt {CANARY}")),
            output_ids=tuple(map(ord, "output stop!")),
            stop_texts=(CANARY, "stop!"),
            eos_id=None,
            min_tokens=0,
            backend=_OrdBackend(),
        )
        self.assertEqual(result["stop_texts"], [CANARY, "stop!"])
        # The echo is exactly the operator's argv, nothing more: with
        # canary-free args the output cannot contain the canary.
        clean = diff_backends(
            prompt_ids=(1, 2, 3),
            output_ids=(4, 5),
            stop_texts=("stop!",),
            eos_id=None,
            min_tokens=0,
            backend=_OrdBackend(),
        )
        self.assertNoCanary(json.dumps(clean, ensure_ascii=True), "diff-backends (clean args)")

    def test_cli_diff_backends_clean_args(self) -> None:
        """Test cli diff backends clean args."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = _write_checkpoint(Path(tmp) / "ckpt")
            proc = _run_cli(
                "diff-backends",
                "--model",
                str(ckpt),
                "--prompt-text",
                "hello world",
                "--output-text",
                "hello stop!",
                "--stops",
                "stop!",
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNoCanary(proc.stdout, "mpl diff-backends stdout (clean args)")
        self.assertNoCanary(proc.stderr, "mpl diff-backends stderr (clean args)")

    def test_cli_diff_backends_echoes_operator_args_by_design(self) -> None:
        """Test cli diff backends echoes operator args by design."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = _write_checkpoint(Path(tmp) / "ckpt")
            proc = _run_cli(
                "diff-backends",
                "--model",
                str(ckpt),
                "--prompt-text",
                f"operator prompt {CANARY}",
                "--output-text",
                "output stop!",
                "--stops",
                f"{CANARY},stop!",
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        # By design: the operator's own argv is echoed in stop_texts.
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["stop_texts"], [CANARY, "stop!"])
        self.assertNoCanary(proc.stderr, "mpl diff-backends stderr (canary args)")


class DiffReportsSweep43(LeakAsserts):
    """diff-reports reduces observations to verdicts; drift carries IDs only."""

    @staticmethod
    def _evidence_report(directory: Path, document: dict | None = None) -> dict:
        """Evidence report."""
        split = parse_dataset(json.dumps(document or _hostile_document()))
        result = run_experiment(
            _CanaryBackend(),
            candidates("llama3", mutate=False)[:1],
            split,
            boundary=(124,),
            max_new_tokens=MAX_NEW_TOKENS,
            seed=7,
        )
        json_path, _ = save_report(result, directory, include_evidence=True)
        return json.loads(json_path.read_text(encoding="utf-8"))

    @staticmethod
    def _decoded_text(report: dict) -> str:
        """Decoded text."""
        decode = _CanaryBackend.decode
        chunks = []
        for entry in report["search"]:
            for obs in entry.get("observations", []):
                chunks.append(decode(tuple(obs["input_ids"])))
                chunks.append(decode(tuple(obs["output_ids"])))
        return "\n".join(chunks)

    def test_self_diff_clean(self) -> None:
        """Test self diff clean."""
        with tempfile.TemporaryDirectory() as tmp:
            report = self._evidence_report(Path(tmp) / "run")
        self.assertIn(CANARY, self._decoded_text(report), "vacuity guard")
        diff = diff_reports(report, report, all_fixtures=True)
        self.assertTrue(diff["evidence"]["old"] and diff["evidence"]["new"])
        self.assertTrue(diff["fixture_diffs"])
        text = json.dumps(diff, indent=2, ensure_ascii=True, sort_keys=True)
        self.assertNoCanary(text, "diff-reports JSON (self diff, all_fixtures)")
        self.assertNoCanary(render_summary(diff), "diff-reports stderr summary")

    def test_fixture_set_drift_carries_ids_only(self) -> None:
        """New since cycle 24: config_drift lists fixture ID sets; the
        canary in a *new* fixture's content must not reach the diff.
        """
        document = _hostile_document()
        document["search"].append(
            {
                "id": "canary-search-3",
                "group": "canary-g3",
                "messages": [{"role": "user", "content": f"brand new {CANARY} content"}],
                "expected": "new answer",
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            old = self._evidence_report(Path(tmp) / "old")
            new = self._evidence_report(Path(tmp) / "new", document=document)
        diff = diff_reports(old, new, all_fixtures=True)
        drift = [e for e in diff["config_drift"] if e["field"] == "fixtures:search"]
        self.assertEqual(len(drift), 1, "the fixture-set drift entry is emitted")
        entry = drift[0]
        # Schema pin: exactly {field, old, new}, and old/new are exactly the
        # sorted fixture-ID lists -- identifiers, never content.
        self.assertEqual(set(entry), {"field", "old", "new"})
        self.assertEqual(entry["old"], ["canary-search-1", "canary-search-2"])
        self.assertEqual(entry["new"], ["canary-search-1", "canary-search-2", "canary-search-3"])
        text = json.dumps(diff, indent=2, ensure_ascii=True, sort_keys=True)
        self.assertNoCanary(text, "diff-reports JSON (fixture-set drift)")
        self.assertNoCanary(render_summary(diff), "diff-reports summary (fixture-set drift)")

    def test_held_out_drift_carries_ids_only(self) -> None:
        """Test held out drift carries ids only."""

        # The probe run selects no winner, so held_out evidence is absent
        # from the raw run_experiment output; fabricate it on both sides
        # (diff_reports only reads exact_match/boundary_found verdicts,
        # fixture ids, and summaries) with the canary planted in an
        # observation's answer and input_ids -- the ultimate reduction
        # proof: even raw content in an observation must not reach the diff.
        def with_held_out(report: dict, extra: bool) -> dict:
            """With held out."""
            observations = [
                {
                    "fixture": "canary-held-1",
                    "exact_match": True,
                    "boundary_found": True,
                    "answer": CANARY,
                    "input_ids": [ord(char) for char in CANARY],
                }
            ]
            if extra:
                observations.append(
                    {
                        "fixture": "canary-held-2",
                        "exact_match": False,
                        "boundary_found": True,
                    }
                )
            report["selected_candidate"] = "llama3"
            report["held_out"] = {
                "summary": {
                    "planned": len(observations),
                    "exact_and_boundary_passes": 1,
                    "aggregate_eligible": True,
                },
                "observations": observations,
            }
            return report

        with tempfile.TemporaryDirectory() as tmp:
            old = with_held_out(self._evidence_report(Path(tmp) / "old"), extra=False)
            new = with_held_out(self._evidence_report(Path(tmp) / "new"), extra=True)
        diff = diff_reports(old, new, all_fixtures=True)
        drift = [e for e in diff["config_drift"] if e["field"] == "fixtures:held_out"]
        self.assertEqual(len(drift), 1)
        self.assertEqual(set(drift[0]), {"field", "old", "new"})
        self.assertEqual(drift[0]["old"], ["canary-held-1"])
        self.assertEqual(drift[0]["new"], ["canary-held-1", "canary-held-2"])
        text = json.dumps(diff, indent=2, ensure_ascii=True, sort_keys=True)
        self.assertNoCanary(text, "diff-reports JSON (held-out drift)")
        self.assertNoCanary(render_summary(diff), "diff-reports summary (held-out drift)")

    def test_cli_diff_reports_clean(self) -> None:
        """The `mpl diff-reports` CLI: JSON on stdout, summary on stderr."""
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = Path(tmp) / "old"
            new_dir = Path(tmp) / "new"
            self._evidence_report(old_dir)
            self._evidence_report(new_dir)
            proc = _run_cli("diff-reports", str(old_dir), str(new_dir), "--all", cwd=Path(tmp))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        diff = json.loads(proc.stdout)
        self.assertTrue(diff["summary"]["identical"])
        self.assertNoCanary(proc.stdout, "mpl diff-reports stdout")
        self.assertNoCanary(proc.stderr, "mpl diff-reports stderr")


class _OrdTokenizer:
    """Stub tokenizer for the debug REPL: code-point encode/decode."""

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> tuple[int, ...]:
        """Encode."""
        return tuple(map(ord, text))

    @staticmethod
    def decode(token_ids: list[int] | tuple[int, ...]) -> str:
        """Decode."""
        return "".join(map(chr, token_ids))


class DebugReplSweep43(LeakAsserts):
    """The debug REPL shows the operator's own dataset BY DESIGN (USAGE.md
    section 10); surfaces that must stay content-free are pinned clean.
    """

    @staticmethod
    def _session(commands: list[str], tokenizer=None) -> str:
        """Session."""
        split = parse_dataset(json.dumps(_hostile_document()))
        out = io.StringIO()
        shell = DebugShell(split, dataset_label="canary-fixtures", tokenizer=tokenizer, stdout=out)
        shell.cmdqueue = list(commands)
        shell.cmdloop()
        return out.getvalue()

    def test_render_shows_fixture_content_by_design(self) -> None:
        """Test render shows fixture content by design."""
        out = self._session([".render", ".quit"])
        self.assertCanaryPresent(out, "debug .render (by design)")

    def test_save_persists_transcript_by_design(self) -> None:
        """.save writes the session transcript (echoes + output) to the
        operator-chosen path; USAGE.md section 10 warns not to paste it
        into public issues.
        """
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "transcript.txt")
            self._session([".render", f".save {target}", ".quit"])
            saved = Path(target).read_text(encoding="utf-8")
        self.assertCanaryPresent(saved, "debug .save transcript (by design)")

    def test_read_script_runs_operator_script_by_design(self) -> None:
        """Test read script runs operator script by design."""
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "session.txt"
            script.write_text(".render\n.quit\n", encoding="utf-8")
            split = parse_dataset(json.dumps(_hostile_document()))
            out = io.StringIO()
            shell = DebugShell(split, dataset_label="canary", stdout=out)
            stopped = shell.read_commands(script)
        self.assertTrue(stopped, ".quit inside the script stops the session")
        self.assertCanaryPresent(out.getvalue(), "debug .read (by design)")

    def test_step_decodes_operator_typed_stream_by_design(self) -> None:
        """With a tokenizer attached, .step prints text= for each walked
        token -- but the walked stream is the operator's own typed IDs, so
        this is an echo of operator input, not a fixture leak.
        """
        canary_ids = ",".join(str(ord(char)) for char in CANARY[:4])
        out = self._session(
            [".boundary 124", f".step {canary_ids}", ".quit"],
            tokenizer=_OrdTokenizer(),
        )
        self.assertIn("text='C'", out)
        self.assertIn("overhang=", out)

    def test_fixture_select_echoes_sanitized_id(self) -> None:
        """Test fixture select echoes sanitized id."""
        out = self._session([".fixture canary-search-1", ".quit"])
        self.assertIn("canary-search-1", out)
        self.assertNoCanary(out, "debug .fixture (ids carry no content)")

    def test_audit_findings_carry_no_message_text(self) -> None:
        """Test audit findings carry no message text."""
        document = _hostile_document()
        document["search"][0]["messages"][0]["content"] += " <|im_start|> literal"
        split = parse_dataset(json.dumps(document))
        out = io.StringIO()
        shell = DebugShell(split, dataset_label="canary", stdout=out)
        shell.cmdqueue = [".audit", ".quit"]
        shell.cmdloop()
        text = out.getvalue()
        self.assertIn("<|im_start|>", text, "the literal is actually flagged")
        self.assertNoCanary(text, "debug .audit")

    def test_fixtures_lists_ids_only(self) -> None:
        """Test fixtures lists ids only."""
        out = self._session([".fixtures", ".quit"])
        self.assertNoCanary(out, "debug .fixtures")

    def test_tokens_without_tokenizer_prints_reminder(self) -> None:
        """Test tokens without tokenizer prints reminder."""
        out = self._session([".tokens", ".quit"])
        self.assertNoCanary(out, "debug .tokens (no tokenizer)")
        self.assertIn("no tokenizer attached", out)

    def test_cli_debug_script_by_design(self) -> None:
        """Test cli debug script by design."""
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset_path(Path(tmp))
            script = Path(tmp) / "session.txt"
            script.write_text(".render\n.quit\n", encoding="utf-8")
            proc = _run_cli("debug", "--dataset", str(dataset), "--script", str(script))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertCanaryPresent(proc.stdout, "mpl debug --script (by design)")
        self.assertNoCanary(proc.stderr, "mpl debug --script stderr")


def _hostile_csv() -> str:
    """Hostile csv."""
    lines = [
        "id,group,partition,expected,user_1",
        f"csv-1,csvg,search,expected {CANARY},user says {CANARY} here",
        "csv-2,csvg,search,other answer,plain question <|im_start|> literal",
        f"csv-3,csvh,held_out,held {CANARY},held asks {CANARY}",
    ]
    return "\n".join(lines) + "\n"


class FixturesBuildSweep(LeakAsserts):
    """`fixtures build`: the fixture file is the operator's own dataset (its
    content appears BY DESIGN); the audit sidecar must stay content-free.
    """

    def test_fixtures_build_sidecar_carries_no_content(self) -> None:
        """Test fixtures build sidecar carries no content."""
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "authoring.csv"
            source.write_text(_hostile_csv(), encoding="utf-8")
            output = Path(tmp) / "fixtures.json"
            proc = _run_cli("fixtures", "build", str(source), "--output", str(output))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual((payload["search_cases"], payload["held_out_cases"]), (2, 1))
            self.assertGreaterEqual(payload["audit_findings"], 1, "the literal is flagged")
            sidecar = Path(payload["audit_path"])
            sidecar_text = sidecar.read_text(encoding="utf-8")
            fixture_text = output.read_text(encoding="utf-8")
        findings = json.loads(sidecar_text)["findings"]
        self.assertEqual(len(findings), payload["audit_findings"])
        # Schema pin: every finding carries the catalog literal, ids and
        # offsets -- never message text. The literal collides with several
        # catalog formats, hence multiple findings for the one fixture.
        for finding in findings:
            with self.subTest(finding=finding):
                self.assertEqual(
                    set(finding),
                    {"fixture_id", "field", "literal", "format_name", "offset"},
                )
                self.assertEqual(finding["fixture_id"], "csv-2")
                self.assertEqual(finding["literal"], "<|im_start|>")
        self.assertNoCanary(sidecar_text, "fixtures build audit sidecar")
        # The built fixture file IS the operator's dataset: the canary is
        # there by design (same bytes the operator authored in the CSV).
        self.assertCanaryPresent(fixture_text, "built fixture file (by design)")
        # stderr warnings name the literal and fixture id, not message text.
        self.assertIn("<|im_start|>", proc.stderr)
        self.assertIn("csv-2", proc.stderr)
        self.assertNoCanary(proc.stderr, "fixtures build stderr")


class InspectSweep43(LeakAsserts):
    """`mpl inspect` prints counts only."""

    def test_inspect_emits_counts_only(self) -> None:
        """Test inspect emits counts only."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _dataset_path(Path(tmp))
            proc = _run_cli("inspect", str(path))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(set(payload), {"search_cases", "held_out_cases", "valid", "contamination"})
        self.assertEqual((payload["search_cases"], payload["held_out_cases"]), (2, 1))
        self.assertNoCanary(proc.stdout, "mpl inspect stdout")
        self.assertNoCanary(proc.stderr, "mpl inspect stderr")


class DoctorSweep43(LeakAsserts):
    """`mpl doctor`: checks and counts only, incl. the portability matrix."""

    def test_examine_dataset_clean(self) -> None:
        """Test examine dataset clean."""
        with tempfile.TemporaryDirectory() as tmp:
            report = examine_dataset(_dataset_path(Path(tmp)))
        text = json.dumps(report, ensure_ascii=True)
        self.assertNoCanary(text, "doctor examine_dataset")
        self.assertEqual(
            (report["dataset"]["search_cases"], report["dataset"]["held_out_cases"]),
            (2, 1),
        )

    def test_doctor_function_clean(self) -> None:
        """Test doctor function clean."""
        with tempfile.TemporaryDirectory() as tmp:
            report = doctor(
                _write_checkpoint(Path(tmp) / "ckpt"),
                _dataset_path(Path(tmp)),
                boundary_ids=(1, 2),
            )
        self.assertEqual(report["verdict"], "ok")
        self.assertNoCanary(json.dumps(report, ensure_ascii=True), "doctor() report")
        # template_conformance details are a fixed vocabulary, never text.
        conformance = report.get("template_conformance") or {}
        detail = conformance.get("detail")
        if detail is not None:
            self.assertNotIn(CANARY, str(detail))

    def test_cli_doctor_with_portability_matrix_clean(self) -> None:
        """Test cli doctor with portability matrix clean."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = _write_checkpoint(Path(tmp) / "ckpt")
            ref = _write_checkpoint(Path(tmp) / "ref")
            dataset = _dataset_path(Path(tmp))
            proc = _run_cli(
                "doctor",
                "--model",
                str(ckpt),
                "--dataset",
                str(dataset),
                "--boundary-ids",
                "1,2",
                "--portability-matrix",
                str(ref),
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["verdict"], "ok")
        self.assertNoCanary(proc.stdout, "mpl doctor stdout")
        # The human-readable portability table goes to stderr; it renders
        # catalog token inventory, never fixture text.
        self.assertNoCanary(proc.stderr, "mpl doctor stderr (portability table)")

    def test_portability_table_renders_no_fixture_text(self) -> None:
        """Test portability table renders no fixture text."""
        with tempfile.TemporaryDirectory() as tmp:
            report = doctor(
                _write_checkpoint(Path(tmp) / "ckpt"),
                _dataset_path(Path(tmp)),
                boundary_ids=(1, 2),
                portability_matrix_path=_write_checkpoint(Path(tmp) / "ref"),
            )
        section = report.get("portability_matrix")
        self.assertIsInstance(section, dict)
        assert isinstance(section, dict)
        table = render_portability_table(section)
        self.assertNoCanary(table, "render_portability_table")
        self.assertNoCanary(json.dumps(section, ensure_ascii=True), "portability_matrix JSON")


class ScoringSweep43(LeakAsserts):
    """stop_matrix / budget_frontier / logprob interpretation: aggregates only."""

    @staticmethod
    def _winner_observations() -> tuple[tuple, tuple[dict, ...]]:
        """Winner observations."""
        split = parse_dataset(json.dumps(_hostile_document()))
        result = run_experiment(
            _CanaryBackend(),
            candidates("llama3", mutate=False)[:1],
            split,
            boundary=(124,),
            max_new_tokens=MAX_NEW_TOKENS,
            seed=7,
        )
        entry = result["search"][0]
        observations = tuple(entry["observations"])
        return split.search, observations

    def test_stop_matrix_rows_carry_no_content(self) -> None:
        """Test stop matrix rows carry no content."""
        fixtures, observations = self._winner_observations()
        rows = stop_matrix(
            _CanaryBackend(),
            "llama3",
            fixtures,
            observations,
            boundary=(124,),
            eos_id=None,
        )
        text = json.dumps(rows, ensure_ascii=True)
        self.assertNoCanary(text, "stop_matrix rows")
        # Schema pin: rows carry fixture ids, booleans, fixed-vocab cells.
        # StopOutcome.detail is a fixed vocabulary by construction
        # (backends.StopOutcome docstring: "never raw backend text").
        fixed_details = {
            "observation_incomplete",
            "eos_id_unavailable",
            "invalid_emulation_input",
            "token_text_unavailable",
            "decode_unavailable",
            "answer_decode_unavailable",
            "no_stop_within_cap",
            "gated_by_min_tokens",
        }
        self.assertTrue(rows)
        for row in rows:
            with self.subTest(fixture=row["fixture"]):
                self.assertEqual(set(row), {"fixture", "native_pass", "cells", "flips"})
                for semantic, cell in row["cells"].items():
                    self.assertIn(semantic, ("hf_stop_string", "vllm_holdback", "eos_appended"))
                    detail = cell.get("detail", "")
                    self.assertIsInstance(detail, str)
                    self.assertTrue(
                        detail in fixed_details or detail.startswith("fired_at_step_"),
                        f"unexpected cell detail: {detail!r}",
                    )

    def test_budget_frontier_carries_no_content(self) -> None:
        """Test budget frontier carries no content."""
        split = parse_dataset(json.dumps(_hostile_document()))
        result = run_experiment(
            _CanaryBackend(),
            candidates("llama3", mutate=False)[:1],
            split,
            boundary=(124,),
            max_new_tokens=MAX_NEW_TOKENS,
            seed=7,
        )
        frontier = budget_frontier(
            result["search"],
            boundary=(124,),
            max_new_tokens=MAX_NEW_TOKENS,
            native_decision=result["decision"],
            native_winner=result.get("selected_candidate"),
        )
        self.assertNoCanary(json.dumps(frontier, ensure_ascii=True), "budget_frontier")

    def test_logprob_interpretation_text_is_fixed_vocabulary(self) -> None:
        """Test logprob interpretation text is fixed vocabulary."""
        hostile_scores = [
            {
                "fixture": "canary-search-1",
                "native_pass": False,
                "status": "ok",
                "mean_logprob": -0.05,
                "total_logprob": -1.0,
                "scored_token_count": 20,
            }
        ]
        interpretation = _interpret_logprob_candidate(hostile_scores)
        self.assertNoCanary(json.dumps(interpretation, ensure_ascii=True), "logprob interpretation")
        self.assertEqual(interpretation["kind"], "greedy_derailed")


class SampleReportHygiene43(LeakAsserts):
    """The committed examples/sample-report/ stays synthetic and evidence-free."""

    def test_committed_sample_report_clean(self) -> None:
        """Test committed sample report clean."""
        raw = (REPO_ROOT / "examples" / "sample-report" / "results.json").read_bytes()
        report = json.loads(raw)
        self.assertFalse(report["raw_evidence_included"])
        keys: set[str] = set()
        _walk_keys(report, keys)
        for banned in ("observations", "input_ids", "output_ids", "hand_ids", "stop_texts"):
            self.assertNotIn(banned, keys, f"{banned} must not ship in the sample report")
        html_text = (REPO_ROOT / "examples" / "sample-report" / "report.html").read_text()
        self.assertNoCanary(raw.decode("utf-8"), "sample results.json")
        self.assertNoCanary(html_text, "sample report.html")
        # Non-vacuous: the sample fixtures' ids do appear by design.
        fixtures = json.loads((REPO_ROOT / "examples" / "fixtures.json").read_text())
        first_id = fixtures["search"][0]["id"]
        self.assertIn(first_id, raw.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
