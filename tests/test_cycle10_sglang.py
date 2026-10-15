"""Cycle 10: the SGLang column for ``mpl diff-backends``.

Implements the Cycle 1 research recommendation (``docs/RESEARCH_LOG.md``,
"Continuous dev -- Cycle 1", SGLang/TGI section, VERIFIED against
sgl-project/sglang @ 12dc623):

- ``emulate_sglang_stop`` (``backends.py``): ports
  ``Req._check_str_based_finish`` (``schedule_batch.py:1836-1847``) --
  first step whose decoded prefix contains the stop string fires -- and
  ``DetokenizerManager.trim_matched_stop``
  (``detokenizer_manager.py:190-221``), which cuts the returned text at
  the leftmost stop occurrence by default (``no_stop_trim=False``,
  ``sampling_params.py:153``), like vLLM and unlike HF. The emulation
  takes NO ``min_tokens`` parameter: upstream ``min_new_tokens`` only
  clips the length budget (``scheduler.py:2631-2634``); the finish path
  never consults it.
- ``MULTI_STOP_SELECTION`` (high): SGLang selects the first-listed stop
  (list-order iteration, with the upstream
  ``# TODO(lmzheng): handle the case where multiple stop strs are hit``)
  while vLLM selects the earliest-completing one
  (``min(stop_index + len(stop_str))``, ties by list order; vLLM PR
  #49391, pinned by ``tests/detokenizer/test_check_stop_strings.py``).
  Pins the research-log caveat: with 1 token/step the
  earlier-completing stop still fires at its own step, so the rules
  agree; list order decides only when >=2 stops match within one check.
- ``NO_MIN_TOKENS_GATE`` (medium): the SGLang column on the existing
  ``MIN_TOKENS_GATED`` class -- a stop vLLM's ``min_tokens`` gate blocks
  still fires under the SGLang emulation.
- ``STR_OVER_EOS_PRECEDENCE`` (high): stop string and EOS fire in the
  same step, so precedence decides the outcome and the engines disagree:
  SGLang's shipped fix for #28802 gives the stop string the win
  (``update_finish_state`` runs the str-based check before the
  token-based one, ``schedule_batch.py:1913-1950``), while TGI's
  documented precedence is length > EOS > stop-sequence (EOS beats the
  stop string -- the reverse of SGLang). Co-fires with ``EOS_PRECEDENCE``
  on the tie.

No TGI cell, deliberately (see ``test_no_tgi_cell_rationale``): every
TGI stop rule is covered by composition of the existing cells, so a TGI
column would duplicate them without adding a distinguishable rule.
SGLang ``stop_regex`` gets a USAGE.md note only (outside stop-string
scope): no cell, pinned by ``test_stop_regex_has_no_cell``.

All backends are fakes decoding through fixed per-token text tables;
no torch, no network, no model weights.
"""

import inspect
import json
import unittest
from html import escape
from typing import Any

from model_protocol_lab.backends import emulate_sglang_stop
from model_protocol_lab.diffbackends import (
    DIVERGENCE_SEVERITY,
    HF_STOP_STRING,
    SGLANG_STOP,
    VLLM_HOLDBACK_STRIP,
    diff_backends,
    divergence_section_data,
)
from model_protocol_lab.reporting import render_html


class TableBackend:
    """Fake backend decoding token ids through a fixed per-token text table.

    Mirrors the cycle-6/7 diff-backend fake: whole-sequence decodes
    concatenate per-token texts.
    """

    fingerprint = "cycle10-sglang-table-backend-not-a-model"

    def __init__(self, texts: dict[int, str]) -> None:
        self._texts = dict(texts)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(self._texts[token] for token in token_ids)


class SglangEmulationTests(unittest.TestCase):
    """The ``emulate_sglang_stop`` port itself, against vLLM/HF behavior."""

    def setUp(self) -> None:
        # 1->"a", 2->"b", 3->"STOP", 4->"c": stop "STOP" completes at step 3.
        self.backend = TableBackend({1: "a", 2: "b", 3: "STOP", 4: "c"})

    def test_fires_at_first_step_containing_stop(self) -> None:
        outcome = emulate_sglang_stop(self.backend, (1, 2, 3, 4), "STOP")
        self.assertTrue(outcome.available)
        self.assertTrue(outcome.fired)
        self.assertEqual(outcome.detail, "fired_at_step_3")
        self.assertEqual(outcome.truncated_ids, (1, 2, 3))

    def test_strips_at_leftmost_occurrence_by_default(self) -> None:
        # Default no_stop_trim=False: trim_matched_stop cuts at the
        # leftmost stop occurrence -- like vLLM's default, unlike HF.
        outcome = emulate_sglang_stop(self.backend, (1, 2, 3, 4), "STOP")
        self.assertEqual(outcome.returned_text, "ab")

    def test_no_stop_trim_keep_variant(self) -> None:
        outcome = emulate_sglang_stop(self.backend, (1, 2, 3, 4), "STOP", no_stop_trim=True)
        self.assertTrue(outcome.fired)
        self.assertEqual(outcome.returned_text, "abSTOP")

    def test_no_min_tokens_parameter(self) -> None:
        # The API-level absence is the divergence: SGLang has no
        # min_tokens equivalent, so the emulation cannot take one.
        params = inspect.signature(emulate_sglang_stop).parameters
        self.assertNotIn("min_tokens", params)
        self.assertNotIn("min_new_tokens", params)

    def test_sglang_column_ungated_where_vllm_gates(self) -> None:
        # min_tokens=1 gates vLLM's holdback (the stop completes at step 1
        # and the step-2 window no longer covers it) but SGLang fires.
        backend = TableBackend({1: "STOP", 2: "x"})
        result = diff_backends(
            prompt_ids=(),
            output_ids=(1, 2),
            stop_texts=("STOP",),
            min_tokens=1,
            backend=backend,
        )
        semantics = result["cells"][0]["semantics"]
        strip = semantics[VLLM_HOLDBACK_STRIP]
        self.assertTrue(strip["available"])
        self.assertFalse(strip["fired"])
        sglang = semantics[SGLANG_STOP]
        self.assertTrue(sglang["available"])
        self.assertTrue(sglang["fired"])
        self.assertEqual(sglang["firing_step"], 1)
        self.assertEqual(sglang["disposition"], "stripped")
        self.assertEqual(sglang["returned_text"], "")

    def test_sglang_firing_step_matches_vllm_and_hf_on_clean_stop(self) -> None:
        result = diff_backends(
            prompt_ids=(),
            output_ids=(1, 2, 3, 4),
            stop_texts=("STOP",),
            backend=self.backend,
        )
        semantics = result["cells"][0]["semantics"]
        steps = {
            name: cell["firing_step"] for name, cell in semantics.items() if name != "eos_appended"
        }
        self.assertEqual(steps[HF_STOP_STRING], 3)
        self.assertEqual(steps[VLLM_HOLDBACK_STRIP], 3)
        self.assertEqual(steps["vllm_holdback_keep"], 3)
        self.assertEqual(steps[SGLANG_STOP], 3)

    def test_invalid_inputs_degrade(self) -> None:
        outcome = emulate_sglang_stop(self.backend, (1, 2), "")
        self.assertFalse(outcome.available)
        outcome = emulate_sglang_stop(self.backend, (1, "x"), "STOP")  # type: ignore[arg-type]
        self.assertFalse(outcome.available)
        outcome = emulate_sglang_stop(self.backend, (1, 2), "STOP", no_stop_trim="yes")  # type: ignore[arg-type]
        self.assertFalse(outcome.available)


class MultiStopSelectionTests(unittest.TestCase):
    """``MULTI_STOP_SELECTION``: first-listed (SGLang) vs earliest-completing
    (vLLM PR #49391)."""

    def setUp(self) -> None:
        # One token decodes to several chars, so two stops can complete
        # within ONE check step -- the research-log caveat case.
        # 1->" The user", 2->" is a": at step 2 both "a" and "is" match.
        self.backend = TableBackend({1: " The user", 2: " is a"})

    def _run(self, stop_texts: tuple[str, ...], **kwargs: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "prompt_ids": (),
            "output_ids": (1, 2),
            "stop_texts": stop_texts,
            "backend": self.backend,
        }
        base.update(kwargs)
        return diff_backends(**base)

    def _classes(self, result: dict[str, Any]) -> list[str]:
        return [finding["class"] for finding in result["divergences"]]

    def test_first_listed_vs_earliest_completing_diverges(self) -> None:
        # stops ("a", "is"): SGLang picks "a" (first-listed); vLLM picks
        # "is" (completes earliest: end index 12 < 14) -- the hard
        # divergence, mirroring the vLLM #49391 repro.
        result = self._run(("a", "is"))
        self.assertIn("MULTI_STOP_SELECTION", self._classes(result))
        finding = next(
            finding
            for finding in result["divergences"]
            if finding["class"] == "MULTI_STOP_SELECTION"
        )
        self.assertIn("sglang_stop selects stop 'a' at step 2", finding["detail"])
        self.assertIn("first-listed", finding["detail"])
        self.assertIn("vllm_holdback_strip selects stop 'is' at step 2", finding["detail"])
        self.assertIn("earliest-completion", finding["detail"])
        self.assertIn("49391", finding["detail"])
        self.assertEqual(finding["stop_text"], "a")

    def test_agree_when_first_listed_is_earliest(self) -> None:
        # stops ("is", "a"): both rules pick "is" -- no selection
        # divergence (KEPT_VS_STRIPPED still fires; filter by class).
        result = self._run(("is", "a"))
        self.assertNotIn("MULTI_STOP_SELECTION", self._classes(result))

    def test_earlier_step_still_wins_with_one_token_per_step(self) -> None:
        # The research-log caveat: with 1 token/step the
        # earlier-completing stop fires at its own step regardless of list
        # order, so both engines agree. 1->"x", 2->"is", 3->" a": "is"
        # completes at step 2, "a" at step 3; SGLang's winner is "is"
        # (smallest firing step beats list order).
        backend = TableBackend({1: "x", 2: "is", 3: " a"})
        result = diff_backends(
            prompt_ids=(),
            output_ids=(1, 2, 3),
            stop_texts=("a", "is"),
            backend=backend,
        )
        self.assertNotIn("MULTI_STOP_SELECTION", self._classes(result))
        self.assertEqual(result["verdict"], "diverge")  # KEPT_VS_STRIPPED still fires

    def test_no_selection_divergence_when_nothing_fires(self) -> None:
        result = self._run(("ZZZ", "QQQ"))
        self.assertNotIn("MULTI_STOP_SELECTION", self._classes(result))
        self.assertEqual(result["verdict"], "agree")

    def test_single_stop_text_never_diverges_on_selection(self) -> None:
        result = self._run(("is",))
        self.assertNotIn("MULTI_STOP_SELECTION", self._classes(result))

    def test_vllm_tie_broken_by_list_order(self) -> None:
        # stops ("ab", "b") on text "ab": both complete with end index 2
        # in the same step; vLLM breaks the tie by list order (PR #49391
        # pins this) and SGLang is first-listed too -- agree.
        backend = TableBackend({1: "ab"})
        result = diff_backends(
            prompt_ids=(), output_ids=(1,), stop_texts=("ab", "b"), backend=backend
        )
        self.assertNotIn("MULTI_STOP_SELECTION", self._classes(result))


class NoMinTokensGateTests(unittest.TestCase):
    """``NO_MIN_TOKENS_GATE``: the SGLang column on ``MIN_TOKENS_GATED``."""

    def test_sglang_fires_where_vllm_gate_blocks(self) -> None:
        backend = TableBackend({1: "STOP", 2: "x"})
        result = diff_backends(
            prompt_ids=(),
            output_ids=(1, 2),
            stop_texts=("STOP",),
            min_tokens=1,
            backend=backend,
        )
        by_class = {finding["class"]: finding for finding in result["divergences"]}
        self.assertIn("MIN_TOKENS_GATED", by_class)
        self.assertIn("NO_MIN_TOKENS_GATE", by_class)
        finding = by_class["NO_MIN_TOKENS_GATE"]
        self.assertEqual(finding["stop_text"], "STOP")
        self.assertIn("sglang_stop", finding["detail"])
        self.assertIn("no min_tokens equivalent", finding["detail"])
        self.assertIn("22014", finding["detail"])
        self.assertEqual(DIVERGENCE_SEVERITY["NO_MIN_TOKENS_GATE"], "medium")

    def test_no_gate_classes_without_min_tokens(self) -> None:
        backend = TableBackend({1: "STOP", 2: "x"})
        result = diff_backends(
            prompt_ids=(),
            output_ids=(1, 2),
            stop_texts=("STOP",),
            min_tokens=0,
            backend=backend,
        )
        classes = [finding["class"] for finding in result["divergences"]]
        self.assertNotIn("NO_MIN_TOKENS_GATE", classes)
        self.assertNotIn("MIN_TOKENS_GATED", classes)


class StrOverEosPrecedenceTests(unittest.TestCase):
    """``STR_OVER_EOS_PRECEDENCE``: same-step stop+EOS tie, where precedence
    -- not firing order -- decides the outcome."""

    def test_same_step_tie_fires_with_precedence_rationale(self) -> None:
        # Token 1 decodes to "STOP" and IS the EOS id: both fire at step 1.
        backend = TableBackend({1: "STOP"})
        result = diff_backends(
            prompt_ids=(),
            output_ids=(1,),
            stop_texts=("STOP",),
            eos_id=1,
            backend=backend,
        )
        by_class = {finding["class"]: finding for finding in result["divergences"]}
        # The tie also trips the firing-order class; the new class records
        # the precedence disagreement on top of it.
        self.assertIn("EOS_PRECEDENCE", by_class)
        self.assertIn("STR_OVER_EOS_PRECEDENCE", by_class)
        finding = by_class["STR_OVER_EOS_PRECEDENCE"]
        self.assertEqual(finding["stop_text"], "STOP")
        self.assertIn("#28802", finding["detail"])
        self.assertIn("TGI", finding["detail"])
        self.assertIn("length > EOS > stop-sequence", finding["detail"])
        self.assertEqual(DIVERGENCE_SEVERITY["STR_OVER_EOS_PRECEDENCE"], "high")

    def test_no_tie_no_precedence_class(self) -> None:
        # EOS at step 1, stop completes at step 2: firing order decides,
        # so only EOS_PRECEDENCE fires.
        backend = TableBackend({1: "E", 2: "STOP"})
        result = diff_backends(
            prompt_ids=(),
            output_ids=(1, 2),
            stop_texts=("STOP",),
            eos_id=1,
            backend=backend,
        )
        classes = [finding["class"] for finding in result["divergences"]]
        self.assertIn("EOS_PRECEDENCE", classes)
        self.assertNotIn("STR_OVER_EOS_PRECEDENCE", classes)


class NoTgiCellRationaleTests(unittest.TestCase):
    """Why there is deliberately no TGI column.

    Every TGI stop rule from the Cycle 1 research is covered by
    composition of the existing cells, so a TGI column would duplicate
    them without adding a distinguishable rule:

    - TGI keeps stop sequences (verified constructively at every layer:
      ``StoppingCriteria`` only detects; full decode incl. stop tokens;
      Rust v3 forwards; router verbatim) -- covered by the
      kept-disposition ``hf_stop_string`` cell.
    - TGI multi-stop is first-listed -- the same rule as the
      ``sglang_stop`` selection in ``MULTI_STOP_SELECTION``.
    - TGI has no ``min_tokens`` equivalent -- like ``sglang_stop``'s
      ungated finish path (``NO_MIN_TOKENS_GATE``).
    - TGI precedence is length > EOS > stop-sequence (EOS beats stop) --
      the documented contrast in ``STR_OVER_EOS_PRECEDENCE``.

    The one genuine TGI quirk (match buffer truncated to the last 200
    chars, so a stop sequence longer than 200 chars can never fire) is
    too narrow for a cell and stays research-log-only.
    """

    def test_no_tgi_semantic_exists(self) -> None:
        backend = TableBackend({1: "a", 2: "b", 3: "STOP", 4: "c"})
        result = diff_backends(
            prompt_ids=(),
            output_ids=(1, 2, 3, 4),
            stop_texts=("STOP",),
            backend=backend,
        )
        semantics = result["cells"][0]["semantics"]
        self.assertTrue(
            all("tgi" not in name.lower() for name in semantics),
            f"unexpected TGI-like cell: {sorted(semantics)}",
        )

    def test_tgi_kept_rule_covered_by_hf_cell(self) -> None:
        backend = TableBackend({1: "a", 2: "b", 3: "STOP", 4: "c"})
        result = diff_backends(
            prompt_ids=(),
            output_ids=(1, 2, 3, 4),
            stop_texts=("STOP",),
            backend=backend,
        )
        hf = result["cells"][0]["semantics"][HF_STOP_STRING]
        self.assertTrue(hf["fired"])
        self.assertEqual(hf["disposition"], "kept")

    def test_tgi_first_listed_rule_covered_by_sglang_selection(self) -> None:
        # TGI's first-listed multi-stop rule is behaviorally identical to
        # the SGLang side of MULTI_STOP_SELECTION: first-listed wins.
        backend = TableBackend({1: " The user", 2: " is a"})
        result = diff_backends(
            prompt_ids=(),
            output_ids=(1, 2),
            stop_texts=("a", "is"),
            backend=backend,
        )
        finding = next(
            finding
            for finding in result["divergences"]
            if finding["class"] == "MULTI_STOP_SELECTION"
        )
        self.assertIn("sglang_stop selects stop 'a' at step 2", finding["detail"])

    def test_tgi_no_min_tokens_covered_by_sglang_ungated_path(self) -> None:
        # TGI has no min_tokens equivalent, like SGLang: the sglang cell
        # fires identically with and without the gate.
        backend = TableBackend({1: "STOP", 2: "x"})
        plain = diff_backends(
            prompt_ids=(), output_ids=(1, 2), stop_texts=("STOP",), backend=backend
        )
        gated = diff_backends(
            prompt_ids=(),
            output_ids=(1, 2),
            stop_texts=("STOP",),
            min_tokens=999,
            backend=backend,
        )
        plain_cell = plain["cells"][0]["semantics"][SGLANG_STOP]
        gated_cell = gated["cells"][0]["semantics"][SGLANG_STOP]
        self.assertTrue(plain_cell["fired"])
        self.assertEqual(plain_cell["firing_step"], gated_cell["firing_step"])
        self.assertEqual(plain_cell["returned_text"], gated_cell["returned_text"])

    def test_tgi_eos_beats_stop_is_the_documented_contrast(self) -> None:
        backend = TableBackend({1: "STOP"})
        result = diff_backends(
            prompt_ids=(),
            output_ids=(1,),
            stop_texts=("STOP",),
            eos_id=1,
            backend=backend,
        )
        finding = next(
            finding
            for finding in result["divergences"]
            if finding["class"] == "STR_OVER_EOS_PRECEDENCE"
        )
        # TGI's reverse precedence is cited as the contrast, not emulated
        # as a cell.
        self.assertIn("TGI", finding["detail"])


class SglangReportTests(unittest.TestCase):
    """The report's Backend divergence section: new classes render with
    severity badges, the SGLang column appears in the matrix, hostile
    strings are escaped, rendering is byte-deterministic."""

    def setUp(self) -> None:
        self.backend = TableBackend({1: " The user", 2: " is a", 3: "STOP", 4: "x"})

    def _section(self, **kwargs: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "observations": [
                # MULTI_STOP_SELECTION: ("a","is") both complete at step 2.
                {"fixture": "fx-multi", "input_ids": (), "output_ids": (1, 2)},
                # NO_MIN_TOKENS_GATE: "STOP" completes at step 1, gated.
                {"fixture": "fx-gate", "input_ids": (), "output_ids": (3, 4)},
            ],
            "stop_texts": ("a", "is", "STOP"),
            "eos_id": 3,
            "min_tokens": 1,
            "backend": self.backend,
        }
        base.update(kwargs)
        section = divergence_section_data(**base)
        self.assertIsNotNone(section)
        assert section is not None
        return section

    def test_new_classes_in_matrix_with_severities(self) -> None:
        section = self._section()
        matrix = section["matrix"]
        for kind in ("MULTI_STOP_SELECTION", "NO_MIN_TOKENS_GATE", "STR_OVER_EOS_PRECEDENCE"):
            self.assertIn(kind, matrix, f"{kind} missing from matrix")
        # SGLang column is attributed in every new class's counts.
        self.assertGreater(matrix["MULTI_STOP_SELECTION"].get("sglang_stop", 0), 0)
        self.assertGreater(matrix["MULTI_STOP_SELECTION"].get("vllm_holdback_strip", 0), 0)
        self.assertGreater(matrix["NO_MIN_TOKENS_GATE"].get("sglang_stop", 0), 0)
        self.assertGreater(matrix["NO_MIN_TOKENS_GATE"].get("vllm_holdback_strip", 0), 0)
        self.assertGreater(matrix["STR_OVER_EOS_PRECEDENCE"].get("sglang_stop", 0), 0)
        self.assertGreater(matrix["STR_OVER_EOS_PRECEDENCE"].get("eos_appended", 0), 0)
        severities = section["severities"]
        self.assertEqual(severities["MULTI_STOP_SELECTION"], "high")
        self.assertEqual(severities["NO_MIN_TOKENS_GATE"], "medium")
        self.assertEqual(severities["STR_OVER_EOS_PRECEDENCE"], "high")

    def test_report_renders_new_classes_with_badges(self) -> None:
        section = self._section()
        html = render_html(
            {"decision": "compatible_on_fixture_set", "backend_divergences": section}
        )
        self.assertIn('id="backend-divergence"', html)
        for kind in ("MULTI_STOP_SELECTION", "NO_MIN_TOKENS_GATE", "STR_OVER_EOS_PRECEDENCE"):
            self.assertIn(kind, html)
        # Severity badges reuse the existing palette: high -> fail red,
        # medium -> neutral slate.
        self.assertIn('<span class="badge fail">high</span>', html)
        self.assertIn('<span class="badge neutral">medium</span>', html)
        # The SGLang column header in the class x semantic matrix.
        self.assertIn("SGLang stop", html)
        # Byte-deterministic rendering.
        html2 = render_html(
            {"decision": "compatible_on_fixture_set", "backend_divergences": section}
        )
        self.assertEqual(html, html2)

    def test_hostile_strings_escaped(self) -> None:
        hostile_stop = "<script>alert(1)</script>"
        hostile_fixture = "<img src=x onerror=alert(1)>"
        section = self._section(
            observations=[
                {"fixture": hostile_fixture, "input_ids": (), "output_ids": (1, 2)},
            ],
            stop_texts=(hostile_stop, "is"),
        )
        html = render_html(
            {"decision": "compatible_on_fixture_set", "backend_divergences": section}
        )
        # The fixture name still renders, escaped. The stop text is withheld
        # from the HTML view entirely (neither raw nor escaped may appear):
        # it can carry raw prompts/generations.
        self.assertNotIn(hostile_fixture, html)
        self.assertIn(escape(hostile_fixture), html)
        self.assertNotIn(hostile_stop, html)
        self.assertNotIn(escape(hostile_stop), html)
        self.assertIn("[withheld from this HTML view]", html)

    def test_stop_regex_has_no_cell(self) -> None:
        # SGLang stop_regex is outside stop-string scope: USAGE.md note
        # only, never a cell or a divergence class.
        result = diff_backends(
            prompt_ids=(),
            output_ids=(3, 4),
            stop_texts=("STOP",),
            backend=self.backend,
        )
        semantics = result["cells"][0]["semantics"]
        self.assertTrue(all("regex" not in name for name in semantics))
        classes = [finding["class"] for finding in result["divergences"]]
        self.assertTrue(all("REGEX" not in kind for kind in classes))


class SglangDeterminismTests(unittest.TestCase):
    def test_diff_backends_deterministic_with_sglang_column(self) -> None:
        backend = TableBackend({1: " The user", 2: " is a", 3: "STOP", 4: "x"})
        kwargs: dict[str, Any] = {
            "prompt_ids": (),
            "output_ids": (1, 2, 3, 4),
            "stop_texts": ("a", "is", "STOP"),
            "eos_id": 3,
            "min_tokens": 1,
            "backend": backend,
        }
        first = diff_backends(**kwargs)
        second = diff_backends(**kwargs)
        self.assertEqual(
            json.dumps(first, sort_keys=True, ensure_ascii=True),
            json.dumps(second, sort_keys=True, ensure_ascii=True),
        )
        # JSON round-trip stable.
        self.assertEqual(
            json.loads(json.dumps(first)),
            json.loads(json.dumps(second)),
        )


if __name__ == "__main__":
    unittest.main()
