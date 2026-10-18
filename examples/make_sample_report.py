"""Generate the committed sample report in examples/sample-report/.

This is a SYNTHETIC demonstration, not a real model run: a scripted backend
stands in for a checkpoint that prefers the unmutated base hypothesis, so the
sample exercises the full pipeline (search -> unique winner -> held-out ->
compatible_on_fixture_set) and shows what a real report looks like.

Regenerating must be byte-identical; tests/test_sample_report.py enforces it.
"""

import sys
from pathlib import Path
from threading import Event

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_protocol_lab.catalog import candidates
from model_protocol_lab.core import Generation
from model_protocol_lab.datasets import load_dataset
from model_protocol_lab.experiments import run_experiment
from model_protocol_lab.reporting import save_report

REPO_ROOT = Path(__file__).resolve().parent.parent
BOUNDARY = (124,)  # ord("|")
MAX_NEW_TOKENS = 16

# The unmutated llama3 hypothesis renders prompts starting with exactly one
# BOS marker and ending with the standard open-assistant prefix. Mutated
# variants either repeat or drop the BOS marker or alter the assistant prefix,
# so this marker test identifies the base hypothesis from the prompt text alone.
WINNER_PREFIX = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
WINNER_SUFFIX = "<|start_header_id|>assistant<|end_header_id|>\n\n"


class DemoBackend:
    """Scripted stand-in for a checkpoint; deterministic and local."""

    fingerprint = "synthetic-demo-backend-v1-not-a-model"

    @staticmethod
    def encode(text: str) -> tuple[int, ...]:
        return tuple(map(ord, text))

    @staticmethod
    def decode(token_ids: tuple[int, ...]) -> str:
        return "".join(map(chr, token_ids))

    def generate(
        self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> Generation:
        prompt = self.decode(input_ids)
        is_winner = (
            prompt.startswith(WINNER_PREFIX)
            and prompt.count("<|begin_of_text|>") == 1
            and prompt.endswith(WINNER_SUFFIX)
        )
        if is_winner:
            if "GREEN" in prompt:
                answer = "GREEN"
            elif "BLUE" in prompt:
                answer = "BLUE"
            elif "numeral 7" in prompt:
                answer = "7"
            elif "SOUTH" in prompt:
                answer = "SOUTH"
            else:
                answer = "WRONG"
        else:
            answer = "WRONG"
        text = (answer + "|").ljust(max_new_tokens)[:max_new_tokens]
        return Generation(tuple(map(ord, text)), "limit")


class HierarchyDemoBackend(DemoBackend):
    """Scripted instruction-following policy for the hierarchy exercise.

    Parses the conflict prompt for the privileged-block answer and the
    user-instruction answer, then: follows the privileged system
    instruction when the block precedes the user instruction (primacy),
    follows the user instruction when the block comes last (recency) --
    except the no-bos variant, which always follows the user. This gives
    the committed sample report genuine adherence variation across both
    positions and candidates, exactly the signal the probe measures.
    """

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        import re

        from model_protocol_lab.hierarchy import PRIVILEGE_CLOSE, PRIVILEGE_OPEN

        prompt = self.decode(input_ids)
        block_start = prompt.find(PRIVILEGE_OPEN)
        block_end = prompt.find(PRIVILEGE_CLOSE)
        system_answer = user_answer = None
        user_at = None
        for match in re.finditer(r"[Rr]eply with exactly:\s*(\S+)", prompt):
            if 0 <= block_start <= match.start() <= block_end:
                system_answer = match.group(1)
            else:
                user_answer = match.group(1)
                user_at = match.start()
        if system_answer is None or user_answer is None or user_at is None:
            return Generation((), "error", "unparseable")
        rebel = not prompt.startswith("<|begin_of_text|>")
        if rebel:
            answer = user_answer
        elif block_start < user_at:
            answer = system_answer
        else:
            answer = user_answer
        text = (answer + "|").ljust(max_new_tokens)[:max_new_tokens]
        return Generation(tuple(map(ord, text)), "limit")


class PrefillDemoBackend(DemoBackend):
    """Scripted prefill policy for the prefill exercise.

    The prefill stage appends the prefill ids after the encoded prompt,
    so the decoded prompt ends with the prefill text; the tail is
    stripped before applying the winner logic. The base candidate honors
    the prefill (emits the plain answer); the no-bos variant restarts
    its turn and re-emits the prefill, duplicating it -- exactly the
    honoring vs restarting signal the experiment measures. This gives
    the committed sample report genuine honoring variation across
    candidates.
    """

    def __init__(self, prefill_text: str) -> None:
        self._prefill_text = prefill_text

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        prompt = self.decode(input_ids)
        if prompt.endswith(self._prefill_text):
            prompt = prompt[: -len(self._prefill_text)]
        is_winner = (
            prompt.startswith(WINNER_PREFIX)
            and prompt.count("<|begin_of_text|>") == 1
            and prompt.endswith(WINNER_SUFFIX)
        )
        if is_winner:
            if "GREEN" in prompt:
                answer = "GREEN"
            elif "BLUE" in prompt:
                answer = "BLUE"
            elif "numeral 7" in prompt:
                answer = "7"
            elif "SOUTH" in prompt:
                answer = "SOUTH"
            else:
                answer = "WRONG"
        else:
            answer = "WRONG"
        rebel = not prompt.startswith("<|begin_of_text|>")
        emitted = (self._prefill_text + answer) if rebel else answer
        text = (emitted + "|").ljust(max_new_tokens)[:max_new_tokens]
        return Generation(tuple(map(ord, text)), "limit")


class SynthhistoryDemoBackend(DemoBackend):
    """Scripted history-sensitivity policy for the synthhistory exercise.

    Behaves exactly like DemoBackend, except the base candidate is
    steered off-answer once the synthetic history reaches three
    exchanges: it echoes the last synthetic assistant turn instead of
    answering -- an emulation of the many-shot steering effect the
    probe measures (longer histories steer harder). One exchange is
    tolerated, so the sample report shows a genuine winner flip only
    under history_6.
    """

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        from model_protocol_lab.synthhistory import SYNTHETIC_TURNS

        prompt = self.decode(input_ids)
        n_exchanges = sum(turn.content in prompt for turn in SYNTHETIC_TURNS[1::2])
        if n_exchanges >= 3 and prompt.startswith(WINNER_PREFIX):
            echo = SYNTHETIC_TURNS[-1].content + "|"
            text = echo.ljust(max_new_tokens)[:max_new_tokens]
            return Generation(tuple(map(ord, text)), "limit")
        return super().generate(input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel)


class BatchDemoBackend(DemoBackend):
    """Scripted batch-capable backend for the batch-invariance exercise.

    Solo ``generate`` behaves exactly like DemoBackend. The scripted
    ``generate_batch`` path returns the identical solo outputs for
    every prompt except the first, where it flips one continuation
    token -- the scripted demonstration that the audit detects
    batch-induced divergence (the exact signal the experiment
    measures). Deterministic.
    """

    def generate_batch(self, prompts, *, max_new_tokens, seed, cancel):
        outputs = []
        for index, ids in enumerate(prompts):
            solo = self.generate(ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel)
            tokens = solo.token_ids
            if index == 0 and tokens:
                # Scripted divergence on the first prompt only, emulating
                # batch-composition nondeterminism.
                tokens = (tokens[0] ^ 1,) + tokens[1:]
            outputs.append(tokens)
        return tuple(outputs)


class PositionDemoBackend(DemoBackend):
    """Scripted positional-bias policy for the position exercise.

    Parses the option lines out of the rendered multiple-choice prompt,
    then emulates a primacy-biased policy for the base candidate (emit
    the first-listed option's text) and a recency-biased policy for the
    no-bos variant (emit the last-listed option's text) -- exactly the
    primacy/recency signal the experiment measures. This gives the
    committed sample report genuine primacy-index variation (+1.0 vs
    -1.0) and genuine winner flips under repositioning.
    """

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        import re

        prompt = self.decode(input_ids)
        options = [
            match.group(2) for match in re.finditer(r"^([A-D])\. (.+)$", prompt, re.MULTILINE)
        ]
        if not options:
            return Generation((), "error", "unparseable")
        rebel = not prompt.startswith("<|begin_of_text|>")
        answer = options[-1] if rebel else options[0]
        text = (answer + "|").ljust(max_new_tokens)[:max_new_tokens]
        return Generation(tuple(map(ord, text)), "limit")


def build_sample_report(output_dir: Path) -> dict:
    split = load_dataset(REPO_ROOT / "examples" / "fixtures.json")
    result = run_experiment(
        DemoBackend(),
        candidates("llama3", mutate=True),
        split,
        boundary=BOUNDARY,
        max_new_tokens=MAX_NEW_TOKENS,
        seed=20261012,
    )
    # Token-budget frontier: attached exactly as `mpl run` does, so the
    # committed sample report exercises the frontier section (chart and
    # decision ribbon). Degrades to an "unavailable" marker instead of
    # failing the build.
    try:
        from model_protocol_lab.experiments import budget_frontier

        result["budget_frontier"] = budget_frontier(
            result.get("search") or [],
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            native_decision=result["decision"],
            native_winner=result.get("selected_candidate"),
        )
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["budget_frontier"] = {
            "available": False,
            "detail": "frontier_unavailable",
        }
    # Held-out budget floor: attached exactly as `mpl run` does, so the
    # committed sample report exercises the held-out floor callout.
    # Degrades to an "unavailable" marker instead of failing the build.
    try:
        from model_protocol_lab.experiments import held_out_budget_floor

        held_out = result.get("held_out") or {}
        held_observations = held_out.get("observations") or []
        held_boundary = result.get("selected_boundary_token_ids")
        if (
            result["decision"] in ("compatible_on_fixture_set", "held_out_failed")
            and held_observations
            and held_boundary
        ):
            result["held_out_budget_floor"] = held_out_budget_floor(
                held_observations,
                boundary=tuple(held_boundary),
                max_new_tokens=MAX_NEW_TOKENS,
                native_verdict=result["decision"],
            )
        else:
            result["held_out_budget_floor"] = {
                "available": False,
                "detail": "held_out_incomplete",
            }
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["held_out_budget_floor"] = {
            "available": False,
            "detail": "held_out_floor_unavailable",
        }
    # Fixture hygiene: attached exactly as `mpl run` does, so the committed
    # sample report exercises the contamination callout in the
    # run-configuration section.
    try:
        from model_protocol_lab.contamination import contamination_report

        result["contamination"] = contamination_report(split.search, split.held_out)
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["contamination"] = {"worst_tier": "clean", "pairs": []}
    # Prompt-boundary canonicality: attached exactly as `model-protocol-lab
    # run` does, so the committed sample report exercises the canonicality
    # section. Degrades to an "unavailable" marker instead of failing the
    # build.
    try:
        from model_protocol_lab.boundary_audit import audit_search_observations

        backend = DemoBackend()
        result["boundary_audit"] = audit_search_observations(
            result.get("search") or [],
            backend.encode,
            backend.decode,
        )
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["boundary_audit"] = {
            "available": False,
            "detail": "boundary_audit_unavailable",
        }
    # Chat-marker isolation: attached via the real pure probe with a
    # scripted tokenizer in which every hostile marker literal is a control
    # token (mirroring split_special_tokens=False), so the committed sample
    # report exercises the isolation matrix. Degrades to an "unavailable"
    # marker instead of failing the build.
    try:
        from model_protocol_lab.control_tokens import (
            HOSTILE_MARKER_STRINGS,
            probe_format,
            summarize_probes,
        )

        _SPEC_IDS = frozenset(1000 + k for k in range(len(HOSTILE_MARKER_STRINGS)))

        def _demo_marker_encode(text: str) -> tuple[int, ...]:
            ids: list[int] = []
            i = 0
            while i < len(text):
                for k, marker in enumerate(HOSTILE_MARKER_STRINGS):
                    if text.startswith(marker, i):
                        ids.append(1000 + k)
                        i += len(marker)
                        break
                else:
                    ids.append(ord(text[i]) % 1000)
                    i += 1
            return tuple(ids)

        _probes: list[dict] = []
        for _fmt in candidates("llama3", mutate=True):
            for _case in split.search:
                _pf = probe_format(_fmt.render, _demo_marker_encode, _SPEC_IDS, _case.messages)
                for _p in _pf["probes"]:
                    _probes.append(
                        {
                            "format": _fmt.name,
                            "fixture": _case.id,
                            "marker": _p["marker"],
                            "status": "ok",
                            "baseline": _p["baseline"],
                            "probed": _p["probed"],
                            "delta": _p["delta"],
                            "isolated": _p["isolated"],
                        }
                    )
        result["marker_isolation"] = summarize_probes(_probes, len(_SPEC_IDS))
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["marker_isolation"] = {
            "status": "unavailable",
            "detail": "marker_isolation_unavailable",
            "probes": [],
        }
    # Template firewall: attached via the real static scanner on a benign
    # synthetic template, so the committed sample report exercises the
    # firewall section. Degrades to an "unavailable" marker instead of
    # failing the build.
    try:
        from model_protocol_lab.template_audit import scan_template_ast

        _demo_template = (
            "{% for message in messages %}"
            "{{ message.role }}: {{ message.content }}\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}assistant:{% endif %}"
        )
        _fw_findings = scan_template_ast(_demo_template)
        if _fw_findings:
            result["template_firewall"] = {
                "status": "denied",
                "detail": "template_firewall_denied",
                "findings": [{"rule": f.rule, "detail": f.detail} for f in _fw_findings],
            }
        else:
            result["template_firewall"] = {
                "status": "allowed",
                "detail": "no_rce_primitives_found",
                "findings": [],
            }
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["template_firewall"] = {
            "status": "unavailable",
            "detail": "template_firewall_unavailable",
            "findings": [],
        }
    # Selection robustness: attached via the real post-hoc analysis on the
    # run's own search observations, so the committed sample report
    # exercises the new section. Degrades to an "unavailable" marker
    # instead of failing the build.
    try:
        from model_protocol_lab.experiments import selection_robustness

        result["selection_robustness"] = selection_robustness(
            DemoBackend(),
            result,
            tuple(split.search),
            boundary=BOUNDARY,
        )
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["selection_robustness"] = {
            "available": False,
            "detail": "selection_robustness_unavailable",
        }
    # Oracle-mixture / protocol-portfolio gap (research wave-2 idea 4):
    # pure post-hoc over the run's own stored search observations
    # (exact answer AND explicit boundary), so the committed sample
    # report exercises the portfolio section. Degrades to an
    # "unavailable" marker instead of failing the build.
    try:
        from model_protocol_lab.experiments import portfolio_analysis

        result["portfolio"] = portfolio_analysis(result, tuple(split.search))
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["portfolio"] = {
            "available": False,
            "detail": "portfolio_unavailable",
        }
    # Cross-checkpoint transfer: self-transfer against the same synthetic
    # backend, so the committed sample report exercises the transfer
    # section (a self-transfer must show "transfers" everywhere -- the
    # sanity property from the IDEAS brief). Degrades to an "unavailable"
    # marker instead of failing the build.
    try:
        from model_protocol_lab.experiments import transfer_portability, transfer_stage

        _fmts = candidates("llama3", mutate=True)
        _outcomes = transfer_stage(
            lambda cid: DemoBackend(),
            ("demo-self",),
            _fmts,
            tuple(split.search),
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            seed=20261012,
        )
        _transfer = transfer_portability(
            result.get("selected_candidate"),
            result["decision"],
            tuple(f.name for f in _fmts),
            _outcomes,
        )
        _transfer["checkpoints"] = [
            {
                "id": "demo-self",
                "path": "synthetic",
                "fingerprint": DemoBackend.fingerprint,
            }
        ]
        result["transfer"] = _transfer
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["transfer"] = {
            "available": False,
            "detail": "transfer_unavailable",
        }
    # Fixture-perturbation robustness: run the real perturbation stage
    # (N=4, one variant per operator) against the synthetic backend, so
    # the committed sample report exercises the perturbation section
    # with genuine flips (the scripted backend keys its answer on exact
    # keywords, so case/typo/Unicode noise really does flip variants --
    # exactly the phrasing-luck signal the experiment measures).
    # Degrades to an "unavailable" marker instead of failing the build.
    try:
        from model_protocol_lab.experiments import perturbation_robustness, perturbation_stage
        from model_protocol_lab.perturbation import PERTURBATION_OPERATORS

        _p_fmts = candidates("llama3", mutate=True)
        _p_stage = perturbation_stage(
            DemoBackend(),
            _p_fmts,
            tuple(split.search),
            n_variants=4,
            operators=PERTURBATION_OPERATORS,
            seed=20261012,
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        result["perturbation"] = perturbation_robustness(
            result.get("selected_candidate"),
            result["decision"],
            tuple(f.name for f in _p_fmts),
            _p_stage,
        )
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["perturbation"] = {
            "available": False,
            "detail": "perturbation_unavailable",
        }
    # Instruction-hierarchy adherence probe: run the real hierarchy
    # stage (both privilege-block positions) against the scripted policy
    # backend, so the committed sample report exercises the hierarchy
    # section with genuine adherence variation and position flips.
    # Degrades to an "unavailable" marker instead of failing the build.
    try:
        from model_protocol_lab.experiments import hierarchy_adherence, hierarchy_stage
        from model_protocol_lab.hierarchy import (
            CONFLICT_SPECS,
            POSITIONS,
            build_conflict_fixtures,
            validate_conflict_specs,
        )

        _h_fmts = candidates("llama3", mutate=True)
        _h_specs = validate_conflict_specs(CONFLICT_SPECS)
        _h_stages = {}
        for _position in POSITIONS:
            _h_fixtures, _h_expectations = build_conflict_fixtures(_h_specs, position=_position)
            _h_stages[_position] = hierarchy_stage(
                HierarchyDemoBackend(),
                _h_fmts,
                _h_fixtures,
                _h_expectations,
                position=_position,
                boundary=BOUNDARY,
                max_new_tokens=MAX_NEW_TOKENS,
                seed=20261012,
            )
        result["hierarchy"] = hierarchy_adherence(
            result.get("selected_candidate"),
            result["decision"],
            tuple(f.name for f in _h_fmts),
            _h_stages,
        )
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["hierarchy"] = {
            "available": False,
            "detail": "hierarchy_unavailable",
        }
    # Assistant-prefill sensitivity experiment: run the real prefill
    # stage (every format-lock prefill) against the scripted
    # honor/restart policy backend, so the committed sample report
    # exercises the prefill section with genuine honoring variation
    # (the no-bos variant restarts its turn and duplicates the
    # prefill). Degrades to an "unavailable" marker instead of failing
    # the build.
    try:
        from model_protocol_lab.experiments import prefill_adherence, prefill_stage
        from model_protocol_lab.prefill import PREFILL_SPECS, validate_prefill_specs

        _p_fmts = candidates("llama3", mutate=True)
        _p_specs = validate_prefill_specs(PREFILL_SPECS)
        _p_stages = {}
        for _p_spec in _p_specs:
            _p_stages[_p_spec["id"]] = prefill_stage(
                PrefillDemoBackend(_p_spec["text"]),
                _p_fmts,
                tuple(split.search),
                prefill=_p_spec,
                boundary=BOUNDARY,
                max_new_tokens=MAX_NEW_TOKENS,
                seed=20261012,
            )
        result["prefill"] = prefill_adherence(
            result.get("selected_candidate"),
            result["decision"],
            tuple(f.name for f in _p_fmts),
            _p_stages,
        )
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["prefill"] = {
            "available": False,
            "detail": "prefill_unavailable",
        }
    # Synthetic-history selection stability experiment: run the real
    # synthhistory stage (both history conditions) against the scripted
    # history-sensitivity policy backend, so the committed sample report
    # exercises the synthhistory section with a genuine winner flip (the
    # base candidate tolerates one history exchange but is steered
    # off-answer by three). Degrades to an "unavailable" marker instead
    # of failing the build.
    try:
        from model_protocol_lab.experiments import (
            synthhistory_stability,
            synthhistory_stage,
        )
        from model_protocol_lab.synthhistory import CONDITIONS, build_history_fixtures

        _s_fmts = candidates("llama3", mutate=True)
        _s_stages = {}
        for _s_condition in CONDITIONS:
            _s_fixtures, _ = build_history_fixtures(tuple(split.search), condition=_s_condition)
            _s_stages[_s_condition] = synthhistory_stage(
                SynthhistoryDemoBackend(),
                _s_fmts,
                _s_fixtures,
                condition=_s_condition,
                boundary=BOUNDARY,
                max_new_tokens=MAX_NEW_TOKENS,
                seed=20261012,
            )
        result["synthhistory"] = synthhistory_stability(
            result.get("selected_candidate"),
            result["decision"],
            tuple(f.name for f in _s_fmts),
            _s_stages,
        )
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["synthhistory"] = {
            "available": False,
            "detail": "synthhistory_unavailable",
        }
    # Batch-invariance audit: run the real batch stage against the
    # scripted batch-capable backend, whose batch path diverges on one
    # fixture -- the sample report exercises the audit with a genuine
    # detected divergence (the exact signal the experiment measures).
    # Degrades to an "unavailable" marker instead of failing the build.
    try:
        from model_protocol_lab.experiments import batch_stage

        _b_backend = BatchDemoBackend()
        _b_fmts = candidates("llama3", mutate=True)
        _b_prompts = {}
        for _b_case in tuple(split.search):
            _b_prompts[_b_case.id] = _b_backend.encode(_b_fmts[0].render(_b_case.messages))
        result["batch_invariance"] = batch_stage(
            _b_backend,
            _b_prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            seed=20261012,
        )
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["batch_invariance"] = {
            "available": False,
            "detail": "batch_unavailable",
        }
    # Position-sensitivity experiment: run the real position stage
    # (every option-position condition) against the scripted
    # primacy/recency policy backend, so the committed sample report
    # exercises the position section with genuine primacy-index
    # variation and winner flips (the base candidate answers the
    # first-listed option, the no-bos variant the last). Degrades to
    # an "unavailable" marker instead of failing the build.
    try:
        from model_protocol_lab.experiments import position_sensitivity, position_stage
        from model_protocol_lab.position import (
            CONDITIONS,
            POSITION_SPECS,
            build_position_fixtures,
            validate_position_specs,
        )

        _pos_fmts = candidates("llama3", mutate=True)
        _pos_specs = validate_position_specs(POSITION_SPECS)
        _pos_stages = {}
        for _pos_condition in CONDITIONS:
            _pos_fixtures = build_position_fixtures(
                _pos_specs, condition=_pos_condition, seed=20261012
            )
            _pos_stages[_pos_condition] = position_stage(
                PositionDemoBackend(),
                _pos_fmts,
                _pos_fixtures,
                condition=_pos_condition,
                boundary=BOUNDARY,
                max_new_tokens=MAX_NEW_TOKENS,
                seed=20261012,
            )
        result["position"] = position_sensitivity(
            result.get("selected_candidate"),
            result["decision"],
            tuple(f.name for f in _pos_fmts),
            _pos_stages,
        )
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["position"] = {
            "available": False,
            "detail": "position_unavailable",
        }
    json_path, html_path = save_report(result, output_dir)
    print(f"decision: {result['decision']}")
    print(f"wrote {json_path} and {html_path}")
    return result


if __name__ == "__main__":
    target = REPO_ROOT / "examples" / "sample-report"
    if target.exists():
        import shutil

        shutil.rmtree(target)
    build_sample_report(target)
