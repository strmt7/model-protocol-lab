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
