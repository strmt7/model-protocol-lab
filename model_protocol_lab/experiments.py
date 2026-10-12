"""Bounded search followed by one frozen held-out evaluation."""
from dataclasses import asdict
from threading import Event
from .core import Backend, Format, Split, evaluate


def _summary(report):
    passed = sum(o.exact_match and o.boundary_found for o in report.observations)
    return {"status": report.status, "planned": report.planned_count,
            "attempted": report.attempted_count,
            "aggregate_eligible": report.aggregate_eligible,
            "exact_and_boundary_passes": passed,
            "rate": passed / report.planned_count if report.aggregate_eligible else None,
            "encoded_input_tokens": sum(len(o.input_ids) for o in report.observations),
            "retained_output_tokens": sum(len(o.output_ids) for o in report.observations)}


def run_experiment(backend: Backend, formats: tuple[Format, ...], split: Split, *,
                   boundary: tuple[int, ...], max_new_tokens: int = 64,
                   seed: int = 0, cancel: Event | None = None) -> dict:
    if type(formats) is not tuple or not 1 <= len(formats) <= 24:
        raise ValueError("Supply between 1 and 24 candidates")
    if any(type(fmt) is not Format for fmt in formats):
        raise ValueError("Invalid candidate type")
    if len({fmt.name for fmt in formats}) != len(formats):
        raise ValueError("Candidate names must be unique")
    if type(split) is not Split:
        raise ValueError("A validated split is required")
    if len(split.search) > 100 or len(split.held_out) > 100:
        raise ValueError("Experiment fixture limit exceeded")
    event = cancel if cancel is not None else Event()
    output = {"schema_version": 1, "selection_rule": "unique_all_search_pass",
              "measurement": "exact_answer_and_boundary", "seed": seed,
              "max_new_tokens": max_new_tokens, "boundary_token_ids": list(boundary),
              "backend_fingerprint": backend.fingerprint,
              "statistical_confidence": "not_estimated",
              "compute_comparison": "not_equal_compute_certified",
              "decision": "not_started", "search": [], "held_out": None,
              "selected_candidate": None}
    winners = []
    for fmt in formats:
        report = evaluate(backend, fmt, split.search, boundary=boundary,
                          max_new_tokens=max_new_tokens, seed=seed, cancel=event)
        summary = _summary(report)
        output["search"].append({"format": asdict(fmt), "summary": summary,
                                 "observations": [asdict(o) for o in report.observations]})
        if not report.aggregate_eligible:
            output["decision"] = report.status
            return output
        if summary["exact_and_boundary_passes"] == report.planned_count:
            winners.append(fmt)
    if event.is_set():
        output["decision"] = "cancelled"
        return output
    if len(winners) != 1:
        output["decision"] = "ambiguous" if winners else "no_candidate"
        return output
    selected = winners[0]
    output["selected_candidate"] = selected.name
    held = evaluate(backend, selected, split.held_out, boundary=boundary,
                    max_new_tokens=max_new_tokens, seed=seed, cancel=event)
    output["held_out"] = {"summary": _summary(held),
                          "observations": [asdict(o) for o in held.observations]}
    if not held.aggregate_eligible:
        output["decision"] = held.status
    else:
        passed = all(o.exact_match and o.boundary_found for o in held.observations)
        output["decision"] = "compatible_on_fixture_set" if passed else "held_out_failed"
    return output
