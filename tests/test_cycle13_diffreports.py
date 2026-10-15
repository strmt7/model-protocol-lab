"""Cycle 13: `mpl diff-reports` -- deterministic comparison of two result dirs.

Covers ``model_protocol_lab/diffreports.py`` and the CLI wiring:

- decision flip and winner change (JSON + stderr summary);
- per-candidate verdict flips on search and held-out;
- per-fixture outcome diffs (changed-only default, ``--all`` for all);
- config-drift warnings (families, candidates, boundaries, max-new-tokens,
  model fingerprint, fixture sets);
- identical runs -> "no differences";
- missing evidence (no ``--include-evidence``) degrades loudly, never guesses;
- hostile/malformed ``results.json`` collapses to fixed error codes, no
  tracebacks, no hangs;
- determinism (same inputs -> byte-identical output) and JSON-serializability.

All result files are synthetic dicts; no model, no network.
"""

import json
import os
import time

import pytest

from model_protocol_lab.cli import main
from model_protocol_lab.diffreports import (
    MAX_RESULTS_BYTES,
    _config_snapshot,
    diff_reports,
    load_results,
    render_summary,
)


def _obs(fixture, ok, candidate="cand"):
    return {
        "candidate": candidate,
        "fixture": fixture,
        "input_ids": [],
        "output_ids": [],
        "answer": "a",
        "exact_match": ok,
        "boundary_found": ok,
        "boundary_text_match": ok,
        "raw_reason": None,
        "error": None,
        "status": "complete",
    }


def _summary(passes, planned):
    return {
        "status": "complete",
        "planned": planned,
        "attempted": planned,
        "aggregate_eligible": True,
        "exact_and_boundary_passes": passes,
        "rate": passes / planned if planned else None,
        "encoded_input_tokens": 0,
        "retained_output_tokens": 0,
        "failure_code": None,
    }


def _entry(name, passes, planned, outcomes=(), boundary=(198,)):
    observations = [_obs(fixture, ok, candidate=name) for fixture, ok in outcomes]
    return {
        "format": {"name": name},
        "candidate_fingerprint": "fp-" + name,
        "boundary_token_ids": list(boundary),
        "summary": _summary(passes, planned),
        "observations": observations,
    }


def _held_out(passes, planned, outcomes):
    return {
        "summary": _summary(passes, planned),
        "observations": [_obs(fixture, ok) for fixture, ok in outcomes],
    }


def _results(
    *,
    decision="compatible_on_fixture_set",
    winner="llama3",
    candidates=("llama3", "chatml"),
    passes=(2, 2),
    planned=2,
    outcomes=(("f1", True), ("f2", True)),
    held_outcomes=(("h1", True),),
    boundary=(198,),
    max_new_tokens=64,
    fingerprint="fp-backend-1",
):
    search = [
        _entry(name, p, planned, outcomes=outcomes, boundary=boundary)
        for name, p in zip(candidates, passes, strict=False)
    ]
    held = _held_out(planned, planned, held_outcomes) if winner is not None else None
    return {
        "schema_version": 1,
        "seed": 0,
        "max_new_tokens": max_new_tokens,
        "boundary_token_ids": list(boundary),
        "backend_fingerprint": fingerprint,
        "decision": decision,
        "selected_candidate": winner,
        "search": search,
        "held_out": held,
    }


def _write_results(tmp_path, name, payload):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


def _run_cli(old, new, capsys, *extra):
    code = main(["diff-reports", str(old), str(new), *extra])
    captured = capsys.readouterr()
    return code, json.loads(captured.out), captured.err


# ---------------------------------------------------------------------------
# Core comparisons
# ---------------------------------------------------------------------------


def test_decision_and_winner_flip(tmp_path, capsys):
    old = _results()
    new = _results(decision="no_candidate", winner=None, passes=(0, 0))
    code, payload, err = _run_cli(
        _write_results(tmp_path, "old", old), _write_results(tmp_path, "new", new), capsys
    )
    assert code == 0
    assert payload["decision"] == {
        "old": "compatible_on_fixture_set",
        "new": "no_candidate",
        "changed": True,
    }
    assert payload["winner"] == {"old": "llama3", "new": None, "changed": True}
    summary = payload["summary"]
    assert summary["decision_changed"] is True
    assert summary["winner_changed"] is True
    assert summary["identical"] is False
    assert "Decision: compatible_on_fixture_set -> no_candidate" in err
    assert "Winner: llama3 -> (none)" in err


def test_winner_change_same_decision():
    old = _results(winner="llama3")
    new = _results(winner="chatml")
    payload = diff_reports(old, new)
    assert payload["winner"]["changed"] is True
    assert payload["winner"]["old"] == "llama3"
    assert payload["winner"]["new"] == "chatml"
    assert payload["decision"]["changed"] is False


def test_verdict_flips_search_and_held_out():
    old = _results()
    new = _results(decision="held_out_failed", passes=(1, 0), held_outcomes=(("h1", False),))
    # make held-out summary match the failing observation count
    new["held_out"]["summary"] = _summary(0, 1)
    payload = diff_reports(old, new)
    flips = {
        (change["stage"], change["candidate"]): (change["old"], change["new"])
        for change in payload["candidate_verdicts"]
    }
    assert flips[("search", "llama3")] == ("pass", "fail")
    assert flips[("search", "chatml")] == ("pass", "fail")
    assert flips[("held_out", "llama3")] == ("pass", "fail")
    assert payload["summary"]["candidates_changed_status"] == 2


def test_absent_candidate_verdict_flip():
    old = _results(candidates=("llama3", "chatml"), passes=(2, 2))
    new = _results(candidates=("llama3",), passes=(2,))
    payload = diff_reports(old, new)
    flips = {
        (change["stage"], change["candidate"]): (change["old"], change["new"])
        for change in payload["candidate_verdicts"]
    }
    assert flips[("search", "chatml")] == ("pass", "absent")


def test_malformed_summary_verdict_unknown():
    old = _results()
    new = _results()
    new["search"][0]["summary"] = {"planned": "two"}  # hostile shape
    payload = diff_reports(old, new)
    flips = {
        (change["stage"], change["candidate"]): (change["old"], change["new"])
        for change in payload["candidate_verdicts"]
    }
    assert flips[("search", "llama3")] == ("pass", "unknown")


# ---------------------------------------------------------------------------
# Per-fixture outcome diffs
# ---------------------------------------------------------------------------


def test_fixture_diffs_changed_only_by_default():
    old = _results()
    new = _results(outcomes=(("f1", True), ("f2", False)))
    payload = diff_reports(old, new)
    changed = [entry for entry in payload["fixture_diffs"] if entry["old"] != entry["new"]]
    # both search candidates flipped on f2
    assert {(e["candidate"], e["fixture"]) for e in changed} == {
        ("llama3", "f2"),
        ("chatml", "f2"),
    }
    assert all(e["old"] == "pass" and e["new"] == "fail" for e in changed)
    assert payload["summary"]["fixtures_flipped"] == 2
    # default lists only changed fixtures
    assert payload["fixture_diffs"] == changed
    assert payload["fixture_diffs_include_unchanged"] is False


def test_all_flag_lists_unchanged_fixtures():
    old = _results()
    new = _results(outcomes=(("f1", True), ("f2", False)))
    payload = diff_reports(old, new, all_fixtures=True)
    assert payload["fixture_diffs_include_unchanged"] is True
    # 2 search candidates x 2 fixtures + 1 held-out fixture
    assert len(payload["fixture_diffs"]) == 5
    unchanged = [e for e in payload["fixture_diffs"] if e["old"] == e["new"]]
    assert len(unchanged) == 3
    # summary still counts only actual flips
    assert payload["summary"]["fixtures_flipped"] == 2


def test_fixture_diffs_deterministic_order():
    old = _results()
    new = _results(outcomes=(("f1", False), ("f2", False)))
    first = diff_reports(old, new, all_fixtures=True)
    second = diff_reports(old, new, all_fixtures=True)

    def keys(payload):
        return [(e["stage"], e["candidate"], e["fixture"]) for e in payload["fixture_diffs"]]

    assert keys(first) == sorted(keys(first))
    assert keys(first) == keys(second)


def test_malformed_observations_skipped_and_counted():
    old = _results()
    new = _results()
    new["search"][0]["observations"].append({"fixture": "fx"})  # missing flags
    new["search"][0]["observations"].append("not-a-dict")
    payload = diff_reports(old, new)
    assert payload["evidence"]["skipped_observations"] == {"old": 0, "new": 2}
    # Skipped junk never changes the verdict: identical on every compared axis.
    assert payload["summary"]["identical"] is True


# ---------------------------------------------------------------------------
# Config drift
# ---------------------------------------------------------------------------


def _drift_fields(payload):
    return {entry["field"]: entry for entry in payload["config_drift"]}


def test_identical_runs_no_differences(capsys):
    run = _results()
    payload = diff_reports(run, json.loads(json.dumps(run)))
    assert payload["summary"]["identical"] is True
    assert payload["config_drift"] == []
    assert payload["candidate_verdicts"] == []
    assert payload["fixture_diffs"] == []
    summary = render_summary(payload)
    assert "No differences: the two runs are identical" in summary


def test_identical_runs_via_cli(tmp_path, capsys):
    run = _results()
    path = _write_results(tmp_path, "run", run)
    code, payload, err = _run_cli(path, path, capsys)
    assert code == 0
    assert payload["summary"]["identical"] is True
    assert "no differences" in err.lower()


def test_drift_families_and_candidates():
    old = _results(candidates=("llama3", "chatml/mutate"), passes=(2, 2))
    new = _results(candidates=("llama3", "gemma"), passes=(2, 2))
    payload = diff_reports(old, new)
    drift = _drift_fields(payload)
    assert drift["families"]["old"] == ["chatml", "llama3"]
    assert drift["families"]["new"] == ["gemma", "llama3"]
    assert drift["candidates"]["old"] == ["chatml/mutate", "llama3"]
    assert "families" in drift and "candidates" in drift
    assert payload["summary"]["config_drift_warnings"] == len(payload["config_drift"])
    rendered = render_summary(payload)
    assert "WARNING: run configurations differ" in rendered
    assert "families" in rendered


def test_drift_boundaries_max_tokens_fingerprint():
    old = _results()
    new = _results(boundary=(198, 271), max_new_tokens=128, fingerprint="fp-backend-2")
    payload = diff_reports(old, new)
    drift = _drift_fields(payload)
    assert drift["max_new_tokens"] == {"field": "max_new_tokens", "old": 64, "new": 128}
    assert drift["backend_fingerprint"]["old"] == "fp-backend-1"
    assert drift["backend_fingerprint"]["new"] == "fp-backend-2"
    assert drift["boundaries"]["old"] == {
        "chatml": [198],
        "llama3": [198],
    }
    assert drift["boundaries"]["new"] == {
        "chatml": [198, 271],
        "llama3": [198, 271],
    }
    assert payload["summary"]["identical"] is False


def test_drift_fixture_sets():
    old = _results()
    new = _results(outcomes=(("f1", True), ("f9", True)))
    payload = diff_reports(old, new)
    drift = _drift_fields(payload)
    assert drift["fixtures:search"]["old"] == ["f1", "f2"]
    assert drift["fixtures:search"]["new"] == ["f1", "f9"]
    # differing fixture sets are flagged, not silently unioned
    assert payload["summary"]["config_drift_warnings"] >= 1


def test_drift_field_order_is_fixed():
    old = _results()
    new = _results(boundary=(1,), max_new_tokens=1, fingerprint="x", candidates=("other",))
    payload = diff_reports(old, new)
    fields = [entry["field"] for entry in payload["config_drift"]]
    expected = [
        "families",
        "candidates",
        "max_new_tokens",
        "boundaries",
        "backend_fingerprint",
    ]
    assert [f for f in fields if not f.startswith("fixtures:")] == expected


# ---------------------------------------------------------------------------
# Missing evidence
# ---------------------------------------------------------------------------


def test_missing_evidence_degrades_loudly():
    old = _results()
    new = _results()
    for entry in new["search"]:
        entry.pop("observations", None)
    new["held_out"].pop("observations", None)
    payload = diff_reports(old, new)
    assert payload["evidence"] == {
        "old": True,
        "new": False,
        "skipped_observations": {"old": 0, "new": 0},
    }
    assert payload["fixture_diffs"] == []
    # verdicts still compare (summaries are always present)
    assert payload["candidate_verdicts"] == []
    # but the runs are not "identical": the per-fixture axis was never compared
    assert payload["summary"]["identical"] is False
    rendered = render_summary(payload)
    assert "Per-fixture comparison unavailable" in rendered
    assert "new run has no --include-evidence observations" in rendered


def test_both_sides_missing_evidence():
    old = _results()
    new = _results()
    for result in (old, new):
        for entry in result["search"]:
            entry.pop("observations", None)
        result["held_out"].pop("observations", None)
    payload = diff_reports(old, new)
    assert payload["evidence"]["old"] is False
    assert payload["evidence"]["new"] is False
    assert payload["summary"]["identical"] is True  # everything comparable matches


# ---------------------------------------------------------------------------
# Hostile / malformed inputs: fixed codes, no tracebacks, no hangs
# ---------------------------------------------------------------------------


def _expect_code(tmp_path, filename, data, code):
    path = tmp_path / filename
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)
    with pytest.raises(ValueError) as excinfo:
        load_results(path)
    assert str(excinfo.value) == code


def test_hostile_inputs_fixed_codes(tmp_path):
    _expect_code(tmp_path, "a.json", "{not json", "results_invalid_json")
    _expect_code(tmp_path, "b.json", '{"decision": 1, "search": []}', "results_wrong_shape")
    _expect_code(tmp_path, "c.json", "[]", "results_not_an_object")
    _expect_code(tmp_path, "d.json", '{"decision": "x"}', "results_wrong_shape")
    _expect_code(
        tmp_path,
        "e.json",
        '{"decision": "x", "search": [], "dup": 1, "dup": 2}',
        "results_duplicate_field",
    )
    _expect_code(
        tmp_path,
        "f.json",
        '{"decision": "x", "search": [], "v": NaN}',
        "results_nonfinite_number",
    )
    _expect_code(tmp_path, "g.json", b"\xff\xfe{", "results_not_utf8")


def test_hostile_inputs_too_large(tmp_path):
    path = tmp_path / "big.json"
    with path.open("wb") as handle:
        handle.write(b'{"decision": "x", "search": [], "pad": "')
        handle.write(b"y" * (MAX_RESULTS_BYTES + 8))
        handle.write(b'"}')
    with pytest.raises(ValueError) as excinfo:
        load_results(path)
    assert str(excinfo.value) == "results_too_large"


def test_hostile_inputs_not_regular_file(tmp_path):
    fifo = tmp_path / "fifo.json"
    os.mkfifo(fifo)
    with pytest.raises(ValueError) as excinfo:
        load_results(fifo)
    assert str(excinfo.value) == "results_not_regular_file"


def test_hostile_inputs_missing(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        load_results(tmp_path / "nope.json")
    assert str(excinfo.value) == "results_path_missing"
    with pytest.raises(ValueError) as excinfo:
        load_results(tmp_path / "nope-dir")
    assert str(excinfo.value) == "results_path_missing"


def test_hostile_inputs_deep_nesting(tmp_path):
    path = tmp_path / "deep.json"
    path.write_text('{"decision": "x", "search": [' + "[" * 4000, encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_results(path)
    assert str(excinfo.value) in ("results_invalid_json", "results_nesting_too_deep")


def test_hostile_inputs_directory_resolution(tmp_path, capsys):
    # a directory resolves to results.json inside it
    run = _results()
    old = _write_results(tmp_path, "old", run)
    new = _write_results(tmp_path, "new", run)
    code, payload, _err = _run_cli(old, new, capsys)
    assert code == 0
    assert payload["summary"]["identical"] is True
    # a directory without results.json fails with a fixed code, no traceback
    empty = tmp_path / "empty"
    empty.mkdir()
    code = main(["diff-reports", str(empty), str(new)])
    captured = capsys.readouterr()
    assert code == 1
    assert "Traceback" not in captured.err
    assert "diff-reports:" in captured.err


def test_malformed_cli_reports_fixed_code_no_traceback(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    good = _write_results(tmp_path, "good", _results())
    code = main(["diff-reports", str(good), str(bad)])
    captured = capsys.readouterr()
    assert code == 1
    assert "Traceback" not in captured.err
    assert "results_invalid_json" in captured.err
    assert "(new input)" in captured.err


def test_stderr_sanitizes_hostile_identifiers(tmp_path, capsys):
    old = _results(candidates=("llama3", "chatml"), passes=(2, 2))
    new = _results(candidates=("llama3", "chatml"), passes=(2, 2))
    evil = "chatml\x1b[31mPWNED\x07"
    new["search"][1]["format"]["name"] = evil
    new["search"][1]["observations"][0]["fixture"] = "f1\x1b[2J"
    new["search"][1]["observations"][0]["exact_match"] = False
    new["search"][1]["observations"][0]["boundary_found"] = False
    payload = diff_reports(old, new)
    rendered = render_summary(payload)
    assert "\x1b" not in rendered
    assert "\x07" not in rendered
    # the sanitized identifiers are still present (readable, not dropped)
    assert "PWNED" in rendered
    # JSON output is ASCII-safe
    text = json.dumps(payload, ensure_ascii=True)
    assert "\x1b" not in text


# ---------------------------------------------------------------------------
# Determinism, serialization, CLI plumbing
# ---------------------------------------------------------------------------


def test_determinism_byte_identical(tmp_path, capsys):
    old = _results()
    new = _results(decision="no_candidate", winner=None, passes=(0, 1))
    first = json.dumps(
        diff_reports(old, new, old_label="a", new_label="b"),
        indent=2,
        ensure_ascii=True,
        sort_keys=True,
    )
    second = json.dumps(
        diff_reports(
            json.loads(json.dumps(new)), json.loads(json.dumps(old)), old_label="b", new_label="a"
        ),
        indent=2,
        ensure_ascii=True,
        sort_keys=True,
    )
    # argument order changes the content legitimately, but same order repeats
    third = json.dumps(
        diff_reports(old, new, old_label="a", new_label="b"),
        indent=2,
        ensure_ascii=True,
        sort_keys=True,
    )
    assert first == third
    assert first != second  # direction matters, deterministically


def test_payload_json_serializable():
    old = _results()
    new = _results(decision="ambiguous", winner=None)
    payload = diff_reports(old, new, all_fixtures=True)
    json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True, allow_nan=False)


def test_cli_accepts_results_json_paths_directly(tmp_path, capsys):
    old = _write_results(tmp_path, "old", _results()) / "results.json"
    new = _write_results(tmp_path, "new", _results()) / "results.json"
    code, payload, _err = _run_cli(old, new, capsys)
    assert code == 0
    assert payload["summary"]["identical"] is True


def test_cli_summary_counts_and_exit_zero_with_differences(tmp_path, capsys):
    old = _results()
    new = _results(outcomes=(("f1", False), ("f2", True)))
    code, payload, err = _run_cli(
        _write_results(tmp_path, "old", old), _write_results(tmp_path, "new", new), capsys
    )
    assert code == 0  # differences are the useful output, not a failure
    assert payload["summary"]["fixtures_flipped"] == 2
    assert "Summary:" in err
    assert "2 fixture(s) flipped" in err


def test_cli_all_flag(tmp_path, capsys):
    old = _results()
    new = _results(outcomes=(("f1", True), ("f2", False)))
    code, payload, _err = _run_cli(
        _write_results(tmp_path, "old", old),
        _write_results(tmp_path, "new", new),
        capsys,
        "--all",
    )
    assert code == 0
    assert payload["fixture_diffs_include_unchanged"] is True
    assert len(payload["fixture_diffs"]) == 5


# ---------------------------------------------------------------------------
# Red-team findings 2-5 (cycle 13): regression pins
# ---------------------------------------------------------------------------


def test_stderr_sanitizes_hostile_decision_strings():
    # Finding 2: decision values come from the hostile results.json and were
    # interpolated raw into the stderr summary (ANSI/OSC injection).
    old = _results(decision="compatible_on_fixture_set\x1b[31mPWNED\x1b[0m")
    new = _results(decision="\x1b]0;pwned\x07no_candidate")
    payload = diff_reports(old, new)
    rendered = render_summary(payload)
    assert "\x1b" not in rendered
    assert "\x07" not in rendered
    assert "PWNED" in rendered  # readable text survives; escapes do not
    assert "Decision:" in rendered


def test_stderr_sanitizes_hostile_source_labels():
    # Finding 2: source path labels were likewise raw (attacker-chosen CLI
    # paths or diff_reports labels reach stderr unfiltered).
    old = _results()
    new = _results()
    payload = diff_reports(old, new, old_label="old\x1b[2J", new_label="new\x07evil")
    rendered = render_summary(payload)
    assert "\x1b" not in rendered
    assert "\x07" not in rendered
    assert "evil" in rendered


def test_config_snapshot_dedup_scales():
    # Finding 3: the O(n^2) `name not in names` list scan hung ~1.5h on
    # 800k candidate names. 50k distinct names must complete well under
    # 10s with the set-based dedup (and stay correct).
    count = 50_000
    result = {
        "decision": "compatible_on_fixture_set",
        "search": [{"format": {"name": f"cand-{i:06d}"}} for i in range(count)],
    }
    start = time.perf_counter()
    snapshot = _config_snapshot(result)
    elapsed = time.perf_counter() - start
    assert elapsed < 10, f"_config_snapshot took {elapsed:.1f}s on {count} names"
    assert len(snapshot["candidates"]) == count
    assert snapshot["candidates"] == sorted(f"cand-{i:06d}" for i in range(count))
    assert snapshot["families"] == sorted({f"cand-{i:06d}" for i in range(count)})


def test_planned_count_drift_flags_silent_mismatch():
    # Finding 4: planned 2 vs 50 (no evidence) used to report
    # identical=True because _search_verdicts dropped the planned counts.
    # Both sides "pass" (all planned fixtures passed), so without the
    # planned comparison every other axis matches.
    old = _results(planned=2, passes=(2, 2))
    new = _results(planned=50, passes=(50, 50))
    payload = diff_reports(old, new)
    drift = _drift_fields(payload)
    assert drift["planned:llama3"] == {"field": "planned:llama3", "old": 2, "new": 50}
    assert drift["planned:chatml"] == {"field": "planned:chatml", "old": 2, "new": 50}
    assert payload["summary"]["identical"] is False
    assert payload["summary"]["config_drift_warnings"] == 2
    rendered = render_summary(payload)
    assert "planned:llama3" in rendered
    assert "2 -> 50" in rendered


def test_evidence_asymmetry_is_not_identical():
    # Finding 5: evidence on exactly one side used to report identical=True
    # with "identical on every compared axis" -- a whole axis was never
    # compared at all. The asymmetry now sinks identical and the summary
    # names the axes that were actually compared.
    old = _results()
    new = _results()
    for entry in new["search"]:
        entry.pop("observations", None)
    new["held_out"].pop("observations", None)
    payload = diff_reports(old, new)
    assert payload["summary"]["identical"] is False
    assert payload["summary"]["evidence_asymmetric"] is True
    rendered = render_summary(payload)
    assert "Per-fixture comparison unavailable" in rendered
    assert "new run has no --include-evidence observations" in rendered


def test_identical_summary_names_compared_axes():
    # Finding 5 (second half): the identical-run line names the axes that
    # were actually compared instead of asserting an unqualified totality.
    run = _results()
    rendered = render_summary(diff_reports(run, json.loads(json.dumps(run))))
    assert "No differences: the two runs are identical on every compared axis" in rendered
    assert "per-fixture evidence" in rendered  # both sides carry evidence
    assert "decision" in rendered and "configuration" in rendered
