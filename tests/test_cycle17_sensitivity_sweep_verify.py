"""Cycle 17: independent differential verification of prompt_sensitivity_sweep.

Target: ``prompt_sensitivity_sweep`` in ``model_protocol_lab/experiments.py``.

NOTE ON THE DOCSTRING: the assignment brief described this function as a
post-hoc recomputation from recorded observations. The actual docstring (and
code) says something different: for ``drop_bytes`` in 1..``max_drop_bytes``,
every prompt is truncated by that many trailing bytes and re-encoded (see
``core.truncate_text_tail``), then the FULL SEARCH RE-RUNS under a
``TruncatedPromptBackend``. The sweep is therefore a live re-execution
sensitivity probe, not a post-hoc derivation. This oracle is built from the
actual docstring definition: for each truncation depth it literally rebuilds
the truncated prompt inputs (render -> drop trailing bytes at whole-UTF-8-
character granularity -> re-encode) and re-derives the decision from first
principles -- per-candidate per-fixture pass/fail on the truncated records,
then the conservative unique-winner rule (exactly one candidate passes every
search fixture -> selected; none -> no_candidate; multiple -> ambiguous), with
the held-out check for the unique winner. The oracle never calls
``run_experiment``, ``evaluate``, ``TruncatedPromptBackend`` or
``truncate_text_tail``: truncation, the boundary scan and the selection rule
are all reimplemented naively below.

Fields compared against the oracle: the sweep points (drop_bytes 1..max),
each depth's ``decision`` / ``selected_candidate`` / ``winner_changed``, the
``flips_at`` list, and the native ``decision`` / ``winner`` summary.

A deterministic scripted backend makes ``generate`` a pure function of the
(prompt) text, so the oracle and the implementation necessarily observe the
same outputs for the same truncated inputs: any divergence is a real bug in
the sweep logic, not backend nondeterminism.
"""

import importlib
import os
import random
import shutil
import sys
import tempfile
import unittest
from threading import Event
from typing import Any

from model_protocol_lab import core as real_core
from model_protocol_lab.experiments import prompt_sensitivity_sweep

# Token IDs from the Unicode private-use area: encode() maps every character to
# its codepoint, so these IDs can never collide with prompt text. The
# scripted backend emits them as the response-boundary marker.
_BOUNDARY_A = 0xE000
_BOUNDARY_B = 0xE001
_CAP = 8  # max_new_tokens for every run; keeps outputs tiny and valid
_PAD_ID = ord("~")  # padding token, never equal to a boundary marker
_SEED = 1234


class ScriptedBackend:
    """Deterministic fake backend: generate() is scripted per prompt text.

    ``script`` maps a prompt string to ``(answer_text, marker_text)``; the
    output is ``answer_text + marker_text`` padded to the token cap with a
    reason of ``"limit"`` (a valid ``Generation``). ``marker_text`` is the
    decoded boundary (``""`` when the scripted output has no boundary).
    Prompts absent from the script fail closed with ``default``.
    """

    def __init__(
        self,
        core_mod: Any,
        script: dict[str, tuple[str, str]],
        default: tuple[str, str] = ("WRONG", ""),
    ) -> None:
        self._core = core_mod
        self._script = dict(script)
        self._default = default
        self.fingerprint = "cycle17-scripted-backend"

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(ord(char) for char in text)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(chr(token) for token in token_ids)

    def generate(
        self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> Any:
        prompt = "".join(chr(token) for token in input_ids)
        answer, marker = self._script.get(prompt, self._default)
        ids = tuple(ord(char) for char in answer + marker)
        if len(ids) > max_new_tokens:
            raise ValueError("scripted output exceeds the token cap")
        padding = (_PAD_ID,) * (max_new_tokens - len(ids))
        return self._core.Generation(token_ids=ids + padding, reason="limit")


# ---------------------------------------------------------------------------
# Naive oracle, from the docstring definition only.
# ---------------------------------------------------------------------------


def _naive_truncate_tail(text: str, drop_bytes: int) -> str | None:
    """Drop trailing bytes; None when the cut would split a UTF-8 codepoint."""
    raw = text.encode("utf-8")
    if not 1 <= drop_bytes < len(raw):
        raise ValueError("drop_bytes must lie within the text's byte length")
    try:
        return raw[:-drop_bytes].decode("utf-8")
    except UnicodeDecodeError:
        return None


def _naive_first_boundary(tokens: tuple[int, ...], stop: tuple[int, ...]) -> int | None:
    for index in range(len(tokens) - len(stop) + 1):
        if tokens[index : index + len(stop)] == stop:
            return index
    return None


def _naive_candidate_boundary(
    name: str, boundary: tuple[int, ...], boundary_map: dict[str, tuple[int, ...]] | None
) -> tuple[int, ...]:
    if boundary_map is None:
        return boundary
    return boundary_map.get(name, boundary)


def _naive_run_decision(
    backend: ScriptedBackend,
    formats: tuple[Any, ...],
    search: tuple[Any, ...],
    held_out: tuple[Any, ...],
    drop_bytes: int,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None,
    max_new_tokens: int,
) -> tuple[str, str | None]:
    """Re-derive (decision, selected_candidate) from first principles.

    drop_bytes == 0 is the native run. A truncation that cannot be applied
    (codepoint split, or drop beyond the prompt's byte length) fails the run
    with the failed decision, mirroring the sweep docstring.
    """

    def truncated_prompt(fmt: Any, case: Any) -> str | None:
        prompt = fmt.render(case.messages)
        if drop_bytes == 0:
            return prompt
        try:
            return _naive_truncate_tail(prompt, drop_bytes)
        except ValueError:
            return None

    def fixture_passes(fmt: Any, case: Any) -> bool | None:
        """True/False per fixture; None when the prompt cannot be truncated."""
        prompt = truncated_prompt(fmt, case)
        if prompt is None:
            return None
        candidate_boundary = _naive_candidate_boundary(fmt.name, boundary, boundary_map)
        output = backend.generate(
            backend.encode(prompt), max_new_tokens=max_new_tokens, seed=0, cancel=Event()
        )
        cut = _naive_first_boundary(output.token_ids, candidate_boundary)
        answer_ids = output.token_ids if cut is None else output.token_ids[:cut]
        return cut is not None and backend.decode(answer_ids) == case.expected

    # NOTE: no early break on a failed fixture. evaluate() records a wrong
    # answer as a completed (failed) observation and CONTINUES: a truncation
    # error at a later fixture still aborts the run with the failed decision.
    # Breaking early would miss that error and misreport e.g. no_candidate.
    winners = []
    for fmt in formats:
        verdicts = []
        for case in search:
            verdict = fixture_passes(fmt, case)
            if verdict is None:
                return "failed", None
            verdicts.append(verdict)
        if all(verdicts):
            winners.append(fmt)
    if not winners:
        return "no_candidate", None
    if len(winners) > 1:
        return "ambiguous", None
    selected = winners[0]
    held_verdicts = []
    for case in held_out:
        verdict = fixture_passes(selected, case)
        if verdict is None:
            return "failed", selected.name
        held_verdicts.append(verdict)
    if all(held_verdicts):
        return "compatible_on_fixture_set", selected.name
    return "held_out_failed", selected.name


def _oracle_sweep(
    backend: ScriptedBackend,
    formats: tuple[Any, ...],
    split: Any,
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None,
    max_drop_bytes: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    native_decision, native_winner = _naive_run_decision(
        backend,
        formats,
        split.search,
        split.held_out,
        0,
        boundary,
        boundary_map,
        max_new_tokens,
    )
    depths = []
    for drop in range(1, max_drop_bytes + 1):
        decision, winner = _naive_run_decision(
            backend,
            formats,
            split.search,
            split.held_out,
            drop,
            boundary,
            boundary_map,
            max_new_tokens,
        )
        changed = winner != native_winner or decision != native_decision
        depths.append(
            {
                "drop_bytes": drop,
                "decision": decision,
                "selected_candidate": winner,
                "winner_changed": changed,
            }
        )
    return {
        "native_decision": native_decision,
        "native_winner": native_winner,
        "max_drop_bytes": max_drop_bytes,
        "depths": depths,
        "flips_at": [depth["drop_bytes"] for depth in depths if depth["winner_changed"]],
    }


def check_sweep_against_oracle(
    sweep_fn: Any,
    backend: ScriptedBackend,
    formats: tuple[Any, ...],
    split: Any,
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None,
    max_drop_bytes: int,
    max_new_tokens: int,
    seed: int,
) -> None:
    """Assert every reported sweep field matches the naive oracle."""
    actual = sweep_fn(
        backend,
        formats,
        split,
        boundary=boundary,
        boundary_map=boundary_map,
        max_drop_bytes=max_drop_bytes,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    expected = _oracle_sweep(
        backend,
        formats,
        split,
        boundary=boundary,
        boundary_map=boundary_map,
        max_drop_bytes=max_drop_bytes,
        max_new_tokens=max_new_tokens,
    )
    assert actual["native_decision"] == expected["native_decision"], (
        f"native_decision: {actual['native_decision']!r} != {expected['native_decision']!r}"
    )
    assert actual["native_winner"] == expected["native_winner"], (
        f"native_winner: {actual['native_winner']!r} != {expected['native_winner']!r}"
    )
    assert actual["max_drop_bytes"] == expected["max_drop_bytes"]
    assert [d["drop_bytes"] for d in actual["depths"]] == [
        d["drop_bytes"] for d in expected["depths"]
    ], f"sweep points: {[d['drop_bytes'] for d in actual['depths']]}"
    assert len(actual["depths"]) == len(expected["depths"])
    drops = [d["drop_bytes"] for d in expected["depths"]]
    for drop, got, want in zip(drops, actual["depths"], expected["depths"], strict=True):
        assert got["decision"] == want["decision"], (
            f"depth {drop}: decision {got['decision']!r} != {want['decision']!r}"
        )
        assert got["selected_candidate"] == want["selected_candidate"], (
            f"depth {drop}: winner {got['selected_candidate']!r} != {want['selected_candidate']!r}"
        )
        assert got["winner_changed"] == want["winner_changed"], (
            f"depth {drop}: winner_changed {got['winner_changed']!r} != {want['winner_changed']!r}"
        )
    assert actual["flips_at"] == expected["flips_at"], (
        f"flips_at: {actual['flips_at']!r} != {expected['flips_at']!r}"
    )


# ---------------------------------------------------------------------------
# Scenario builders.
# ---------------------------------------------------------------------------

_CONTENT_POOL = (
    "",
    "x",
    "hi",
    "what is 2+2?",
    "héllo wörld",
    "end emoji 🎉",
    "café ☕",
    "日本語",
    "a",
    "ab",
    "some longer prompt text with spaces",
    "z",
    "q?",
)
_EXPECTED_POOL = ("yes", "no", "42", "a", "ok")
_WRONG_POOL = ("WRONG", "nope", "0", "maybe", "")


def _build_fuzz_scenario(
    rng: random.Random, core_mod: Any
) -> tuple[ScriptedBackend, tuple[Any, ...], Any, dict[str, Any]]:
    n_candidates = rng.randint(2, 5)
    n_search = rng.randint(1, 6)
    n_held = rng.randint(1, 3)
    max_drop_bytes = rng.randint(1, 4)
    use_boundary_map = rng.random() < 0.5

    formats = tuple(
        core_mod.Format(
            name=f"cand-{index}",
            bos=rng.choice(("", "<s>", "BOS:")),
            user_prefix=rng.choice(("U:", "[user]", ">", "")),
            assistant_prefix=rng.choice(("A:", "[assistant]", "")),
            turn_end=rng.choice(("", "\n", "<|end|>")),
            repeat_bos=rng.random() < 0.3,
        )
        for index in range(n_candidates)
    )

    def make_fixture(prefix: str, index: int) -> Any:
        content = rng.choice(_CONTENT_POOL)
        if prefix == "h":
            # Split forbids identical message inputs crossing the split.
            content = f"held: {content}"
        return core_mod.Fixture(
            id=f"{prefix}{index}",
            messages=(core_mod.Message(role="user", content=content),),
            expected=rng.choice(_EXPECTED_POOL),
            group=f"cycle17-{prefix}",
        )

    search = tuple(make_fixture("s", i) for i in range(n_search))
    held_out = tuple(make_fixture("h", i) for i in range(n_held))
    split = core_mod.Split(search=search, held_out=held_out)

    boundary = (_BOUNDARY_A,)
    boundary_map = None
    if use_boundary_map:
        boundary_map = {formats[0].name: (_BOUNDARY_B,)}

    script: dict[str, tuple[str, str]] = {}
    for fmt in formats:
        candidate_marker = chr(_naive_candidate_boundary(fmt.name, boundary, boundary_map)[0])
        for case in search + held_out:
            prompt = fmt.render(case.messages)
            # Some fixtures pass only because of late tokens: full prompt passes,
            # every truncation fails.
            late_token_only = rng.random() < 0.15
            for drop in range(0, max_drop_bytes + 1):
                if drop:
                    try:
                        key = _naive_truncate_tail(prompt, drop)
                    except ValueError:
                        key = None
                else:
                    key = prompt
                if key is None:
                    # Un-truncatable: encode raises in the implementation, so the
                    # script entry is never consulted; the oracle never consults
                    # it either.
                    continue
                if late_token_only:
                    passes = drop == 0
                else:
                    passes = rng.random() < (0.55 if drop == 0 else 0.25)
                if passes:
                    script[key] = (case.expected, candidate_marker)
                else:
                    mode = rng.random()
                    if mode < 0.4:
                        script[key] = (rng.choice(_WRONG_POOL), candidate_marker)
                    elif mode < 0.7:
                        script[key] = (case.expected, "")
                    else:
                        script[key] = (rng.choice(_WRONG_POOL), "")
    backend = ScriptedBackend(core_mod, script)
    kwargs = {
        "boundary": boundary,
        "boundary_map": boundary_map,
        "max_drop_bytes": max_drop_bytes,
        "max_new_tokens": _CAP,
        "seed": _SEED,
    }
    return backend, formats, split, kwargs


def _tiny_formats(core_mod: Any) -> tuple[Any, ...]:
    # Single-byte prefixes/suffixes so truncations of 1..2 bytes stay at
    # whole-character granularity (multibyte truncation is covered by the
    # fuzz pool and the codepoint-split edge test).
    return (
        core_mod.Format(
            name="cand-a",
            bos="",
            user_prefix=">",
            assistant_prefix="<",
            turn_end="",
            repeat_bos=False,
        ),
        core_mod.Format(
            name="cand-b",
            bos="",
            user_prefix="!",
            assistant_prefix="?",
            turn_end="",
            repeat_bos=False,
        ),
    )


def _script_all_drops(
    script: dict[str, tuple[str, str]],
    fmt: Any,
    case: Any,
    expected: str,
    max_drop: int,
    marker: str,
) -> None:
    """Script the full prompt and every truncation 1..max_drop as passing."""
    full = fmt.render(case.messages)
    script[full] = (expected, marker)
    for drop in range(1, max_drop + 1):
        try:
            truncated = _naive_truncate_tail(full, drop)
        except ValueError:
            continue
        if truncated is None:
            continue
        script[truncated] = (expected, marker)


def _tiny_fixture(core_mod: Any, fixture_id: str, content: str, expected: str) -> Any:
    group = "cycle17-search" if fixture_id.startswith("s") else "cycle17-held"
    return core_mod.Fixture(
        id=fixture_id,
        messages=(core_mod.Message(role="user", content=content),),
        expected=expected,
        group=group,
    )


class SweepOracleFuzzTests(unittest.TestCase):
    """Seeded differential fuzzing: implementation vs naive oracle."""

    TRIALS = 60

    def test_sweep_matches_oracle(self) -> None:
        for trial in range(self.TRIALS):
            with self.subTest(trial=trial):
                rng = random.Random(0xC170 + trial)  # noqa: S311 -- deterministic test seed
                backend, formats, split, kwargs = _build_fuzz_scenario(rng, real_core)
                check_sweep_against_oracle(
                    prompt_sensitivity_sweep, backend, formats, split, **kwargs
                )


class SweepEdgeCaseTests(unittest.TestCase):
    """Hand-built edge cases: truncation semantics, flip semantics, validation."""

    def _run(
        self,
        core_mod: Any,
        backend: ScriptedBackend,
        formats: tuple[Any, ...],
        split: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return prompt_sensitivity_sweep(backend, formats, split, **kwargs)

    def test_codepoint_split_depth_records_failed_decision(self) -> None:
        # "🎉" is 4 UTF-8 bytes and prompt-final here (empty assistant_prefix):
        # drops 1..3 split the codepoint (failed depth), drop 4 truncates cleanly.
        core_mod = real_core
        formats = (
            core_mod.Format(
                name="cand-a",
                bos="",
                user_prefix=">",
                assistant_prefix="",
                turn_end="",
                repeat_bos=False,
            ),
            core_mod.Format(
                name="cand-b",
                bos="",
                user_prefix="»",
                assistant_prefix="",
                turn_end="",
                repeat_bos=False,
            ),
        )
        search = (_tiny_fixture(core_mod, "s0", "end emoji 🎉", "yes"),)
        held_out = (_tiny_fixture(core_mod, "h0", "plain", "no"),)
        split = core_mod.Split(search=search, held_out=held_out)
        script = {}
        for fmt in formats:
            full = fmt.render(search[0].messages)
            script[full] = ("yes", chr(_BOUNDARY_A)) if fmt.name == "cand-a" else ("WRONG", "")
            clean = _naive_truncate_tail(full, 4)
            assert clean is not None
            script[clean] = ("yes", chr(_BOUNDARY_A)) if fmt.name == "cand-a" else ("WRONG", "")
            held_full = fmt.render(held_out[0].messages)
            script[held_full] = ("no", chr(_BOUNDARY_A))
            held_clean = _naive_truncate_tail(held_full, 4)
            assert held_clean is not None
            script[held_clean] = ("no", chr(_BOUNDARY_A))
        backend = ScriptedBackend(core_mod, script)
        kwargs = {
            "boundary": (_BOUNDARY_A,),
            "boundary_map": None,
            "max_drop_bytes": 4,
            "max_new_tokens": _CAP,
            "seed": _SEED,
        }
        check_sweep_against_oracle(prompt_sensitivity_sweep, backend, formats, split, **kwargs)
        result = self._run(core_mod, backend, formats, split, **kwargs)
        self.assertEqual(result["native_decision"], "compatible_on_fixture_set")
        self.assertEqual(result["native_winner"], "cand-a")
        for depth in result["depths"][:3]:
            self.assertEqual(depth["decision"], "failed")
            self.assertIsNone(depth["selected_candidate"])
            self.assertTrue(depth["winner_changed"])
        self.assertEqual(result["depths"][3]["drop_bytes"], 4)
        self.assertEqual(result["depths"][3]["decision"], "compatible_on_fixture_set")
        self.assertEqual(result["depths"][3]["selected_candidate"], "cand-a")
        self.assertFalse(result["depths"][3]["winner_changed"])
        self.assertEqual(result["flips_at"], [1, 2, 3])

    def test_drop_beyond_prompt_length_records_failed_decision(self) -> None:
        # ">" is a single byte and prompt-final here (empty content and empty
        # assistant_prefix): every drop in 1..3 exceeds the prompt length.
        core_mod = real_core
        formats = (
            core_mod.Format(
                name="cand-a",
                bos="",
                user_prefix=">",
                assistant_prefix="",
                turn_end="",
                repeat_bos=False,
            ),
            core_mod.Format(
                name="cand-b",
                bos="",
                user_prefix="!",
                assistant_prefix="",
                turn_end="",
                repeat_bos=False,
            ),
        )
        search = (_tiny_fixture(core_mod, "s0", "", "yes"),)
        held_out = (_tiny_fixture(core_mod, "h0", "h", "no"),)
        split = core_mod.Split(search=search, held_out=held_out)
        script = {}
        for fmt in formats:
            full = fmt.render(search[0].messages)
            self.assertEqual(len(full.encode("utf-8")), 1)
            script[full] = ("yes", chr(_BOUNDARY_A)) if fmt.name == "cand-a" else ("WRONG", "")
            script[fmt.render(held_out[0].messages)] = ("no", chr(_BOUNDARY_A))
        backend = ScriptedBackend(core_mod, script)
        kwargs = {
            "boundary": (_BOUNDARY_A,),
            "boundary_map": None,
            "max_drop_bytes": 3,
            "max_new_tokens": _CAP,
            "seed": _SEED,
        }
        check_sweep_against_oracle(prompt_sensitivity_sweep, backend, formats, split, **kwargs)
        result = self._run(core_mod, backend, formats, split, **kwargs)
        self.assertTrue(all(d["decision"] == "failed" for d in result["depths"]))
        self.assertEqual(result["flips_at"], [1, 2, 3])

    def test_decision_flip_with_same_winner_counts(self) -> None:
        # The winner is unchanged at depth 1 but the held-out check flips from
        # compatible to failed: winner_changed must still be True because the
        # decision differs.
        core_mod = real_core
        formats = _tiny_formats(core_mod)
        search = (_tiny_fixture(core_mod, "s0", "question", "yes"),)
        held_out = (_tiny_fixture(core_mod, "h0", "holdout", "no"),)
        split = core_mod.Split(search=search, held_out=held_out)
        script = {}
        for fmt in formats:
            search_full = fmt.render(search[0].messages)
            held_full = fmt.render(held_out[0].messages)
            if fmt.name == "cand-a":
                script[search_full] = ("yes", chr(_BOUNDARY_A))
                script[held_full] = ("no", chr(_BOUNDARY_A))
                truncated = _naive_truncate_tail(search_full, 1)
                assert truncated is not None
                script[truncated] = ("yes", chr(_BOUNDARY_A))
                held_truncated = _naive_truncate_tail(held_full, 1)
                assert held_truncated is not None
                script[held_truncated] = ("WRONG", chr(_BOUNDARY_A))
            else:
                script[search_full] = ("WRONG", "")
                script[held_full] = ("WRONG", "")
        backend = ScriptedBackend(core_mod, script)
        kwargs = {
            "boundary": (_BOUNDARY_A,),
            "boundary_map": None,
            "max_drop_bytes": 1,
            "max_new_tokens": _CAP,
            "seed": _SEED,
        }
        check_sweep_against_oracle(prompt_sensitivity_sweep, backend, formats, split, **kwargs)
        result = self._run(core_mod, backend, formats, split, **kwargs)
        depth = result["depths"][0]
        self.assertEqual(depth["selected_candidate"], "cand-a")
        self.assertEqual(result["native_decision"], "compatible_on_fixture_set")
        self.assertEqual(depth["decision"], "held_out_failed")
        self.assertTrue(depth["winner_changed"])
        self.assertEqual(result["flips_at"], [1])

    def test_no_flip_gives_empty_flips_at(self) -> None:
        core_mod = real_core
        formats = _tiny_formats(core_mod)
        search = (_tiny_fixture(core_mod, "s0", "question", "yes"),)
        held_out = (_tiny_fixture(core_mod, "h0", "holdout", "no"),)
        split = core_mod.Split(search=search, held_out=held_out)
        script = {}
        for fmt in formats:
            if fmt.name == "cand-a":
                _script_all_drops(script, fmt, search[0], "yes", 2, chr(_BOUNDARY_A))
                _script_all_drops(script, fmt, held_out[0], "no", 2, chr(_BOUNDARY_A))
        # cand-b never passes: it is absent from the script and fails closed.
        backend = ScriptedBackend(core_mod, script)
        kwargs = {
            "boundary": (_BOUNDARY_A,),
            "boundary_map": None,
            "max_drop_bytes": 2,
            "max_new_tokens": _CAP,
            "seed": _SEED,
        }
        check_sweep_against_oracle(prompt_sensitivity_sweep, backend, formats, split, **kwargs)
        result = self._run(core_mod, backend, formats, split, **kwargs)
        self.assertEqual(result["flips_at"], [])
        self.assertTrue(all(not d["winner_changed"] for d in result["depths"]))

    def test_multi_token_boundary(self) -> None:
        # A two-token boundary exercises the boundary scan at every depth.
        core_mod = real_core
        boundary = (_BOUNDARY_A, 0xE002)
        marker = chr(_BOUNDARY_A) + chr(0xE002)
        formats = _tiny_formats(core_mod)
        search = (_tiny_fixture(core_mod, "s0", "question", "yes"),)
        held_out = (_tiny_fixture(core_mod, "h0", "holdout", "no"),)
        split = core_mod.Split(search=search, held_out=held_out)
        script: dict[str, tuple[str, str]] = {}
        for fmt in formats:
            _script_all_drops(script, fmt, search[0], "yes", 2, marker)
            _script_all_drops(script, fmt, held_out[0], "no", 2, marker)
        # At depth 1 cand-b stops passing search: cand-a becomes the unique
        # winner instead of the native ambiguous tie.
        truncated = _naive_truncate_tail(formats[1].render(search[0].messages), 1)
        assert truncated is not None
        script[truncated] = ("WRONG", marker)
        backend = ScriptedBackend(core_mod, script)
        kwargs = {
            "boundary": boundary,
            "boundary_map": None,
            "max_drop_bytes": 2,
            "max_new_tokens": _CAP,
            "seed": _SEED,
        }
        check_sweep_against_oracle(prompt_sensitivity_sweep, backend, formats, split, **kwargs)
        result = self._run(core_mod, backend, formats, split, **kwargs)
        self.assertEqual(result["native_decision"], "ambiguous")
        self.assertIsNone(result["native_winner"])
        depth1 = result["depths"][0]
        self.assertEqual(depth1["decision"], "compatible_on_fixture_set")
        self.assertEqual(depth1["selected_candidate"], "cand-a")
        self.assertTrue(depth1["winner_changed"])
        self.assertFalse(result["depths"][1]["winner_changed"])
        self.assertEqual(result["flips_at"], [1])

    def test_sweep_points_are_exactly_one_through_max(self) -> None:
        core_mod = real_core
        formats = _tiny_formats(core_mod)
        search = (_tiny_fixture(core_mod, "s0", "q", "yes"),)
        held_out = (_tiny_fixture(core_mod, "h0", "hq", "no"),)
        split = core_mod.Split(search=search, held_out=held_out)
        backend = ScriptedBackend(core_mod, {})
        result = prompt_sensitivity_sweep(
            backend,
            formats,
            split,
            boundary=(_BOUNDARY_A,),
            max_drop_bytes=5,
            max_new_tokens=_CAP,
            seed=_SEED,
        )
        self.assertEqual([d["drop_bytes"] for d in result["depths"]], [1, 2, 3, 4, 5])

    def test_deterministic_across_runs(self) -> None:
        rng = random.Random(0xC171)  # noqa: S311 -- deterministic test seed
        backend, formats, split, kwargs = _build_fuzz_scenario(rng, real_core)
        first = prompt_sensitivity_sweep(backend, formats, split, **kwargs)
        second = prompt_sensitivity_sweep(backend, formats, split, **kwargs)
        self.assertEqual(first, second)

    def test_max_drop_bytes_validation(self) -> None:
        core_mod = real_core
        formats = _tiny_formats(core_mod)
        split = core_mod.Split(
            search=(_tiny_fixture(core_mod, "s0", "q", "yes"),),
            held_out=(_tiny_fixture(core_mod, "h0", "hq", "no"),),
        )
        backend = ScriptedBackend(core_mod, {})
        for bad in (0, 9, -1, "2", 2.5, None):
            with self.subTest(max_drop_bytes=bad), self.assertRaises(ValueError):
                prompt_sensitivity_sweep(
                    backend,
                    formats,
                    split,
                    boundary=(_BOUNDARY_A,),
                    max_drop_bytes=bad,
                    max_new_tokens=_CAP,
                    seed=_SEED,
                )


class SweepSensitivityTests(unittest.TestCase):
    """Prove the oracle harness catches mutations of the implementation.

    Each test copies the whole ``model_protocol_lab`` package to a temp dir
    under a unique name (all intra-package imports are relative, so the copy
    imports cleanly), applies a textual mutation to the copy's
    ``experiments.py`` -- with an exact-occurrence guard so a drifted source
    fails loudly instead of silently not mutating -- and asserts the oracle
    checker raises AssertionError on the mutated sweep.
    """

    def _mutated(self, tag: str, patches: list[tuple[str, str, int]]) -> tuple[Any, Any]:
        pkg_dir = os.path.dirname(os.path.abspath(real_core.__file__))
        tmp = tempfile.mkdtemp(prefix="c17sens_")
        self.addCleanup(shutil.rmtree, tmp, True)
        pkg_name = f"c17sens_{tag}"
        shutil.copytree(pkg_dir, os.path.join(tmp, pkg_name))
        exp_path = os.path.join(tmp, pkg_name, "experiments.py")
        with open(exp_path, encoding="utf-8") as handle:
            src = handle.read()
        for old, new, expected_count in patches:
            actual = src.count(old)
            assert actual == expected_count, (tag, old, actual)
            src = src.replace(old, new)
        with open(exp_path, "w", encoding="utf-8") as handle:
            handle.write(src)
        sys.path.insert(0, tmp)
        self.addCleanup(sys.path.remove, tmp)
        experiments_mod = importlib.import_module(f"{pkg_name}.experiments")
        core_mod = importlib.import_module(f"{pkg_name}.core")
        return experiments_mod, core_mod

    def _flip_scenario(self, core_mod: Any) -> tuple[ScriptedBackend, tuple, Any, dict]:
        """Two candidates, one fixture: native selects cand-a, depth 1 flips."""
        formats = _tiny_formats(core_mod)
        search = (_tiny_fixture(core_mod, "s0", "question", "yes"),)
        held_out = (_tiny_fixture(core_mod, "h0", "holdout", "no"),)
        split = core_mod.Split(search=search, held_out=held_out)
        script: dict[str, tuple[str, str]] = {}
        for fmt in formats:
            search_full = fmt.render(search[0].messages)
            held_full = fmt.render(held_out[0].messages)
            search_trunc = _naive_truncate_tail(search_full, 1)
            held_trunc = _naive_truncate_tail(held_full, 1)
            assert search_trunc is not None and held_trunc is not None
            if fmt.name == "cand-a":
                script[search_full] = ("yes", chr(_BOUNDARY_A))
                script[held_full] = ("no", chr(_BOUNDARY_A))
                script[search_trunc] = ("WRONG", "")
                script[held_trunc] = ("no", chr(_BOUNDARY_A))
            else:
                script[search_full] = ("WRONG", "")
                script[held_full] = ("WRONG", "")
                script[search_trunc] = ("yes", chr(_BOUNDARY_A))
                script[held_trunc] = ("no", chr(_BOUNDARY_A))
        backend = ScriptedBackend(core_mod, script)
        kwargs = {
            "boundary": (_BOUNDARY_A,),
            "boundary_map": None,
            "max_drop_bytes": 2,
            "max_new_tokens": _CAP,
            "seed": _SEED,
        }
        return backend, formats, split, kwargs

    def _assert_mutation_caught(self, tag: str, patches: list[tuple[str, str, int]]) -> None:
        experiments_mod, core_mod = self._mutated(tag, patches)
        backend, formats, split, kwargs = self._flip_scenario(core_mod)
        with self.assertRaises(AssertionError):
            check_sweep_against_oracle(
                experiments_mod.prompt_sensitivity_sweep, backend, formats, split, **kwargs
            )

    def test_unmutated_copy_passes_oracle(self) -> None:
        experiments_mod, core_mod = self._mutated("control", [])
        backend, formats, split, kwargs = self._flip_scenario(core_mod)
        check_sweep_against_oracle(
            experiments_mod.prompt_sensitivity_sweep, backend, formats, split, **kwargs
        )

    def test_off_by_one_sweep_range_is_caught(self) -> None:
        self._assert_mutation_caught(
            "range",
            [
                (
                    "for drop in range(1, max_drop_bytes + 1):",
                    "for drop in range(2, max_drop_bytes + 2):",
                    1,
                )
            ],
        )

    _FLIP_LINE = 'changed = winner != native_winner or mutated["decision"] != native["decision"]'
    _FLIP_INVERTED = (
        'changed = winner == native_winner and mutated["decision"] == native["decision"]'
    )

    def test_inverted_flip_detection_is_caught(self) -> None:
        self._assert_mutation_caught("invert", [(self._FLIP_LINE, self._FLIP_INVERTED, 1)])

    def test_decision_ignored_in_flip_is_caught(self) -> None:
        # A mutant that only compares winners misses decision-only flips (same
        # winner, held-out verdict changed). Build that scenario explicitly.
        experiments_mod, core_mod = self._mutated(
            "nowinner",
            [(self._FLIP_LINE, "changed = winner != native_winner", 1)],
        )
        formats = _tiny_formats(core_mod)
        search = (_tiny_fixture(core_mod, "s0", "question", "yes"),)
        held_out = (_tiny_fixture(core_mod, "h0", "holdout", "no"),)
        split = core_mod.Split(search=search, held_out=held_out)
        script: dict[str, tuple[str, str]] = {}
        for fmt in formats:
            search_full = fmt.render(search[0].messages)
            held_full = fmt.render(held_out[0].messages)
            search_trunc = _naive_truncate_tail(search_full, 1)
            held_trunc = _naive_truncate_tail(held_full, 1)
            assert search_trunc is not None and held_trunc is not None
            if fmt.name == "cand-a":
                script[search_full] = ("yes", chr(_BOUNDARY_A))
                script[held_full] = ("no", chr(_BOUNDARY_A))
                script[search_trunc] = ("yes", chr(_BOUNDARY_A))
                script[held_trunc] = ("WRONG", chr(_BOUNDARY_A))
            else:
                script[search_full] = ("WRONG", "")
                script[held_full] = ("WRONG", "")
        backend = ScriptedBackend(core_mod, script)
        kwargs = {
            "boundary": (_BOUNDARY_A,),
            "boundary_map": None,
            "max_drop_bytes": 1,
            "max_new_tokens": _CAP,
            "seed": _SEED,
        }
        with self.assertRaises(AssertionError):
            check_sweep_against_oracle(
                experiments_mod.prompt_sensitivity_sweep, backend, formats, split, **kwargs
            )


if __name__ == "__main__":
    unittest.main()
