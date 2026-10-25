"""Cycle 206: token-depth sensitivity sweep (``run --token-depth-sweep``).

The instrument (docs/TOKENDEPTH_DESIGN.md, adjudicated in
~/workspace/agents_out/cycle206/coordinator_adjudication.md): the shipped
byte sweep can only mangle the tail of a multi-token turn-closing suffix
(byte cap 8), while this ladder drops whole trailing tokens. These tests
pin the adjudication's load-bearing points:

- the token wrapper drops token ids (never a re-encode), carries the
  distinct ``+prompt-trunc-{K}t`` fingerprint, fails closed beyond the
  prompt length, and refuses a prefill-wrapped inner (B1: a naive
  ``[:-K]`` over a PrefillBackend drops prefill ids first);
- the sweep payload carries per-depth ``status`` and gates
  ``winner_changed`` on ``status == "applied"`` (corrected A1), so a
  depth that never ran is listed in ``inapplicable_at`` and never in
  ``flips_at`` -- the byte sweep's failed-depth-counts-as-flip flaw
  (experiments.py:3509/:3523, pinned by test_cycle17) is fixed here
  only, byte mode untouched (P3);
- inapplicability is decided by a pre-run token-length check, never
  inferred post-hoc from a failed decision (a genuinely applied depth
  whose run fails still counts when its decision differs);
- byte and token sweeps share one native ``run_experiment`` when both
  are requested (saving exactly one, not two);
- the report gains a separate Token-depth section (three touchpoints)
  while the byte section's output is unchanged by the token key.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest import mock

from model_protocol_lab import backends, experiments, reporting
from model_protocol_lab import cli as cli_module
from model_protocol_lab.backends import TokenTruncatedPromptBackend
from model_protocol_lab.cli import build_parser, main
from model_protocol_lab.core import Fixture, Format, Generation, Message, Split
from model_protocol_lab.experiments import (
    prompt_sensitivity_sweep,
    prompt_token_depth_sweep,
    run_experiment,
)
from model_protocol_lab.prefill import PrefillBackend

_BOUNDARY = 0xE000
_CAP = 8


class ScriptedBackend:
    """Char-token backend: encode maps chars to codepoints, generate
    looks the decoded prompt up in a script (default WRONG, no marker).
    """

    def __init__(self, script: dict[str, tuple[str, str]]) -> None:
        self._script = dict(script)
        self.fingerprint = "cycle206-scripted-backend"

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(ord(char) for char in text)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(chr(token) for token in token_ids)

    def generate(
        self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> Generation:
        prompt = "".join(chr(token) for token in input_ids)
        answer, marker = self._script.get(prompt, ("WRONG", ""))
        text = answer + marker
        ids = tuple(ord(char) for char in text)
        if len(ids) > max_new_tokens:
            raise ValueError("scripted output exceeds the token cap")
        padding = (ord("~"),) * (max_new_tokens - len(ids))
        return Generation(token_ids=ids + padding, reason="limit")


_FMT_A = Format(name="cand-a", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>")
_FMT_B = Format(name="cand-b", bos="", user_prefix="U>", assistant_prefix="A>", turn_end="<e>")
_SEARCH = (
    Fixture(
        id="s0",
        messages=(Message(role="user", content="hi"),),
        expected="yes",
        group="g1",
    ),
)
_HELD = (
    Fixture(
        id="h0",
        messages=(Message(role="user", content="hey"),),
        expected="yes",
        group="g2",
    ),
)
_SPLIT = Split(search=_SEARCH, held_out=_HELD)


def _script_for_flip_at_1() -> dict[str, tuple[str, str]]:
    """Native cand-a passes; drop-1 flips it (search fails), drop-2
    passes again -- so flips_at is exactly [1], not [1, 2]."""
    script: dict[str, tuple[str, str]] = {}
    for fmt in (_FMT_A, _FMT_B):
        full = fmt.render(_SEARCH[0].messages)
        held_full = fmt.render(_HELD[0].messages)
        script[full] = ("yes", chr(_BOUNDARY)) if fmt.name == "cand-a" else ("WRONG", "")
        script[held_full] = ("yes", chr(_BOUNDARY)) if fmt.name == "cand-a" else ("WRONG", "")
        # Truncated prompts: drop 1 -> cand-a fails; drop 2 -> passes again.
        script[full[:-1]] = ("WRONG", "")
        script[full[:-2]] = ("yes", chr(_BOUNDARY)) if fmt.name == "cand-a" else ("WRONG", "")
        script[held_full[:-1]] = ("WRONG", "")
        script[held_full[:-2]] = ("yes", chr(_BOUNDARY)) if fmt.name == "cand-a" else ("WRONG", "")
    return script


def _sweep(**kwargs) -> dict:
    backend = ScriptedBackend(_script_for_flip_at_1())
    return prompt_token_depth_sweep(
        backend,
        (_FMT_A, _FMT_B),
        _SPLIT,
        boundary=(_BOUNDARY,),
        max_new_tokens=_CAP,
        seed=0,
        **kwargs,
    )


class TokenWrapperTests(unittest.TestCase):
    def test_drops_whole_token_ids_and_fingerprint_uses_t_suffix(self) -> None:
        backend = ScriptedBackend({})
        wrapped = TokenTruncatedPromptBackend(backend, 2)
        self.assertEqual(wrapped.fingerprint, "cycle206-scripted-backend+prompt-trunc-2t")
        self.assertEqual(wrapped.encode("hello"), (ord("h"), ord("e"), ord("l")))
        self.assertEqual(wrapped.decode((ord("x"),)), "x")

    def test_beyond_length_raises_instead_of_empty_prompt(self) -> None:
        wrapped = TokenTruncatedPromptBackend(ScriptedBackend({}), 2)
        with self.assertRaises(ValueError):
            wrapped.encode("hi")
        with self.assertRaises(ValueError):
            TokenTruncatedPromptBackend(ScriptedBackend({}), 0)
        with self.assertRaises(ValueError):
            TokenTruncatedPromptBackend(ScriptedBackend({}), "2")  # type: ignore[arg-type]

    def test_non_string_fingerprint_rejected(self) -> None:
        backend = ScriptedBackend({})
        backend.fingerprint = 42  # type: ignore[assignment]
        with self.assertRaises(ValueError):
            TokenTruncatedPromptBackend(backend, 1)

    def test_prefill_wrapped_inner_refused(self) -> None:
        # B1: naive [:-K] over a PrefillBackend drops prefill ids first,
        # leaving the prompt intact -- a silent wrong measurement.
        inner = PrefillBackend(ScriptedBackend({}), (900, 901), "p")
        with self.assertRaises(ValueError):
            TokenTruncatedPromptBackend(inner, 1)

    def test_generate_passes_ids_through_unchanged(self) -> None:
        backend = ScriptedBackend({})
        wrapped = TokenTruncatedPromptBackend(backend, 1)
        generation = wrapped.generate((ord("a"),), max_new_tokens=_CAP, seed=0, cancel=Event())
        self.assertEqual(generation.reason, "limit")


class TokenSweepTests(unittest.TestCase):
    def test_p1_shape_granularity_native_and_suffix_counts(self) -> None:
        result = _sweep(max_drop_tokens=2)
        self.assertEqual(result["granularity"], "tokens")
        self.assertEqual(result["native_winner"], "cand-a")
        self.assertEqual(result["native_decision"], "compatible_on_fixture_set")
        self.assertEqual(result["max_drop_tokens"], 2)
        self.assertEqual(result["suffix_token_counts"], {"cand-a": 2, "cand-b": 2})
        self.assertEqual([d["drop_tokens"] for d in result["depths"]], [1, 2])
        self.assertTrue(all(d["status"] == "applied" for d in result["depths"]))
        self.assertEqual(result["inapplicable_at"], [])

    def test_flip_at_first_depth_only(self) -> None:
        result = _sweep(max_drop_tokens=2)
        self.assertEqual(result["depths"][0]["winner_changed"], True)
        self.assertEqual(result["depths"][0]["decision"], "no_candidate")
        self.assertEqual(result["depths"][1]["winner_changed"], False)
        self.assertEqual(result["flips_at"], [1])

    def test_p2_beyond_length_is_inapplicable_not_a_flip(self) -> None:
        # Single-char prompt ">": every drop meets or exceeds its length.
        fmt = Format(name="tiny", bos="", user_prefix=">", assistant_prefix="", turn_end="")
        search = (
            Fixture(
                id="s0",
                messages=(Message(role="user", content=""),),
                expected="yes",
                group="g1",
            ),
        )
        split = Split(search=search, held_out=_HELD)
        backend = ScriptedBackend({})
        result = prompt_token_depth_sweep(
            backend,
            (fmt,),
            split,
            boundary=(_BOUNDARY,),
            max_drop_tokens=2,
            max_new_tokens=_CAP,
            seed=0,
        )
        self.assertEqual(result["min_prompt_tokens"], 1)
        for depth in result["depths"]:
            self.assertEqual(depth["status"], "inapplicable")
            # The run's failed decision is recorded, not skipped...
            self.assertEqual(depth["decision"], "failed")
            # ...but it is never counted as a flip, even though the
            # decision differs from the native no_candidate.
            self.assertFalse(depth["winner_changed"])
        self.assertEqual(result["flips_at"], [])
        self.assertEqual(result["inapplicable_at"], [1, 2])

    def test_mixed_statuses_from_shortest_prompt(self) -> None:
        # Search prompt ">ab" (3 tokens): drops 1-2 applied, 3-4
        # inapplicable. Held-out ">hey" (4) does not rescue drop 3.
        fmt = Format(name="short", bos="", user_prefix=">", assistant_prefix="", turn_end="")
        search = (
            Fixture(
                id="s0",
                messages=(Message(role="user", content="ab"),),
                expected="yes",
                group="g1",
            ),
        )
        held = (
            Fixture(
                id="h0",
                messages=(Message(role="user", content="hey"),),
                expected="yes",
                group="g2",
            ),
        )
        split = Split(search=search, held_out=held)
        backend = ScriptedBackend({})
        result = prompt_token_depth_sweep(
            backend,
            (fmt,),
            split,
            boundary=(_BOUNDARY,),
            max_drop_tokens=4,
            max_new_tokens=_CAP,
            seed=0,
        )
        self.assertEqual(
            [d["status"] for d in result["depths"]],
            ["applied", "applied", "inapplicable", "inapplicable"],
        )
        self.assertEqual(result["inapplicable_at"], [3, 4])
        self.assertEqual(result["flips_at"], [])

    def test_applied_failure_still_counts_when_decision_differs(self) -> None:
        # An applied depth whose truncated run genuinely fails (not
        # because the prompt was too short) is a change, not a
        # labelling artefact: here drop 1 flips the winner.
        result = _sweep(max_drop_tokens=1)
        self.assertEqual(result["depths"][0]["status"], "applied")
        self.assertTrue(result["depths"][0]["winner_changed"])
        self.assertEqual(result["flips_at"], [1])

    def test_precheck_skips_natively_failing_prompts(self) -> None:
        # A prompt whose encoding fails natively carries no
        # truncation-specific fact: the pre-run check skips it (the
        # except branch) and the minimum comes from the prompts that
        # do encode, while the sweep itself still completes.
        class BoomBackend(ScriptedBackend):
            def encode(self, text: str) -> tuple[int, ...]:
                if "BOOM" in text:
                    raise ValueError("boom encode")
                return super().encode(text)

        search = (
            Fixture(
                id="s0",
                messages=(Message(role="user", content="BOOM"),),
                expected="yes",
                group="g1",
            ),
        )
        split = Split(search=search, held_out=_HELD)
        backend = BoomBackend({})
        result = prompt_token_depth_sweep(
            backend,
            (_FMT_A, _FMT_B),
            split,
            boundary=(_BOUNDARY,),
            max_drop_tokens=2,
            max_new_tokens=_CAP,
            seed=0,
        )
        held_lengths = [len(fmt.render(_HELD[0].messages)) for fmt in (_FMT_A, _FMT_B)]
        self.assertEqual(result["min_prompt_tokens"], min(held_lengths))
        self.assertEqual(result["depths"][0]["status"], "applied")

    def test_suffix_count_none_when_suffix_unencodable(self) -> None:
        # The suffix annotation degrades to None, never a sweep
        # failure, when the suffix alone does not encode.
        class SuffixFailBackend(ScriptedBackend):
            def encode(self, text: str) -> tuple[int, ...]:
                if text == "a:":
                    raise ValueError("suffix unencodable")
                return super().encode(text)

        backend = SuffixFailBackend(_script_for_flip_at_1())
        result = prompt_token_depth_sweep(
            backend,
            (_FMT_A, _FMT_B),
            _SPLIT,
            boundary=(_BOUNDARY,),
            max_drop_tokens=1,
            max_new_tokens=_CAP,
            seed=0,
        )
        self.assertEqual(result["suffix_token_counts"], {"cand-a": None, "cand-b": 2})

    def test_validation_mirrors_byte_range(self) -> None:
        backend = ScriptedBackend({})
        for bad in (0, 9, "2"):
            with self.assertRaises(ValueError):
                prompt_token_depth_sweep(
                    backend,
                    (_FMT_A,),
                    _SPLIT,
                    boundary=(_BOUNDARY,),
                    max_drop_tokens=bad,  # type: ignore[arg-type]
                )

    def test_supplied_native_is_honoured(self) -> None:
        # Cycle 207 (wave-10 B10-2) changed the supplied-native
        # contract: a baseline is honoured only when it carries the
        # configuration_digest of exactly this sweep's configuration.
        # A genuine native computed for these parameters is honoured;
        # hand-made dicts and wrong-configuration results are rejected.
        backend = ScriptedBackend(_script_for_flip_at_1())
        native = run_experiment(
            backend,
            (_FMT_A, _FMT_B),
            _SPLIT,
            boundary=(_BOUNDARY,),
            max_new_tokens=_CAP,
            seed=0,
        )
        result = _sweep(max_drop_tokens=1, native=native)
        self.assertEqual(result["native_decision"], native["decision"])
        self.assertEqual(result["native_winner"], native["selected_candidate"])
        for bad_native in (
            {},
            {"decision": "x"},
            {"selected_candidate": "y"},
            {"decision": "ambiguous", "selected_candidate": None},
        ):
            with self.assertRaises(ValueError):
                _sweep(max_drop_tokens=1, native=bad_native)  # type: ignore[arg-type]
        stale = run_experiment(
            ScriptedBackend(_script_for_flip_at_1()),
            (_FMT_A, _FMT_B),
            _SPLIT,
            boundary=(_BOUNDARY,),
            max_new_tokens=_CAP,
            seed=1,
        )
        with self.assertRaises(ValueError):
            _sweep(max_drop_tokens=1, native=stale)

    def test_shared_native_is_used_by_both_sweeps(self) -> None:
        # The CLI's shared-native path passes one native dict to both
        # sweeps; each sweep must honour it rather than recomputing.
        # Cycle 207: the shared native must be a genuine result for
        # these parameters (digest-stamped); hand-made dicts no longer
        # qualify as a shared native.
        backend = ScriptedBackend(_script_for_flip_at_1())
        native = run_experiment(
            backend,
            (_FMT_A, _FMT_B),
            _SPLIT,
            boundary=(_BOUNDARY,),
            max_new_tokens=_CAP,
            seed=0,
        )
        byte_result = prompt_sensitivity_sweep(
            backend,
            (_FMT_A, _FMT_B),
            _SPLIT,
            boundary=(_BOUNDARY,),
            max_drop_bytes=1,
            max_new_tokens=_CAP,
            seed=0,
            native=native,
        )
        token_result = prompt_token_depth_sweep(
            backend,
            (_FMT_A, _FMT_B),
            _SPLIT,
            boundary=(_BOUNDARY,),
            max_drop_tokens=1,
            max_new_tokens=_CAP,
            seed=0,
            native=native,
        )
        self.assertEqual(byte_result["native_winner"], native["selected_candidate"])
        self.assertEqual(token_result["native_winner"], native["selected_candidate"])
        self.assertEqual(byte_result["native_decision"], native["decision"])
        self.assertEqual(token_result["native_decision"], native["decision"])


class TokenReportingTests(unittest.TestCase):
    def _payload(self) -> dict:
        return {
            "granularity": "tokens",
            "native_winner": "cand-a",
            "native_decision": "compatible_on_fixture_set",
            "max_drop_tokens": 2,
            "depths": [
                {
                    "drop_tokens": 1,
                    "status": "applied",
                    "decision": "no_candidate",
                    "selected_candidate": None,
                    "winner_changed": True,
                },
                {
                    "drop_tokens": 2,
                    "status": "inapplicable",
                    "decision": "failed",
                    "selected_candidate": None,
                    "winner_changed": False,
                },
            ],
            "flips_at": [1],
            "inapplicable_at": [2],
        }

    def test_token_section_renders_status_and_inapplicable(self) -> None:
        html = reporting._token_depth_section({"sensitivity_sweep_tokens": self._payload()})
        self.assertIn('id="token-depth-sweep"', html)
        self.assertIn("Token-depth sensitivity sweep", html)
        self.assertIn("Dropped tokens", html)
        self.assertIn("inapplicable", html)
        self.assertIn("Depths where the winner or decision changed: [1].", html)
        self.assertIn("Depths that could not be applied (shorter than the depth): [2].", html)

    def test_token_section_skips_non_dict_depths(self) -> None:
        payload = self._payload()
        payload["depths"] = ["junk", payload["depths"][0]]
        html = reporting._token_depth_section({"sensitivity_sweep_tokens": payload})
        self.assertIn("Dropped tokens", html)
        self.assertNotIn("junk", html)

    def test_token_section_empty_and_absent(self) -> None:
        self.assertIn(
            "No depths were evaluated.",
            reporting._token_depth_section({"sensitivity_sweep_tokens": {"depths": []}}),
        )
        self.assertEqual(reporting._token_depth_section({}), "")

    def test_token_section_escapes_hostile_values(self) -> None:
        payload = self._payload()
        payload["depths"][0]["selected_candidate"] = "<x>"
        html = reporting._token_depth_section({"sensitivity_sweep_tokens": payload})
        self.assertIn("&lt;x&gt;", html)
        self.assertNotIn("<x>", html)

    def test_byte_section_unchanged_by_token_key(self) -> None:
        byte_payload = {
            "native_winner": "chatml",
            "native_decision": "compatible_on_fixture_set",
            "flips_at": [3],
            "depths": [
                {
                    "drop_bytes": 1,
                    "winner_changed": False,
                    "decision": "compatible_on_fixture_set",
                    "selected_candidate": "chatml",
                },
                {
                    "drop_bytes": 3,
                    "winner_changed": True,
                    "decision": "ambiguous",
                    "selected_candidate": None,
                },
            ],
        }
        byte_only = reporting._sweep_section({"sensitivity_sweep": byte_payload})
        with_token = reporting._sweep_section(
            {
                "sensitivity_sweep": byte_payload,
                "sensitivity_sweep_tokens": self._payload(),
            }
        )
        self.assertEqual(byte_only, with_token)
        self.assertIn("Native winner: chatml (compatible_on_fixture_set).", byte_only)
        self.assertIn("Depths where the winner or decision changed: [3].", byte_only)

    def test_full_report_includes_token_section_only_when_present(self) -> None:
        base = {"decision": "compatible_on_fixture_set", "search": []}
        without = reporting.render_html(base)
        self.assertNotIn('id="token-depth-sweep"', without)
        with_token = reporting.render_html({**base, "sensitivity_sweep_tokens": self._payload()})
        self.assertIn('id="token-depth-sweep"', with_token)
        self.assertIn("Token-depth sensitivity sweep", with_token)


class PerProbeEscapeTests(unittest.TestCase):
    """Wave-10 red-team A-F2: the four per-probe HTML sections used bare
    escape() on payload-derived strings, so a lone surrogate passed
    through verbatim and save_report's UTF-8 write raised
    UnicodeEncodeError. All four sections now route payload-derived
    interpolations through _escape_text (controls stripped), while
    U+2028/U+2029 pass through unchanged (cycle-100 pin)."""

    _SUR = "\ud800"

    def test_token_section_strips_lone_surrogate(self) -> None:
        payload = {
            "native_winner": "chatml" + self._SUR,
            "native_decision": "compatible_on_fixture_set",
            "flips_at": [],
            "inapplicable_at": [],
            "depths": [
                {
                    "drop_tokens": 1,
                    "status": "applied",
                    "decision": "no_candidate" + self._SUR,
                    "selected_candidate": "chatml",
                    "winner_changed": False,
                }
            ],
        }
        html = reporting._token_depth_section({"sensitivity_sweep_tokens": payload})
        self.assertNotIn(self._SUR, html)
        html.encode("utf-8")
        self.assertIn("chatml", html)
        self.assertIn("no_candidate", html)

    def test_sweep_section_strips_surrogate_and_controls(self) -> None:
        payload = {
            "native_winner": "chatml",
            "native_decision": "compatible_on_fixture_set",
            "flips_at": [],
            "depths": [
                {
                    "drop_bytes": 1,
                    "winner_changed": False,
                    "decision": "compat\x00ible\x1b" + self._SUR,
                    "selected_candidate": "chatml",
                }
            ],
        }
        html = reporting._sweep_section({"sensitivity_sweep": payload})
        self.assertNotIn(self._SUR, html)
        self.assertNotIn("\x00", html)
        self.assertNotIn("\x1b", html)
        html.encode("utf-8")
        self.assertIn("compatible", html)

    def test_rotation_section_strips_surrogate_name_and_winner(self) -> None:
        result = {
            "rotation": {
                "available": True,
                "candidates": {"chatml" + self._SUR: {"status": "unavailable"}},
                "conditions": {"rot_0": {"winner": "chatml" + self._SUR}},
            }
        }
        html = reporting._rotation_section(result)
        self.assertNotIn(self._SUR, html)
        html.encode("utf-8")
        self.assertIn("chatml", html)

    def test_field_order_section_strips_surrogate_name_and_direction(self) -> None:
        result = {
            "field_order": {
                "available": True,
                "candidates": {
                    "chatml" + self._SUR: {"status": "unavailable"},
                    "gemma": {
                        "status": "available",
                        "decision": {"verdict": "falsified", "direction": "up" + self._SUR},
                    },
                },
            }
        }
        html = reporting._field_order_section(result)
        self.assertNotIn(self._SUR, html)
        html.encode("utf-8")
        self.assertIn("field-order invariance falsified", html)

    def test_save_report_survives_surrogate_sweep_payload(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        base = json.loads((repo_root / "examples" / "sample-report" / "results.json").read_text())
        base["sensitivity_sweep_tokens"] = {
            "native_winner": "chatml" + self._SUR,
            "native_decision": "compatible_on_fixture_set",
            "flips_at": [],
            "inapplicable_at": [],
            "depths": [
                {
                    "drop_tokens": 1,
                    "status": "applied",
                    "decision": "compatible" + self._SUR,
                    "selected_candidate": "chatml",
                    "winner_changed": False,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            json_path, html_path = reporting.save_report(base, Path(tmp) / "rep")
            html_text = html_path.read_text(encoding="utf-8")
            json_exists = json_path.exists()
        self.assertIn('id="token-depth-sweep"', html_text)
        self.assertNotIn(self._SUR, html_text)
        self.assertTrue(json_exists)


class _Sentinel(dict):
    def __missing__(self, key):  # noqa: ANN001, ANN202
        return self


def _write_fixture_file(directory: str) -> Path:
    path = Path(directory) / "fixture.json"
    path.write_text(
        json.dumps(
            {
                "search": [
                    {
                        "id": "s1",
                        "group": "g1",
                        "messages": [{"role": "user", "content": "hi"}],
                        "expected": "yo",
                    }
                ],
                "held_out": [
                    {
                        "id": "h1",
                        "group": "g2",
                        "messages": [{"role": "user", "content": "hey"}],
                        "expected": "yo",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


class _FakeHFBackend:
    fingerprint = "fake-hf"

    def __init__(self, *args, **kwargs) -> None:
        pass

    def encode(self, text):  # noqa: ANN001, ANN202
        return (7, 8)


class TokenCliTests(unittest.TestCase):
    def test_parser_defaults_and_flags(self) -> None:
        ns = build_parser().parse_args(
            [
                "run",
                "--model",
                "m",
                "--dataset",
                "d",
                "--family",
                "chatml",
                "--boundary-ids",
                "198",
                "--output",
                "o",
            ]
        )
        self.assertFalse(ns.token_depth_sweep)
        self.assertEqual(ns.max_drop_tokens, 4)
        ns = build_parser().parse_args(
            [
                "run",
                "--model",
                "m",
                "--dataset",
                "d",
                "--family",
                "chatml",
                "--boundary-ids",
                "198",
                "--output",
                "o",
                "--token-depth-sweep",
                "--max-drop-tokens",
                "7",
            ]
        )
        self.assertTrue(ns.token_depth_sweep)
        self.assertEqual(ns.max_drop_tokens, 7)

    def test_max_drop_tokens_range_enforced_up_front(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = main(
                    [
                        "run",
                        "--model",
                        "m",
                        "--dataset",
                        "nonexistent.json",
                        "--family",
                        "chatml",
                        "--output",
                        "o",
                        "--token-depth-sweep",
                        "--max-drop-tokens",
                        "9",
                    ]
                )
        except SystemExit as exc:
            code, err_text = exc.code, err.getvalue()
        else:
            err_text = err.getvalue()
        self.assertEqual(code, 2)
        self.assertIn("max-drop-tokens", err_text)

    def _run_with(self, argv_extra, exp_patches):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture_file(tmp)
            outdir = Path(tmp) / "report"
            result_doc = {
                "decision": "compatible_on_fixture_set",
                "selected_candidate": None,
                "search": [],
                "held_out": {},
            }
            sentinel = _Sentinel()
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(backends, "LocalHFBackend", _FakeHFBackend))
                stack.enter_context(
                    mock.patch.object(cli_module, "run_experiment", return_value=result_doc)
                )
                stack.enter_context(
                    mock.patch.object(
                        cli_module,
                        "save_report",
                        return_value=(str(outdir / "results.json"), str(outdir / "report.md")),
                    )
                )
                from model_protocol_lab import doctor as doctor_mod
                from model_protocol_lab import reporting as reporting_mod

                stack.enter_context(
                    mock.patch.object(doctor_mod, "template_conformance", return_value={"ok": True})
                )
                stack.enter_context(
                    mock.patch.object(
                        experiments, "budget_frontier", return_value={"available": False}
                    )
                )
                for name in exp_patches:
                    stack.enter_context(mock.patch.object(experiments, name, return_value=sentinel))

                def fake_prepare(path):  # noqa: ANN001, ANN202
                    Path(path).mkdir(parents=True, exist_ok=False)
                    return Path(path)

                stack.enter_context(
                    mock.patch.object(reporting_mod, "prepare_report_dir", fake_prepare)
                )
                argv = [
                    "run",
                    "--model",
                    "ckpt-dir",
                    "--dataset",
                    str(fixture),
                    "--family",
                    "chatml",
                    "--boundary-ids",
                    "198,199",
                    "--output",
                    str(outdir),
                    "--seed",
                    "7",
                    *argv_extra,
                ]
                out, err = io.StringIO(), io.StringIO()
                try:
                    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                        code = cli_module.main(argv)
                except SystemExit as exc:
                    kind, code = "exit", exc.code
                else:
                    kind, code = "return", code
            return (kind, code), result_doc, sentinel

    def test_token_depth_sweep_dispatches(self) -> None:
        (kind, code), result_doc, sentinel = self._run_with(
            ["--token-depth-sweep"],
            ["prompt_token_depth_sweep"],
        )
        self.assertEqual((kind, code), ("return", 0))
        self.assertIs(result_doc["sensitivity_sweep_tokens"], sentinel)

    def test_both_sweeps_share_one_native(self) -> None:
        # Both flags together: the CLI computes one shared native via
        # experiments.run_experiment and passes it to both sweeps, so
        # the shared call count is exactly one (not two sweep natives).
        calls: list = []
        real_run = experiments.run_experiment

        def counting_run(*args, **kwargs):  # noqa: ANN001, ANN202
            calls.append(1)
            return {
                "decision": "compatible_on_fixture_set",
                "selected_candidate": None,
                "search": [],
                "held_out": {},
            }

        with mock.patch.object(experiments, "run_experiment", side_effect=counting_run):
            (kind, code), result_doc, sentinel = self._run_with(
                ["--sensitivity-sweep", "--token-depth-sweep"],
                ["prompt_sensitivity_sweep", "prompt_token_depth_sweep"],
            )
        self.assertEqual((kind, code), ("return", 0))
        self.assertIs(result_doc["sensitivity_sweep"], sentinel)
        self.assertIs(result_doc["sensitivity_sweep_tokens"], sentinel)
        # The CLI main run is patched at cli_module level in _run_with;
        # the only experiments.run_experiment call is the shared native.
        self.assertEqual(len(calls), 1)
        self.assertIsNot(real_run, None)


if __name__ == "__main__":
    unittest.main()
