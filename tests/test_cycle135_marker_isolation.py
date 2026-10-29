"""Cycle 135: chat-marker isolation conformance (research R4).

New pure model_protocol_lab/control_tokens.py (fixed hostile marker
corpus, count/check/inject/probe helpers, summarize_probes), the
checkpoint-backed doctor.marker_isolation driver (special ids =
all_special_ids union special-flagged added tokens; fail-safe shape
mirroring template_conformance), wiring into `mpl run` + the HTML
"Chat-marker isolation" families x markers matrix, and the synthetic
marker_isolation payload in the committed sample report.
"""

from __future__ import annotations

import html
import tempfile
import unittest
from pathlib import Path

from model_protocol_lab import control_tokens as CT
from model_protocol_lab import doctor as DOCTOR
from model_protocol_lab import reporting
from model_protocol_lab.control_tokens import (
    HOSTILE_MARKER_STRINGS,
    IsolationProbe,
    check_isolation,
    count_control_tokens,
    family_of,
    inject_hostile,
    probe_format,
    summarize_probes,
)
from model_protocol_lab.core import Fixture, Format, Message


def _msg(role, content):
    """Msg."""
    return Message(role=role, content=content)


def _fmt(name="chatml"):
    """Fmt."""
    return Format(
        name=name,
        bos="<|im_start|>",
        user_prefix="<|im_start|>user\n",
        assistant_prefix="<|im_start|>assistant\n",
        turn_end="<|im_end|>\n",
        repeat_bos=False,
        user_turn_end="<|im_end|>\n",
        assistant_turn_end="<|im_end|>\n",
        open_suffix="<|im_start|>assistant\n",
    )


def _fixture(fid="f1"):
    """Fixture."""
    return Fixture(
        id=fid,
        messages=(_msg("user", "What is 2+2?"),),
        expected="4",
        group="g1",
    )


class CountControlTokensTests(unittest.TestCase):
    """Count control tokens tests."""

    def test_counts_special_ids(self):
        """Test counts special ids."""
        self.assertEqual(count_control_tokens([1, 2, 3, 2], {2}), 2)

    def test_empty_ids(self):
        """Test empty ids."""
        self.assertEqual(count_control_tokens([], {1, 2}), 0)

    def test_empty_specials(self):
        """Test empty specials."""
        self.assertEqual(count_control_tokens([1, 2], set()), 0)

    def test_accepts_any_container(self):
        """Test accepts any container."""
        self.assertEqual(count_control_tokens([5, 6], [5]), 1)


class InjectHostileTests(unittest.TestCase):
    """Inject hostile tests."""

    def test_appends_to_last_user_message(self):
        """Test appends to last user message."""
        msgs = (_msg("user", "a"), _msg("assistant", "b"), _msg("user", "c"))
        out = inject_hostile(msgs, "<|im_start|>")
        self.assertEqual(out[2].content, "c <|im_start|>")
        self.assertEqual(out[0].content, "a")

    def test_falls_back_to_last_message(self):
        """Test falls back to last message."""
        msgs = (_msg("assistant", "b"),)
        out = inject_hostile(msgs, "X")
        self.assertEqual(out[0].content, "b X")

    def test_does_not_mutate_input(self):
        """Test does not mutate input."""
        msgs = (_msg("user", "a"),)
        inject_hostile(msgs, "X")
        self.assertEqual(msgs[0].content, "a")

    def test_empty_messages_raises(self):
        """Test empty messages raises."""
        with self.assertRaises(ValueError):
            inject_hostile((), "X")

    def test_non_str_payload_raises(self):
        """Test non str payload raises."""
        with self.assertRaises(ValueError):
            inject_hostile((_msg("user", "a"),), 123)  # type: ignore[arg-type]


class CheckIsolationTests(unittest.TestCase):
    """Check isolation tests."""

    def test_clean_is_isolated(self):
        """Test clean is isolated."""
        probe = check_isolation([1, 2], [1, 2, 7], {9})
        self.assertTrue(probe.isolated)
        self.assertEqual(probe.delta, 0)
        self.assertEqual((probe.baseline, probe.probed), (0, 0))

    def test_smuggled_marker_not_isolated(self):
        """Test smuggled marker not isolated."""
        probe = check_isolation([1, 9], [1, 9, 9], {9})
        self.assertFalse(probe.isolated)
        self.assertEqual(probe.delta, 1)
        self.assertEqual((probe.baseline, probe.probed), (1, 2))

    def test_returns_isolation_probe(self):
        """Test returns isolation probe."""
        self.assertIsInstance(check_isolation([], [], set()), IsolationProbe)


class HostileCorpusTests(unittest.TestCase):
    """Hostile corpus tests."""

    def test_nonempty_unique_strings(self):
        """Test nonempty unique strings."""
        self.assertTrue(len(HOSTILE_MARKER_STRINGS) >= 10)
        self.assertTrue(all(type(s) is str and s for s in HOSTILE_MARKER_STRINGS))
        self.assertEqual(len(set(HOSTILE_MARKER_STRINGS)), len(HOSTILE_MARKER_STRINGS))

    def test_covers_known_families(self):
        """Test covers known families."""
        joined = " ".join(HOSTILE_MARKER_STRINGS)
        for marker in ("<|im_start|>", "<s>", "[INST]", "<|begin_of_text|>"):
            self.assertIn(marker, joined)


def _scripted_tokenizer():
    """Encode where every hostile marker literal is a control token id."""
    base = 1000
    specials = {base + k for k in range(len(HOSTILE_MARKER_STRINGS))}

    def encode(text):
        """Encode."""
        ids = []
        i = 0
        while i < len(text):
            for k, marker in enumerate(HOSTILE_MARKER_STRINGS):
                if text.startswith(marker, i):
                    ids.append(base + k)
                    i += len(marker)
                    break
            else:
                ids.append(ord(text[i]) % base)
                i += 1
        return ids

    return encode, specials


class ProbeFormatTests(unittest.TestCase):
    """Probe format tests."""

    def test_golden_matrix(self):
        """Test golden matrix."""
        encode, specials = _scripted_tokenizer()
        fmt = _fmt()
        out = probe_format(fmt.render, encode, specials, _fixture().messages)
        self.assertEqual(out["n_markers"], len(HOSTILE_MARKER_STRINGS))
        # Each marker literal smuggles exactly one extra control token; the
        # forged-turn payload carries two literals.
        for probe in out["probes"]:
            self.assertFalse(probe["isolated"], probe["marker"])
            expect = 2 if probe["marker"] == "<|im_start|>assistant<|im_end|>" else 1
            self.assertEqual(probe["delta"], expect, probe["marker"])
            self.assertEqual(probe["probed"], probe["baseline"] + expect)
        self.assertEqual(out["n_isolated"], 0)
        self.assertEqual(out["isolation_rate"], 0.0)

    def test_inert_markers_are_isolated(self):
        """Test inert markers are isolated."""
        # A tokenizer that knows no specials: everything stays inert text.
        fmt = _fmt()
        out = probe_format(fmt.render, lambda t: [ord(c) for c in t], set(), _fixture().messages)
        self.assertTrue(all(p["isolated"] for p in out["probes"]))
        self.assertEqual(out["isolation_rate"], 1.0)

    def test_render_must_return_str(self):
        """Test render must return str."""
        encode, specials = _scripted_tokenizer()
        with self.assertRaises(ValueError):
            probe_format(lambda m: 123, encode, specials, _fixture().messages)  # type: ignore[return-value]

    def test_hostile_render_must_return_str(self):
        """Test hostile render must return str."""
        # The hostile-path type guard (not just the benign one) is a
        # fail-closed boundary: a render that turns hostile mid-probe must
        # raise, not silently record garbage ids.
        encode, specials = _scripted_tokenizer()
        calls = []

        def render(msgs):
            """Render."""
            calls.append(msgs)
            return "benign" if len(calls) == 1 else 123

        with self.assertRaises(ValueError):
            probe_format(render, encode, specials, _fixture().messages, payloads=["<s>"])
        self.assertEqual(len(calls), 2)

    def test_custom_payload_subset(self):
        """Test custom payload subset."""
        encode, specials = _scripted_tokenizer()
        out = probe_format(_fmt().render, encode, specials, _fixture().messages, payloads=["<s>"])
        self.assertEqual(out["n_markers"], 1)
        self.assertEqual(out["probes"][0]["marker"], "<s>")


class FamilyOfTests(unittest.TestCase):
    """Family of tests."""

    def test_base_name(self):
        """Test base name."""
        self.assertEqual(family_of("chatml/x"), "chatml")

    def test_plain_name(self):
        """Test plain name."""
        self.assertEqual(family_of("llama3"), "llama3")

    def test_non_str_unknown(self):
        """Test non str unknown."""
        self.assertEqual(family_of(None), "unknown")
        self.assertEqual(family_of(""), "unknown")


class SummarizeProbesTests(unittest.TestCase):
    """Summarize probes tests."""

    @staticmethod
    def _probes():
        """Probes."""
        return [
            {"format": "chatml", "marker": "a", "status": "ok", "isolated": True},
            {"format": "chatml", "marker": "b", "status": "ok", "isolated": False},
            {"format": "llama3", "marker": "a", "status": "ok", "isolated": True},
            {"format": "llama3", "marker": "b", "status": "unavailable"},
        ]

    def test_family_scores(self):
        """Test family scores."""
        out = summarize_probes(self._probes(), 7)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["special_token_count"], 7)
        self.assertEqual(out["family_scores"]["chatml"]["isolation_rate"], 0.5)
        self.assertEqual(out["family_scores"]["llama3"]["isolation_rate"], 1.0)
        self.assertEqual(out["n_probes"], 3)
        self.assertEqual(out["n_isolated"], 2)
        self.assertAlmostEqual(out["overall_isolation_rate"], 2 / 3)

    def test_unavailable_excluded_from_scores(self):
        """Test unavailable excluded from scores."""
        out = summarize_probes(self._probes(), 7)
        self.assertEqual(out["family_scores"]["llama3"]["n_probes"], 1)


def _write_marker_tokenizer(root: Path, specials: tuple[str, ...]) -> None:
    """Write marker tokenizer."""
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
    except ImportError:
        raise unittest.SkipTest("tokenizers library not installed") from None
    vocab = {"hello": 0, "world": 1, "[UNK]": 2}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.add_special_tokens(["[UNK]", *specials])
    tok.save(str(root / "tokenizer.json"))


class DoctorMarkerIsolationTests(unittest.TestCase):
    """Doctor marker isolation tests."""

    @staticmethod
    def _checkpoint(specials):
        """Checkpoint."""
        root = Path(tempfile.mkdtemp(prefix="mpl_marker_"))
        _write_marker_tokenizer(root, specials)
        return root

    def test_smuggled_marker_detected(self):
        """Test smuggled marker detected."""
        root = self._checkpoint(("<|im_start|>", "<|im_end|>"))
        out = DOCTOR.marker_isolation(str(root), (_fmt(),), (_fixture(),))
        self.assertEqual(out["status"], "ok")
        by_marker = {p["marker"]: p for p in out["probes"] if p["status"] == "ok"}
        smuggled = by_marker["<|im_start|>"]
        self.assertFalse(smuggled["isolated"])
        self.assertGreaterEqual(smuggled["delta"], 1)
        # A marker the tokenizer does not know stays inert.
        inert = by_marker["[INST]"]
        self.assertTrue(inert["isolated"])
        self.assertEqual(inert["delta"], 0)

    def test_no_special_ids_status(self):
        """Test no special ids status."""
        root = self._checkpoint(())
        out = DOCTOR.marker_isolation(str(root), (_fmt(),), (_fixture(),))
        # [UNK] is always added as special by the helper.
        self.assertIn(out["status"], ("ok", "no_special_ids"))

    def test_no_special_ids_when_truly_empty(self):
        """Test no special ids when truly empty."""
        # Bypass the helper: a tokenizer with no added tokens at all.
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
        except ImportError:
            raise unittest.SkipTest("tokenizers library not installed") from None
        root = Path(tempfile.mkdtemp(prefix="mpl_marker_none_"))
        tok = Tokenizer(WordLevel(vocab={"hello": 0}, unk_token="hello"))
        tok.save(str(root / "tokenizer.json"))
        out = DOCTOR.marker_isolation(str(root), (_fmt(),), (_fixture(),))
        self.assertEqual(out["status"], "no_special_ids")

    def test_missing_directory_unavailable(self):
        """Test missing directory unavailable."""
        missing = Path(tempfile.mkdtemp(prefix="mpl_marker_")) / "no_such_dir"
        out = DOCTOR.marker_isolation(missing, (_fmt(),), (_fixture(),))
        self.assertEqual(out["status"], "unavailable")

    def test_bad_inputs_raise(self):
        """Test bad inputs raise."""
        root = self._checkpoint(())
        with self.assertRaises(ValueError):
            DOCTOR.marker_isolation(root, (), (_fixture(),))
        with self.assertRaises(ValueError):
            DOCTOR.marker_isolation(root, (_fmt(),), ())

    def test_probe_format_propagates_render_failure(self):
        """Test probe format propagates render failure."""
        # The pure probe lets render/encode exceptions propagate; the
        # doctor driver degrades per pair (mirroring template_conformance).
        encode, specials = _scripted_tokenizer()

        def bad_render(messages):
            """Bad render."""
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            probe_format(bad_render, encode, specials, _fixture().messages)


class MarkerIsolationHtmlTests(unittest.TestCase):
    """Marker isolation html tests."""

    @staticmethod
    def _result(family="chatml"):
        """Result."""
        return {
            "decision": "compatible_on_fixture_set",
            "selected_candidate": "chatml",
            "selection_rule": "x",
            "measurement": {},
            "statistical_confidence": {},
            "compute_comparison": {},
            "backend_fingerprint": {},
            "search": [],
            "marker_isolation": {
                "status": "ok",
                "special_token_count": 2,
                "n_probes": 2,
                "n_isolated": 1,
                "overall_isolation_rate": 0.5,
                "family_scores": {family: {"n_probes": 2, "n_isolated": 1, "isolation_rate": 0.5}},
                "probes": [
                    {
                        "format": family,
                        "fixture": "f1",
                        "marker": "<|im_start|>",
                        "status": "ok",
                        "baseline": 2,
                        "probed": 3,
                        "delta": 1,
                        "isolated": False,
                    },
                    {
                        "format": family,
                        "fixture": "f1",
                        "marker": "[INST]",
                        "status": "ok",
                        "baseline": 2,
                        "probed": 2,
                        "delta": 0,
                        "isolated": True,
                    },
                ],
            },
        }

    def test_matrix_section_renders(self):
        """Test matrix section renders."""
        rendered = reporting.render_html(self._result())
        self.assertIn('id="marker-isolation"', rendered)
        self.assertIn("Chat-marker isolation", rendered)
        self.assertIn("smuggled", rendered)
        self.assertIn("isolated", rendered)

    def test_absent_when_unavailable(self):
        """Test absent when unavailable."""
        result = self._result()
        result["marker_isolation"] = {"status": "unavailable"}
        rendered = reporting.render_html(result)
        self.assertNotIn('id="marker-isolation"', rendered)

    def test_hostile_names_escaped(self):
        """Test hostile names escaped."""
        hostile = '"><img src=x onerror=alert(1)>'
        rendered = reporting.render_html(self._result(family=hostile))
        self.assertNotIn(hostile, rendered)
        self.assertIn(html.escape(hostile), rendered)

    def test_hostile_rates_do_not_crash(self):
        """Test hostile rates do not crash."""
        # Non-numeric rates in a hand-crafted payload degrade to "n/a"
        # instead of raising inside the format spec.
        result = self._result()
        check = result["marker_isolation"]
        check["overall_isolation_rate"] = "high"
        check["family_scores"]["chatml"]["isolation_rate"] = [1, 2]
        rendered = reporting.render_html(result)
        self.assertIn('id="marker-isolation"', rendered)
        # Same crash class in the boundary-canonicality section (cycle 134).
        result2 = self._result()
        result2["boundary_audit"] = {
            "available": True,
            "n_usable": 1,
            "n_problem": 0,
            "overall_rate": "zero",
            "per_candidate": {},
            "per_family": {"chatml": {"n_usable": 1, "n_problem": 0, "rate": "zero"}},
        }
        rendered2 = reporting.render_html(result2)
        self.assertIn('id="boundary-canonicality"', rendered2)


class MarkerIsolationMutationGateTests(unittest.TestCase):
    """4 committed mutants on control_tokens.py; the suite must kill each."""

    MUTANTS = {
        # M1: every probe reports isolated.
        "m1": (
            "isolated=delta == 0",
            "isolated=True",
        ),
        # M2: control-token count off by one.
        "m2": (
            "return sum(1 for i in ids if i in special_ids)",
            "return sum(1 for i in ids if i in special_ids) + 1",
        ),
        # M3: hostile payload injected into the FIRST user message.
        "m3": (
            'for i, m in enumerate(msgs):\n        if m.role == "user":\n            target = i',
            'for i, m in enumerate(msgs):\n        if m.role == "user":\n'
            "            target = i\n            break",
        ),
        # M4: delta sign flipped.
        "m4": (
            "delta = probed - baseline",
            "delta = baseline - probed",
        ),
    }

    def test_mutants_killed(self):
        """Test mutants killed."""
        import subprocess
        import sys

        src = Path(CT.__file__)
        original = src.read_text(encoding="utf-8")
        killed = []
        survived = []
        # Fast, targeted test subset per mutant (keeps the gate under a
        # minute); each subset must fail on its mutant and pass on clean.
        base = "tests/test_cycle135_marker_isolation.py"
        targets = {
            "m1": f"{base}::ProbeFormatTests::test_golden_matrix",
            "m2": f"{base}::CountControlTokensTests",
            "m3": f"{base}::InjectHostileTests::test_appends_to_last_user_message",
            "m4": f"{base}::CheckIsolationTests::test_smuggled_marker_not_isolated",
        }
        for name, (old, new) in self.MUTANTS.items():
            self.assertIn(old, original, f"mutant {name} anchor missing")
            src.write_text(original.replace(old, new, 1), encoding="utf-8")
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        "-p",
                        "no:cacheprovider",
                        "-x",
                        "--tb=no",
                        targets[name],
                    ],
                    cwd=str(Path(__file__).resolve().parent.parent),
                    capture_output=True,
                    timeout=300,
                    check=False,
                )
            finally:
                src.write_text(original, encoding="utf-8")
                # Invalidate the bytecode cache: a same-size mutant (e.g.
                # the M4 sign flip) can leave a .pyc whose (mtime, size)
                # still validates against the reverted source, poisoning
                # later imports in this process run with mutant bytecode.
                for pyc in src.parent.glob("__pycache__/control_tokens.*.pyc"):
                    pyc.unlink(missing_ok=True)
            if proc.returncode != 0:
                killed.append(name)
            else:
                survived.append(name)
        self.assertEqual(survived, [], f"mutants survived: {survived}")
        self.assertEqual(sorted(killed), sorted(self.MUTANTS))


if __name__ == "__main__":
    unittest.main()
