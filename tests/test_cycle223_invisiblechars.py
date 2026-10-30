"""Cycle 223: invisible-character insertion probe
(wave-12 design round 1, BUILD-as-amended, coordinator
#72).

Spec citations are to docs/INVISIBLECHARS_DESIGN.md (the
candidate text as amended) and to the adjudication's ten
amendments A1-A10 in the design doc's adjudication
section.

Every amendment has named evidence here: the digest-to-
integer conversion and the number derivation are pinned in
executed values (A1, build debt 3); the operating
characteristics are re-derived by exact enumeration under
the marginal convention and by exact per-case bivariate
quadrature for the paired rows, reproducing the corrected
figures including the family-wise paired 0.4139 (A2,
build debt 4); the guard battery carries an ids dimension
proving the completion domain is the id-divergent domain
(A3, build debt 1); the incomplete cap is re-derived at
its amended value and the worlds are re-executed at it
(A4, build debt 2); the decomposition is pinned as the
ordered decision list with its precedence witnesses (A5);
the visible control's battery is corrected, its form_only
empty by construction (A6); the wrong-value class is
scoped to the frozen grammar, out-of-grammar value-shaped
text landing in non_answer (A7); MIN_D is shown to sit on
the [9, 16] price plateau (A8); the printed claims are
checked at sentence level against the module (A9); and
the shipped operating characteristics are the reproduced
ones (A10).
"""

from __future__ import annotations

import copy
import re
import threading
import unittest
from hashlib import sha256
from math import comb
from statistics import NormalDist
from unittest import mock

from model_protocol_lab import diffreports, experiments, reporting
from model_protocol_lab import invisiblechars as ic
from model_protocol_lab.core import Format, Generation

_CHATML = Format("chatml", "", "<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_end|>\n")
_LLAMA3 = Format(
    "llama3",
    "<|begin_of_text|>",
    "<|start_header_id|>user<|end_header_id|>\n\n",
    "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "<|eot_id|>",
)
_CPS = ("\u200b", "\ufe0f", "\U000e0001", "\u00ad", "\u2060")
_NORM = NormalDist()


class _WordBackend:
    """Word-token scripted backend.

    Modes: ``perfect`` (the expected value read out of the
    fact sentence with the inserted code points stripped,
    boundary on), ``echo`` (the value exactly as rendered,
    inserted code points included), ``garbage`` (a fixed
    non-answer), ``noboundary`` (the expected answer with
    no boundary token), ``explode`` (generate raises),
    ``long_nonascii`` (a 2,048-character non-ASCII answer).
    With ``strip=True`` the encoder removes the five
    invisible code points before tokenising, so every
    invisible arm is token-identical to the control (the
    ids-identical regime).
    """

    fingerprint = "sha256:test-invisible-chars"

    def __init__(self, mode: str = "perfect", strip: bool = False) -> None:
        """Init  ."""
        self.mode = mode
        self.strip = strip
        self.calls = 0
        self.vocab: dict[str, int] = {}
        self.rev: dict[int, str] = {}

    def encode(self, text):
        """Encode."""
        if self.strip:
            for code_point in _CPS:
                text = text.replace(code_point, "")
        ids = []
        for token in text.replace("\n", " \n ").split(" "):
            if not token:
                continue
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 1
                self.rev[self.vocab[token]] = token
            ids.append(self.vocab[token])
        return tuple(ids)

    def decode(self, ids):
        """Decode."""
        return " ".join(self.rev.get(int(i), "?") for i in ids)

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        """Generate."""
        self.calls += 1
        if self.mode == "explode":
            raise RuntimeError("boom")
        text = self.decode(list(input_ids))
        match = re.search(r"inspection code is (\S+?)\.", text)
        raw = match.group(1) if match else "unknownword"
        if self.mode == "perfect":
            answer = raw
            for code_point in _CPS:
                answer = answer.replace(code_point, "")
        elif self.mode == "garbage":
            answer = "unknownword"
        elif self.mode == "long_nonascii":
            answer = "あ" * 2048
        else:
            answer = raw
        boundary = self.encode("endmark")[0]
        out = list(self.encode(answer))
        if self.mode != "noboundary":
            out.append(boundary)
        pad = boundary if self.mode != "noboundary" else self.encode("fillerword")[0]
        out = (out + [pad] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


def _payload(backend=None, formats=(_CHATML,), seed=0, **stage_kwargs):
    """Payload."""
    backend = backend if backend is not None else _WordBackend()
    stage_kwargs.setdefault("max_context", 8192)
    stage = experiments.invisible_chars_stage(
        backend,
        formats,
        boundary=tuple(backend.encode("endmark")),
        seed=seed,
        **stage_kwargs,
    )
    return experiments.invisible_chars_analysis(stage, tuple(f.name for f in formats))


def _cell(text=None, digest=None, boundary=True, tokens=100, verdict=None):
    """Cell."""
    if digest is None and text is not None:
        digest = ic.digest_text(text)
    if verdict is None:
        verdict = False
    return {
        "verdict": verdict,
        "boundary_found": boundary,
        "boundary_text_match": None,
        "input_tokens": tokens,
        "answer_digest": digest,
        "answer_text": text,
    }


_CASE0 = ic.generate_cases(0)[0]
_EXPECTED0 = _CASE0["expected"]
_DIGEST0 = ic.digest_text(_EXPECTED0)


class TestGeneratorFreeze(unittest.TestCase):
    """A1 + build debt 3: the generator's conversion and
    panel are frozen in executed values."""

    def test_generator_spec_text_pinned(self):
        """Generator spec text pinned."""
        self.assertIn("int.from_bytes(", ic.GENERATOR_SPEC)
        self.assertIn(".digest()[:8], 'big')", ic.GENERATOR_SPEC)
        self.assertIn("number = 1000 + draw % 9000", ic.GENERATOR_SPEC)
        self.assertIn("giving up after", ic.GENERATOR_SPEC)
        self.assertIn("1000 consecutive redraws", ic.GENERATOR_SPEC)

    def test_derive_conversion_pinned(self):
        """Derive conversion pinned."""
        material = b"0|invisiblechars|0|0|0"
        expected = int.from_bytes(sha256(material).digest()[:8], "big")
        self.assertEqual(ic._derive(0, 0, 0, 0), expected)
        self.assertEqual(ic._derive(0, 0, 0, 0), 4815527247134056197)
        # The number derivation: 1000 + draw % 9000.
        draw = ic._derive(0, 0, 1, 0)
        self.assertEqual(1000 + draw % 9000, 1366)
        self.assertEqual(ic.WORDS[ic._derive(0, 0, 0, 0) % len(ic.WORDS)], "vale")

    def test_seed0_panel_literals(self):
        """Seed0 panel literals."""
        cases = ic.generate_cases(0)
        self.assertEqual(len(cases), 24)
        self.assertEqual(cases[0]["value"], "vale-1366")
        self.assertEqual(cases[0]["neutral_indices"], [19, 0])
        self.assertEqual(cases[1]["value"], "pine-3887")
        self.assertEqual(cases[1]["neutral_indices"], [18, 3])
        self.assertEqual(cases[2]["value"], "sage-1338")
        self.assertEqual(cases[23]["value"], "ochre-8234")
        self.assertEqual(cases[23]["topic"], "birch")
        self.assertEqual(
            [case["fixture_id"] for case in cases],
            [f"invisiblechars-{i:02d}" for i in range(24)],
        )
        self.assertEqual([case["topic"] for case in cases], list(ic.TOPICS))

    def test_panel_distinct_and_deterministic(self):
        """Panel distinct and deterministic."""
        for seed in (0, 1, 7, 42, 20261030):
            cases = ic.generate_cases(seed)
            self.assertEqual(len({case["value"] for case in cases}), 24)
            self.assertEqual(cases, ic.generate_cases(seed))
            for case in cases:
                self.assertTrue(ic.validate_case(case, seed))

    def test_redraw_on_collision(self):
        """Redraw on collision."""
        original = ic._derive
        case0 = ic.generate_cases(0)[0]

        def wrapped(seed, index, slot, counter):
            """Wrapped."""
            if index == 1 and counter == 0:
                return original(seed, 0, slot, 0)
            return original(seed, index, slot, counter)

        with mock.patch.object(ic, "_derive", wrapped):
            case1 = ic.generate_cases(0)[1]
        self.assertNotEqual(case1["value"], case0["value"])
        word = ic.WORDS[original(0, 1, 0, 1) % len(ic.WORDS)]
        number = 1000 + original(0, 1, 1, 1) % 9000
        self.assertEqual(case1["value"], f"{word}-{number}")

    def test_redraw_exhaustion(self):
        """Redraw exhaustion."""
        with (
            mock.patch.object(ic, "_derive", lambda *args: 7),
            self.assertRaises(ic._RedrawExhausted),
        ):
            ic.generate_cases(0)

    def test_validate_case_rejects_tampering(self):
        """Validate case rejects tampering."""
        case = copy.deepcopy(ic.generate_cases(0)[0])
        self.assertTrue(ic.validate_case(case, 0))
        case["neutral_indices"] = [4, 4]
        self.assertFalse(ic.validate_case(case, 0))
        case = copy.deepcopy(ic.generate_cases(0)[0])
        case["expected"] = "vale-9999"
        self.assertFalse(ic.validate_case(case, 0))
        case = copy.deepcopy(ic.generate_cases(0)[0])
        case["topic"] = "summit"
        self.assertFalse(ic.validate_case(case, 0))

    def test_arm_content_structure(self):
        """Arm content structure."""
        for case in ic.generate_cases(0):
            control = ic.arm_content(case, "control")
            self.assertTrue(control.isascii())
            question = ic.question_text(case)
            for arm in ic.TREATED_ARMS:
                content = ic.arm_content(case, arm)
                code_point = ic.ARM_CODEPOINTS[arm]
                # Exactly two insertions; on the visible arm
                # the count includes the value's own hyphen
                # separator, which the control also carries.
                expected_count = 2 if arm in ic.INVISIBLE_ARMS else 3
                self.assertEqual(content.count(code_point), expected_count)
                self.assertTrue(content.endswith(question))
            for arm in ic.INVISIBLE_ARMS:
                content = ic.arm_content(case, arm)
                self.assertEqual(content.replace(ic.ARM_CODEPOINTS[arm], ""), control)
            fact = ic.fact_sentence(case)
            for arm in ic.INVISIBLE_ARMS:
                treated = ic.treated_fact(case, ic.ARM_CODEPOINTS[arm])
                self.assertEqual(len(treated), len(fact) + 2)
        panel = ic.generate_cases(0)
        for arm in ic.INVISIBLE_ARMS:
            total = sum(ic.arm_content(case, arm).count(ic.ARM_CODEPOINTS[arm]) for case in panel)
            self.assertEqual(total, 48)

    def test_treated_fact_sites(self):
        """Treated fact sites."""
        case = ic.generate_cases(0)[0]
        treated = ic.treated_fact(case, "\u200b")
        self.assertEqual(treated, "The ha\u200brbor inspection code is va\u200ble-1366.")

    def test_population_audit(self):
        """Population audit."""
        audit = ic.audit_arm_population()
        self.assertEqual(audit, ic.POPULATION_AUDIT)
        self.assertTrue(audit["ok"])
        for arm in ic.INVISIBLE_ARMS:
            entry = audit["arms"][arm]
            self.assertTrue(entry["satisfies_predicate"])
            self.assertIn(entry["category"], ("Cf", "Mn"))
            self.assertEqual(entry["combining_class"], 0)
            self.assertNotEqual(entry["name"], "")
        self.assertEqual(audit["arms"]["zwsp"]["code_point"], "U+200B")
        self.assertEqual(audit["arms"]["tag"]["code_point"], "U+E0001")
        self.assertEqual(audit["arms"]["visible_hyphen"]["category"], "Pd")
        self.assertFalse(audit["arms"]["visible_hyphen"]["satisfies_predicate"])

    def test_literal_copies_pinned(self):
        """Literal copies pinned."""
        from model_protocol_lab import canonicalform, density

        self.assertEqual(ic.WORDS, canonicalform.WORDS)
        self.assertEqual(ic.NEUTRAL_STATEMENTS, density.NEUTRAL_STATEMENTS)

    def test_protocol_fingerprint_stable(self):
        """Protocol fingerprint stable."""
        protocol = ic.protocol_object(0, "sha256:test")
        self.assertEqual(
            ic.protocol_fingerprint(0, "sha256:test"),
            ic.protocol_fingerprint(0, "sha256:test"),
        )
        self.assertNotEqual(
            ic.protocol_fingerprint(0, "sha256:test"),
            ic.protocol_fingerprint(1, "sha256:test"),
        )
        self.assertEqual(protocol["case_count"], 24)
        self.assertEqual(protocol["seed"], 0)
        self.assertEqual(protocol["population_audit"], ic.POPULATION_AUDIT)


class TestDecomposition(unittest.TestCase):
    """A5/A6/A7: the ordered decision list and its
    witnesses."""

    def test_pass_first(self):
        """Pass first."""
        cell = _cell(text=_EXPECTED0, verdict=True)
        self.assertEqual(ic.decompose_cell(cell, _EXPECTED0, _DIGEST0, "zwsp"), "pass")

    def test_boundary_only_by_digest(self):
        """Boundary only by digest."""
        cell = _cell(text=_EXPECTED0, boundary=False)
        self.assertEqual(ic.decompose_cell(cell, _EXPECTED0, _DIGEST0, "zwsp"), "boundary_only")

    def test_boundary_precedence_over_text(self):
        """Boundary precedence over text."""
        # A digest-equal answer is boundary_only even when
        # its text additionally carries the code point
        # off-site: the digest reading precedes every
        # text reading (A5).
        cell = _cell(digest=_DIGEST0, text=None, boundary=False)
        self.assertEqual(ic.decompose_cell(cell, _EXPECTED0, _DIGEST0, "zwsp"), "boundary_only")

    def test_wrong_value_in_grammar(self):
        """Wrong value in grammar."""
        cell = _cell(text="cedar-5678")
        self.assertEqual(ic.decompose_cell(cell, _EXPECTED0, _DIGEST0, "zwsp"), "wrong_value")

    def test_wrong_value_same_word_other_number(self):
        """Wrong value same word other number."""
        cell = _cell(text="vale-9999")
        self.assertEqual(ic.decompose_cell(cell, _EXPECTED0, _DIGEST0, "zwsp"), "wrong_value")

    def test_out_of_grammar_shapes_are_non_answer(self):
        """Out of grammar shapes are non answer."""
        for text in ("zebra-1234", "vale-0123", "VALE-1366", "vale-13660", "vale1366"):
            cell = _cell(text=text)
            self.assertEqual(
                ic.decompose_cell(cell, _EXPECTED0, _DIGEST0, "zwsp"),
                "non_answer",
                text,
            )

    def test_echo_invisible(self):
        """Echo invisible."""
        word, _, number = _EXPECTED0.partition("-")
        echo = word[:2] + "\u200b" + word[2:] + "-" + number
        cell = _cell(text=echo)
        self.assertEqual(ic.decompose_cell(cell, _EXPECTED0, _DIGEST0, "zwsp"), "echo_invisible")

    def test_form_only_offsite(self):
        """Form only offsite."""
        cell = _cell(text="vale-13\u200b66")
        self.assertEqual(ic.decompose_cell(cell, _EXPECTED0, _DIGEST0, "zwsp"), "form_only")

    def test_form_only_beats_echo_for_extra_offsite(self):
        """Form only beats echo for extra offsite."""
        # Site echo plus an additional off-site code point
        # is not the exact site echo, and stripping recovers
        # the expected value: form_only (A5 witness).
        word, _, number = _EXPECTED0.partition("-")
        text = word[:2] + "\u200b" + word[2:] + "-13" + "\u200b" + "66"
        self.assertEqual(number, "1366")
        cell = _cell(text=text)
        self.assertEqual(ic.decompose_cell(cell, _EXPECTED0, _DIGEST0, "zwsp"), "form_only")

    def test_visible_arm_battery(self):
        """Visible arm battery."""
        # A6's corrected battery: the single-hyphen form
        # and the doubled-hyphen form are both outside the
        # frozen grammar, hence non_answer; form_only is
        # empty by construction on the visible arm.
        for text in ("va-le-1366", "vale--1366"):
            cell = _cell(text=text)
            self.assertEqual(
                ic.decompose_cell(cell, _EXPECTED0, _DIGEST0, "visible_hyphen"),
                "non_answer",
                text,
            )
        cell = _cell(text="cedar-5678")
        self.assertEqual(
            ic.decompose_cell(cell, _EXPECTED0, _DIGEST0, "visible_hyphen"),
            "wrong_value",
        )

    def test_control_arm_has_no_echo_or_form(self):
        """Control arm has no echo or form."""
        cell = _cell(text="va\u200ble-1366")
        self.assertEqual(ic.decompose_cell(cell, _EXPECTED0, _DIGEST0, "control"), "non_answer")

    def test_missing_text_is_non_answer(self):
        """Missing text is non answer."""
        cell = _cell(text=None, digest=ic.digest_text("something else"))
        self.assertEqual(ic.decompose_cell(cell, _EXPECTED0, _DIGEST0, "zwsp"), "non_answer")

    def test_pair_decomposition(self):
        """Pair decomposition."""
        control = _cell(text=_EXPECTED0, verdict=True)
        same = _cell(text=_EXPECTED0, boundary=False)
        self.assertEqual(ic.decompose_pair(control, same), "boundary_only")
        other = _cell(text="cedar-5678")
        self.assertEqual(ic.decompose_pair(control, other), "exact_only")
        no_boundary = _cell(text="cedar-5678", boundary=False)
        self.assertEqual(ic.decompose_pair(control, no_boundary), "mixed")


def _base_rule(d: int, j: int) -> str:
    """The unguarded label rule (test-local oracle piece)."""
    if d < ic.MIN_D:
        return "insufficient"
    if j >= ic.MIN_J:
        return "discordant_asymmetric"
    if j == 0:
        return "no_discordance_observed"
    return "below_threshold"


def _oracle_label(d: int, j: int, k: int) -> str:
    """Independent worst-case-completion oracle for
    classify_headline: enumerate every completion of the
    k missing pairs (each lands as control-pass+wrong,
    control-pass+other, or control-fail) and require the
    acquittal to be completion-stable; fires issue on the
    observed counts; the cap and the D floor refuse."""
    if k > ic.MAX_INCOMPLETE:
        return "insufficient"
    if d < ic.MIN_D:
        return "insufficient"
    if j >= ic.MIN_J:
        return "discordant_asymmetric"
    outcomes = set()
    for a in range(k + 1):
        for b in range(k + 1 - a):
            outcomes.add(_base_rule(d + a + b, j + a))
    if len(outcomes) == 1:
        only = outcomes.pop()
        if only in ("no_discordance_observed", "below_threshold"):
            return only
    return "insufficient"


class TestClassifyGuard(unittest.TestCase):
    """A3/A4: the guarded headline against an independent
    completion oracle."""

    def test_classify_matches_oracle_on_grid(self):
        """Classify matches oracle on grid."""
        for d in range(0, 16):
            for j in range(0, 9):
                for k in range(0, 9):
                    self.assertEqual(
                        ic.classify_headline(d, j, k),
                        _oracle_label(d, j, k),
                        (d, j, k),
                    )

    def test_named_rows(self):
        """Named rows."""
        self.assertEqual(ic.classify_headline(24, 6, 0), "discordant_asymmetric")
        self.assertEqual(ic.classify_headline(10, 6, 6), "discordant_asymmetric")
        self.assertEqual(ic.classify_headline(24, 0, 0), "no_discordance_observed")
        self.assertEqual(ic.classify_headline(24, 0, 1), "insufficient")
        self.assertEqual(ic.classify_headline(24, 5, 0), "below_threshold")
        self.assertEqual(ic.classify_headline(24, 5, 1), "insufficient")
        self.assertEqual(ic.classify_headline(24, 1, 4), "below_threshold")
        self.assertEqual(ic.classify_headline(9, 6, 0), "insufficient")
        self.assertEqual(ic.classify_headline(24, 6, 7), "insufficient")
        self.assertEqual(ic.MIN_D, 10)
        self.assertEqual(ic.MIN_J, 6)
        self.assertEqual(ic.MAX_INCOMPLETE, 6)


def _record(case, cells, details, identical_arms=()):
    """Record."""
    digests = {arm: f"digest-{arm}" for arm in ic.ARM_KEYS}
    digest_control = digests["control"]
    for arm in identical_arms:
        digests[arm] = digest_control
    return {
        "fixture": case["fixture_id"],
        "topic": case["topic"],
        "expected_digest": ic.digest_text(case["expected"]),
        "content_digests": dict.fromkeys(ic.ARM_KEYS, "x" * 64),
        "arm_tokens": dict.fromkeys(ic.ARM_KEYS, 100),
        "id_digests": digests,
        "ids_identical": {arm: arm in identical_arms for arm in ic.TREATED_ARMS},
        "id_length_delta": dict.fromkeys(ic.TREATED_ARMS, 0),
        "cells": cells,
        "cell_details": details,
    }


class TestGuardBatteryIdsDimension(unittest.TestCase):
    """Build debt 1 / A3: the completion domain is the
    id-divergent domain; ids-identical incomplete cases
    never enter the completion."""

    def _cells(self, case, verdict=True, missing=()):
        """Cells."""
        cells = {}
        details = {}
        for arm in ic.ARM_KEYS:
            if arm in missing:
                cells[arm] = None
                details[arm] = "arm_evaluation_failed"
            else:
                expected = case["expected"]
                cells[arm] = _cell(
                    text=expected,
                    verdict=verdict,
                    tokens=100,
                )
                details[arm] = ""
        return cells, details

    def test_ids_identical_incomplete_never_enters(self):
        """Ids identical incomplete never enters."""
        cases = ic.generate_cases(0)
        records = []
        for case in cases:
            cells, details = self._cells(case, missing=("zwsp",))
            records.append(_record(case, cells, details, identical_arms=("zwsp",)))
        aggregate = ic.aggregate_candidate(records, cases)
        head = aggregate["headline"]["zwsp"]
        self.assertEqual(head["n_population"], 0)
        self.assertEqual(head["n_incomplete"], 0)
        self.assertEqual(head["label"], "insufficient")
        self.assertEqual(aggregate["ids"]["zwsp"]["n_ids_identical"], 24)
        # The visible arm, divergent by construction on the
        # same records, headlines normally.
        visible = aggregate["headline"]["visible_hyphen"]
        self.assertEqual(visible["n_population"], 24)
        self.assertEqual(visible["label"], "no_discordance_observed")

    def test_mixed_domain_counts_only_divergent_incompletion(self):
        """Mixed domain counts only divergent incompletion."""
        cases = ic.generate_cases(0)
        records = []
        for index, case in enumerate(cases):
            if index < 6:
                cells, details = self._cells(case, missing=("zwsp",))
                records.append(_record(case, cells, details, identical_arms=("zwsp",)))
            elif index < 8:
                cells, details = self._cells(case, missing=("zwsp",))
                records.append(_record(case, cells, details))
            else:
                cells, details = self._cells(case)
                records.append(_record(case, cells, details))
        aggregate = ic.aggregate_candidate(records, cases)
        head = aggregate["headline"]["zwsp"]
        self.assertEqual(head["n_population"], 16)
        self.assertEqual(head["n_incomplete"], 2)
        self.assertEqual(head["label"], "insufficient")
        self.assertEqual(aggregate["n_incomplete"], 8)


def _worlds(cap_trials: int = 4000, seed: int = 20261030) -> dict[str, dict[str, float]]:
    """Worlds A-C re-executed against the module's
    classifier (build debt 2, at the amended cap)."""
    import random

    rng = random.Random(seed)  # noqa: S311 -- deterministic test seed
    results: dict[str, dict[str, float]] = {}
    for world in ("A", "B", "C"):
        counts = {"guarded_fire": 0, "guarded_below": 0, "unguarded_fire": 0, "unguarded_below": 0}
        for _ in range(cap_trials):
            cpass = [rng.random() < 0.90 for _ in range(24)]
            q = 0.10 if world == "A" else 0.50
            afail = [cp and rng.random() < q for cp in cpass]
            if world == "A":
                inc = [i < 12 and cpass[i] and not afail[i] for i in range(24)]
            elif world == "B":
                inc = [afail[i] and rng.random() < 0.6 for i in range(24)]
            else:
                inc = [rng.random() < 0.15 for _ in range(24)]
            d = sum(1 for i in range(24) if cpass[i] and not inc[i])
            j = sum(1 for i in range(24) if afail[i] and not inc[i])
            n = sum(inc)
            guarded = ic.classify_headline(d, j, n)
            unguarded = _base_rule(d, j)
            counts["guarded_fire"] += guarded == "discordant_asymmetric"
            counts["guarded_below"] += guarded == "below_threshold"
            counts["unguarded_fire"] += unguarded == "discordant_asymmetric"
            counts["unguarded_below"] += unguarded == "below_threshold"
        results[world] = {key: value / cap_trials for key, value in counts.items()}
    return results


class TestWorldsAtAmendedCap(unittest.TestCase):
    """Build debt 2: worlds A-C at the amended cap (>6)."""

    def test_worlds(self):
        """Worlds."""
        worlds = _worlds()
        # World A (benign incompletion, tiny true effect):
        # the guarded fire stays at the floor (adjudicated
        # 0.0014 at the amended cap).
        self.assertLess(worlds["A"]["guarded_fire"], 0.01)
        # World B (60 percent of discordant cells lost):
        # the guarded acquittal nearly vanishes (0.0133)
        # against the unguarded one (0.7285).
        self.assertLess(worlds["B"]["guarded_below"], 0.05)
        self.assertGreater(worlds["B"]["unguarded_below"], 0.5)
        # World C (uniform incompletion, licensed drop):
        # the guarded fire survives (0.8932).
        self.assertGreater(worlds["C"]["guarded_fire"], 0.80)
        self.assertLessEqual(worlds["C"]["guarded_fire"], worlds["C"]["unguarded_fire"] + 0.02)


def _fire_independent(a: float, q: float, n: int = 24, min_d: int = 10) -> float:
    """Exact fire probability under the marginal
    (independent-pair) convention: D ~ Bin(n, a),
    J | D ~ Bin(D, q)."""
    total = 0.0
    for d in range(min_d, n + 1):
        p_d = comb(n, d) * a**d * (1 - a) ** (n - d)
        for j in range(ic.MIN_J, d + 1):
            total += p_d * comb(d, j) * q**j * (1 - q) ** (d - j)
    return total


def _transplant_fire(n: int, pc: float, pa: float) -> float:
    """Two-directional transplant (the sibling
    canonical-form construction, as executed in the design
    evidence): independent control/arm outcomes per case;
    fire iff either discordant count reaches 2 (the
    equal-count clause of the executed computation is
    subsumed by that threshold) -- evaluated here by the
    exact dynamic program over (control-only, arm-only)
    counts."""
    probs = (pc * pa, pc * (1 - pa), (1 - pc) * pa, (1 - pc) * (1 - pa))
    dp = {(0, 0): 1.0}
    for _ in range(n):
        nxt: dict[tuple[int, int], float] = {}
        for (c, a), w in dp.items():
            for dc, da, pi in (
                (0, 0, probs[0]),
                (1, 0, probs[1]),
                (0, 1, probs[2]),
                (0, 0, probs[3]),
            ):
                key = (c + dc, a + da)
                nxt[key] = nxt.get(key, 0.0) + w * pi
        dp = nxt
    return sum(w for (c, a), w in dp.items() if (c == a and c >= 2) or max(c, a) >= 2)


def _phi2(h: float, k: float, rho: float) -> float:
    """Bivariate standard normal CDF by 1-D quadrature."""
    steps = 4000
    lo = -8.0
    if h <= lo:
        return 0.0
    width = (h - lo) / steps
    scale = (1 - rho * rho) ** 0.5

    def integrand(x: float) -> float:
        """Integrand."""
        return _NORM.pdf(x) * _NORM.cdf((k - rho * x) / scale)

    total = integrand(lo) + integrand(h)
    for i in range(1, steps):
        total += (4 if i % 2 else 2) * integrand(lo + i * width)
    return total * width / 3


def _fire_paired(pc: float, pa: float, rho: float = 0.5, n: int = 24) -> float:
    """Exact fire probability under the per-case copula:
    pJ = pc - Phi2(tc, ta; rho); the trinomial sum over
    (wrong pairs, both-pass pairs, control-fail cases)."""
    p_both = _phi2(_NORM.inv_cdf(pc), _NORM.inv_cdf(pa), rho)
    p_j = pc - p_both
    p_c = 1 - pc
    total = 0.0
    for j in range(ic.MIN_J, n + 1):
        for b in range(max(0, ic.MIN_D - j), n + 1 - j):
            c = n - j - b
            total += comb(n, j) * comb(n - j, b) * p_j**j * p_both**b * p_c**c
    return total


class TestOperatingCharacteristics(unittest.TestCase):
    """Build debt 4 / A2/A8/A10: the adjudicated figures,
    re-derived exactly."""

    def test_independent_rows(self):
        """Independent rows."""
        # Rows are (control-pass rate, conditional wrong-
        # value rate q) under the marginal convention of
        # amendment A2: q is the arm's wrong-value marginal
        # rate, which under independence equals the rate
        # conditional on a control pass.
        self.assertEqual(round(_fire_independent(0.90, 0.10), 4), 0.0173)
        self.assertEqual(round(_fire_independent(0.80, 0.20), 4), 0.1743)
        self.assertEqual(round(_fire_independent(0.50, 0.50), 4), 0.5503)
        self.assertEqual(round(_fire_independent(0.90, 0.20), 4), 0.2542)
        self.assertEqual(round(_fire_independent(0.90, 0.35), 4), 0.8159)
        self.assertEqual(round(_fire_independent(0.85, 0.40), 4), 0.8761)
        self.assertEqual(round(_fire_independent(0.90, 0.60), 4), 0.9990)

    def test_panel_sweep_and_sensitivity_rows(self):
        """Panel sweep and sensitivity rows."""
        self.assertEqual(round(_fire_independent(0.90, 0.35, n=16), 4), 0.3904)
        self.assertEqual(round(_fire_independent(0.90, 0.35, n=20), 4), 0.6395)
        self.assertEqual(round(_fire_independent(0.90, 0.10, n=18), 4), 0.0038)
        self.assertEqual(round(_fire_independent(0.90, 0.35, n=18), 4), 0.5214)

    def test_transplant_row(self):
        """Transplant row."""
        self.assertEqual(round(_transplant_fire(24, 0.90, 0.90), 4), 0.8897)

    def test_paired_rows(self):
        """Paired rows."""
        self.assertEqual(round(_fire_paired(0.90, 0.90), 4), 0.0044)
        self.assertEqual(round(_fire_paired(0.90, 0.65), 4), 0.6981)

    def test_familywise_rows(self):
        """Familywise rows."""
        price = _fire_independent(0.90, 0.10)
        self.assertEqual(round(120 * price, 2), 2.07)
        self.assertEqual(round(1 - (1 - price) ** 120, 4), 0.8764)
        self.assertEqual(round(144 * price, 2), 2.49)
        self.assertEqual(round(1 - (1 - price) ** 144, 4), 0.9187)
        paired = _fire_paired(0.90, 0.90)
        self.assertEqual(round(120 * paired, 2), 0.53)
        # A9(v): the printed figure uses the exact paired
        # rate (0.4139), not the design-phase Monte Carlo
        # point estimate (0.4144).
        self.assertEqual(round(1 - (1 - paired) ** 120, 4), 0.4139)
        self.assertEqual(round(120 * _fire_independent(0.50, 0.50), 2), 66.04)

    def test_min_d_plateau(self):
        """Min d plateau."""
        prices = {d: _fire_independent(0.90, 0.10, min_d=d) for d in (9, 10, 16)}
        self.assertLess(abs(prices[9] - prices[10]), 0.001)
        self.assertLess(abs(prices[16] - prices[10]), 0.001)

    def test_module_dict_matches_reproduction(self):
        """Module dict matches reproduction."""
        oc = ic.OPERATING_CHARACTERISTICS
        self.assertEqual(oc["fire_competent_independent"], 0.0173)
        self.assertEqual(oc["fire_competent_paired"], 0.0044)
        self.assertEqual(oc["power_drop25_base90"], 0.8159)
        self.assertEqual(oc["power_drop25_base85"], 0.8761)
        self.assertEqual(oc["power_drop10_base90"], 0.2542)
        self.assertEqual(oc["power_drop25_paired"], 0.6981)
        self.assertEqual(oc["sensitivity18_drop25"], 0.5214)
        self.assertEqual(oc["familywise_paired_any_120"], 0.4139)
        self.assertEqual(oc["familywise_competent_expected_120"], 2.07)


class TestStagePayloads(unittest.TestCase):
    """Stage integration through scripted backends."""

    def test_perfect_payload(self):
        """Perfect payload."""
        payload = _payload()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["n_cases"], 24)
        self.assertEqual(payload["n_candidates"], 1)
        entry = payload["candidates"]["chatml"]
        for arm in ("control", *ic.INVISIBLE_ARMS):
            self.assertEqual(entry["arms"][arm]["rate"], 1.0)
            self.assertEqual(entry["arms"][arm]["decomposition"]["pass"], 24)
        # The perfect reader answers the visible arm's
        # treated value as rendered, which is not the
        # expected value: rate 0, all non_answer.
        visible = entry["arms"]["visible_hyphen"]
        self.assertEqual(visible["rate"], 0.0)
        self.assertEqual(visible["decomposition"]["non_answer"], 24)
        for arm in ic.TREATED_ARMS:
            head = entry["headline"][arm]
            self.assertEqual(head["label"], "no_discordance_observed")
            self.assertEqual(head["n_population"], 24)
            self.assertEqual(head["control_passes"], 24)
        self.assertTrue(diffreports._valid_invisible_chars(payload))

    def test_echo_payload(self):
        """Echo payload."""
        payload = _payload(_WordBackend("echo"))
        entry = payload["candidates"]["chatml"]
        for arm in ic.INVISIBLE_ARMS:
            decomposition = entry["arms"][arm]["decomposition"]
            self.assertEqual(decomposition["echo_invisible"], 24)
            head = entry["headline"][arm]
            self.assertEqual(head["wrong_values"], 0)
            self.assertEqual(head["label"], "no_discordance_observed")
        visible = entry["arms"]["visible_hyphen"]["decomposition"]
        self.assertEqual(visible["non_answer"], 24)
        self.assertEqual(visible["form_only"], 0)
        self.assertEqual(visible["echo_invisible"], 0)
        self.assertTrue(diffreports._valid_invisible_chars(payload))

    def test_strip_backend_ids_regime(self):
        """Strip backend ids regime."""
        payload = _payload(_WordBackend("perfect", strip=True))
        entry = payload["candidates"]["chatml"]
        for arm in ic.INVISIBLE_ARMS:
            self.assertEqual(entry["ids"][arm]["n_ids_identical"], 24)
            head = entry["headline"][arm]
            self.assertEqual(head["n_population"], 0)
            self.assertEqual(head["label"], "insufficient")
        visible = entry["headline"]["visible_hyphen"]
        self.assertEqual(visible["n_population"], 24)
        self.assertEqual(visible["label"], "no_discordance_observed")
        self.assertTrue(diffreports._valid_invisible_chars(payload))

    def test_garbage_payload(self):
        """Garbage payload."""
        payload = _payload(_WordBackend("garbage"))
        entry = payload["candidates"]["chatml"]
        for arm in ic.TREATED_ARMS:
            head = entry["headline"][arm]
            self.assertEqual(head["control_passes"], 0)
            self.assertEqual(head["label"], "insufficient")
            self.assertEqual(head["n_non_answer"], 24)
        self.assertTrue(diffreports._valid_invisible_chars(payload))

    def test_noboundary_payload(self):
        """Noboundary payload."""
        payload = _payload(_WordBackend("noboundary"))
        entry = payload["candidates"]["chatml"]
        # With no boundary token anywhere, evaluate takes
        # the whole padded output as the answer, so every
        # cell's text is the answer plus filler and lands
        # in non_answer (the sibling suites' observation);
        # true boundary_only cells are exercised at the
        # aggregate level below.
        for arm in ic.ARM_KEYS:
            decomposition = entry["arms"][arm]["decomposition"]
            self.assertEqual(decomposition["non_answer"], 24)
        self.assertTrue(diffreports._valid_invisible_chars(payload))

    def test_boundary_only_at_aggregate_level(self):
        """Boundary only at aggregate level."""
        cases = ic.generate_cases(0)
        records = []
        for case in cases:
            cells = {}
            details = {}
            for arm in ic.ARM_KEYS:
                cells[arm] = _cell(text=case["expected"], boundary=False, tokens=100)
                details[arm] = ""
            records.append(_record(case, cells, details))
        aggregate = ic.aggregate_candidate(records, cases)
        for arm in ic.ARM_KEYS:
            self.assertEqual(aggregate["arms"][arm]["decomposition"]["boundary_only"], 24)
        head = aggregate["headline"]["zwsp"]
        self.assertEqual(head["pair_decomposition"]["boundary_only"], 0)
        self.assertEqual(head["label"], "insufficient")

    def test_explode_payload(self):
        """Explode payload."""
        payload = _payload(_WordBackend("explode"))
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["n_incomplete"], 24)
        record = entry["records"][0]
        self.assertEqual(record["cell_details"]["zwsp"], "arm_evaluation_failed")
        self.assertIsNone(record["cells"]["zwsp"])
        self.assertEqual(entry["headline"]["zwsp"]["label"], "insufficient")
        self.assertTrue(diffreports._valid_invisible_chars(payload))

    def test_context_exceeded_payload(self):
        """Context exceeded payload."""
        payload = _payload(max_context=10)
        entry = payload["candidates"]["chatml"]
        record = entry["records"][0]
        self.assertEqual(record["cell_details"]["control"], "arm_context_exceeded")
        self.assertEqual(entry["headline"]["zwsp"]["label"], "insufficient")
        self.assertTrue(diffreports._valid_invisible_chars(payload))

    def test_cancelled_stage(self):
        """Cancelled stage."""
        backend = _WordBackend()
        cancel = threading.Event()
        cancel.set()
        stage = experiments.invisible_chars_stage(
            backend,
            (_CHATML,),
            boundary=tuple(backend.encode("endmark")),
            seed=0,
            max_context=8192,
            cancel=cancel,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")
        analysis = experiments.invisible_chars_analysis(stage, ("chatml",))
        self.assertFalse(analysis["available"])
        self.assertEqual(analysis["detail"], "cancelled")

    def test_screen_failure_from_tampered_generator(self):
        """Screen failure from tampered generator."""
        cases = [dict(case) for case in ic.generate_cases(0)]
        cases[3]["neutral_indices"] = [2, 2]
        with mock.patch.object(ic, "generate_cases", lambda seed: tuple(cases)):
            payload = _payload()
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "invisible_chars_screen_failed")

    def test_redraw_exhaustion_fails_closed(self):
        """Redraw exhaustion fails closed."""
        with mock.patch.object(ic, "_derive", lambda *args: 7):
            stage = experiments.invisible_chars_stage(
                _WordBackend(),
                (_CHATML,),
                boundary=(1,),
                seed=0,
                max_context=8192,
            )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "invisible_chars_screen_failed")
        self.assertEqual(stage["audit"]["by_reason"], {"redraw_exhausted": 1})

    def test_record_budget_exceeded(self):
        """Record budget exceeded."""
        formats = tuple(
            Format(f"fmt{i}", "", "<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_end|>\n")
            for i in range(6)
        )
        payload = _payload(_WordBackend("long_nonascii"), formats=formats)
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "invisible_chars_record_budget_exceeded")

    def test_two_format_payload(self):
        """Two format payload."""
        payload = _payload(formats=(_CHATML, _LLAMA3))
        self.assertEqual(payload["n_candidates"], 2)
        self.assertTrue(diffreports._valid_invisible_chars(payload))


class TestMirrorValidation(unittest.TestCase):
    """The diffreports mirror: acceptance and tamper
    rejection."""

    def setUp(self):
        """Set up."""
        self.payload = _payload(_WordBackend("echo"))

    def test_valid_payload_accepted(self):
        """Valid payload accepted."""
        self.assertTrue(diffreports._valid_invisible_chars(self.payload))

    def test_label_tamper_rejected(self):
        """Label tamper rejected."""
        payload = copy.deepcopy(self.payload)
        payload["candidates"]["chatml"]["headline"]["zwsp"]["label"] = "discordant_asymmetric"
        self.assertFalse(diffreports._valid_invisible_chars(payload))

    def test_count_tamper_rejected(self):
        """Count tamper rejected."""
        payload = copy.deepcopy(self.payload)
        payload["candidates"]["chatml"]["headline"]["zwsp"]["wrong_values"] += 1
        self.assertFalse(diffreports._valid_invisible_chars(payload))

    def test_ids_flag_tamper_rejected(self):
        """Ids flag tamper rejected."""
        payload = copy.deepcopy(self.payload)
        record = payload["candidates"]["chatml"]["records"][0]
        record["ids_identical"]["zwsp"] = not record["ids_identical"]["zwsp"]
        self.assertFalse(diffreports._valid_invisible_chars(payload))

    def test_content_digest_tamper_rejected(self):
        """Content digest tamper rejected."""
        payload = copy.deepcopy(self.payload)
        record = payload["candidates"]["chatml"]["records"][0]
        record["content_digests"]["wj"] = "0" * 64
        self.assertFalse(diffreports._valid_invisible_chars(payload))

    def test_answer_text_tamper_rejected(self):
        """Answer text tamper rejected."""
        payload = copy.deepcopy(self.payload)
        record = payload["candidates"]["chatml"]["records"][0]
        record["cells"]["control"]["answer_text"] = "cedar-5678"
        self.assertFalse(diffreports._valid_invisible_chars(payload))

    def test_input_tokens_tamper_rejected(self):
        """Input tokens tamper rejected."""
        payload = copy.deepcopy(self.payload)
        record = payload["candidates"]["chatml"]["records"][0]
        record["cells"]["control"]["input_tokens"] += 1
        self.assertFalse(diffreports._valid_invisible_chars(payload))

    def test_verdict_tamper_rejected(self):
        """Verdict tamper rejected."""
        payload = copy.deepcopy(self.payload)
        record = payload["candidates"]["chatml"]["records"][0]
        record["cells"]["control"]["verdict"] = False
        self.assertFalse(diffreports._valid_invisible_chars(payload))

    def test_fingerprint_tamper_rejected(self):
        """Fingerprint tamper rejected."""
        payload = copy.deepcopy(self.payload)
        payload["protocol"]["fingerprint"] = "0" * 64
        self.assertFalse(diffreports._valid_invisible_chars(payload))

    def test_oc_tamper_rejected(self):
        """Oc tamper rejected."""
        payload = copy.deepcopy(self.payload)
        payload["candidates"]["chatml"]["operating_characteristics"]["fire_competent_paired"] = 0.5
        self.assertFalse(diffreports._valid_invisible_chars(payload))

    def test_diff_self_unchanged(self):
        """Diff self unchanged."""
        diff = diffreports.diff_invisible_chars(self.payload, self.payload)
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])
        self.assertTrue(diff["comparable"])

    def test_diff_echo_vs_perfect_changed(self):
        """Diff echo vs perfect changed."""
        other = _payload(_WordBackend("perfect"))
        diff = diffreports.diff_invisible_chars(self.payload, other)
        self.assertTrue(diff["changed"])
        metrics = {row["metric"] for row in diff["candidate_drift"]}
        self.assertIn("zwsp_rate", metrics)

    def test_diff_missing_side(self):
        """Diff missing side."""
        diff = diffreports.diff_invisible_chars(self.payload, None)
        self.assertEqual(diff["available"], "old")
        self.assertFalse(diff["changed"])
        diff = diffreports.diff_invisible_chars(None, None)
        self.assertEqual(diff["available"], "neither")


class TestReportingSection(unittest.TestCase):
    """The rendered section: populated and absent."""

    def test_section_renders(self):
        """Section renders."""
        payload = _payload(_WordBackend("echo"))
        html = reporting._invisible_chars_section({"invisible_chars": payload})
        self.assertIn("Invisible characters", html)
        self.assertIn("no_discordance_observed", html)
        self.assertIn("U+200B", html)
        self.assertIn("U+E0001", html)
        self.assertIn("2.07", html)
        self.assertIn("0.4139", html)
        self.assertIn("echo_invisible", html)

    def test_section_absent(self):
        """Section absent."""
        self.assertEqual(reporting._invisible_chars_section({}), "")
        unavailable = {"available": False, "detail": "invisible_chars_unavailable"}
        self.assertEqual(reporting._invisible_chars_section({"invisible_chars": unavailable}), "")


if __name__ == "__main__":
    unittest.main()


def _gap_case():
    """Case."""
    return copy.deepcopy(ic.generate_cases(0)[0])


def _gap_record(case, cells, identical_arms=()):
    """Record."""
    return _record(case, cells, dict.fromkeys(ic.ARM_KEYS, ""), identical_arms)


class _FlakyCells(dict):
    """A cells mapping that drops one arm after two reads."""

    def get(self, key, default=None):  # noqa: D102 -- test double
        if key == "zwsp":
            self._reads = getattr(self, "_reads", 0) + 1
            if self._reads > 2:
                return None
        return super().get(key, default)


class TestCoverageGaps(unittest.TestCase):
    """Gap battery: the producer module's defensive branches,
    the stage's screen returns, and the reporting fallback,
    each exercised at its own line (cycle-174 gap-pass pattern).
    """

    def test_audit_none_codepoint_raises(self):
        """Audit."""
        with (
            mock.patch.dict(ic.ARM_CODEPOINTS, {"zwsp": None}),
            self.assertRaises(ValueError),
        ):
            ic.audit_arm_population()

    def test_audit_unassigned_codepoint(self):
        """Audit."""
        with mock.patch.dict(ic.ARM_CODEPOINTS, {"zwsp": "͸"}):
            audit = ic.audit_arm_population()
        self.assertFalse(audit["ok"])
        self.assertEqual(audit["arms"]["zwsp"]["name"], "")
        self.assertFalse(audit["arms"]["zwsp"]["satisfies_predicate"])

    def test_audit_wrong_category(self):
        """Audit."""
        with mock.patch.dict(ic.ARM_CODEPOINTS, {"zwsp": "a"}):
            audit = ic.audit_arm_population()
        self.assertFalse(audit["ok"])
        self.assertFalse(audit["arms"]["zwsp"]["satisfies_predicate"])

    def test_validate_id_out_of_range(self):
        """Validate."""
        case = _gap_case()
        case["fixture_id"] = "invisiblechars-99"
        self.assertFalse(ic.validate_case(case, 0))

    def test_validate_regeneration_mismatch(self):
        """Validate."""
        case = _gap_case()
        case["value"] = "vale-9999"
        self.assertFalse(ic.validate_case(case, 0))

    def test_validate_malformed_id(self):
        """Validate."""
        self.assertFalse(ic.validate_case({"fixture_id": "bogus"}, 0))

    def _validate_patched(self, case):
        with mock.patch.object(ic, "generate_case", lambda seed, index: case):
            return ic.validate_case(case, 0)

    def test_validate_belt_branches(self):
        """Validate."""
        base = _gap_case()
        wrong_topic = dict(base, topic="xxxxx")
        self.assertFalse(self._validate_patched(wrong_topic))
        bad_shape = dict(base, value="Vale-1366", expected="Vale-1366")
        self.assertFalse(self._validate_patched(bad_shape))
        bad_word = dict(base, value="zebra-1234", expected="zebra-1234")
        self.assertFalse(self._validate_patched(bad_word))
        bad_number = dict(base, value="vale-0999", expected="vale-0999")
        self.assertFalse(self._validate_patched(bad_number))
        collided = dict(base, neutral_indices=[0, 0])
        self.assertFalse(self._validate_patched(collided))
        out_of_range = dict(base, neutral_indices=[0, 99])
        self.assertFalse(self._validate_patched(out_of_range))

    def test_validate_short_topic_branch(self):
        """Validate."""
        case = dict(_gap_case(), topic="ab")
        topics = ("ab",) + ic.TOPICS[1:]
        with mock.patch.object(ic, "TOPICS", topics):
            self.assertFalse(self._validate_patched(case))

    def test_validate_nonascii_control_branch(self):
        """Validate."""
        statements = ("Résumé holds the room.",) + ic.NEUTRAL_STATEMENTS[1:]
        case = dict(_gap_case(), neutral_indices=[0, 1])
        with mock.patch.object(ic, "NEUTRAL_STATEMENTS", statements):
            self.assertFalse(self._validate_patched(case))

    def test_validate_none_codepoint_branches(self):
        """Validate."""
        with mock.patch.dict(ic.ARM_CODEPOINTS, {"zwsp": None}):
            self.assertFalse(self._validate_patched(_gap_case()))
        with mock.patch.dict(ic.ARM_CODEPOINTS, {"visible_hyphen": None}):
            self.assertFalse(self._validate_patched(_gap_case()))

    def test_validate_treated_shape_branches(self):
        """Validate."""
        with mock.patch.dict(ic.ARM_CODEPOINTS, {"visible_hyphen": ""}):
            self.assertFalse(self._validate_patched(_gap_case()))
        real_content = ic.arm_content

        def short_content(case, arm):
            """Short."""
            content = real_content(case, arm)
            if arm == "zwsp":
                return content.replace(ic.ARM_CODEPOINTS["zwsp"], "", 1)
            return content

        with mock.patch.object(ic, "arm_content", short_content):
            self.assertFalse(self._validate_patched(_gap_case()))

    def test_aggregate_hostile_record_shapes(self):
        """Aggregate."""
        case = _gap_case()
        passing = _gap_record(case, {"control": _cell(case["expected"], verdict=True)})
        no_cells = dict(passing, cells=[])
        aggregates = ic.aggregate_candidate([no_cells], [case])
        self.assertEqual(aggregates["headline"]["zwsp"]["n_population"], 0)
        both = _gap_record(
            case,
            {
                "control": _cell(case["expected"], verdict=True),
                "zwsp": _cell(case["expected"], verdict=True),
            },
        )
        no_details = dict(both, cell_details=[])
        aggregates = ic.aggregate_candidate([no_details], [case])
        self.assertEqual(aggregates["headline"]["zwsp"]["n_incomplete"], 0)
        no_ids = dict(passing, id_digests=[])
        aggregates = ic.aggregate_candidate([no_ids], [case])
        self.assertEqual(aggregates["ids"]["zwsp"]["n_ids_divergent"], 1)
        self.assertEqual(aggregates["headline"]["zwsp"]["n_population"], 0)
        none_tokens = dict(passing, arm_tokens={"control": 5, "zwsp": None, "vs16": 5})
        aggregates = ic.aggregate_candidate([none_tokens], [case])
        self.assertEqual(aggregates["ids"]["zwsp"]["id_length_delta"], [0])

    def test_aggregate_same_ids_flip(self):
        """Aggregate."""
        case = _gap_case()
        record = _gap_record(
            case,
            {
                "control": _cell(case["expected"], verdict=True),
                "zwsp": _cell("hmm", verdict=False),
            },
            identical_arms=("zwsp",),
        )
        aggregates = ic.aggregate_candidate([record], [case])
        self.assertEqual(aggregates["ids"]["zwsp"]["n_same_ids_verdict_flips"], 1)

    def test_aggregate_wrong_value_and_reverse_pairs(self):
        """Aggregate."""
        case = _gap_case()
        wrong = _gap_record(
            case,
            {
                "control": _cell(case["expected"], verdict=True),
                "zwsp": _cell("cedar-5678", verdict=False),
            },
        )
        headline = ic.aggregate_candidate([wrong], [case])["headline"]["zwsp"]
        self.assertEqual(headline["wrong_values"], 1)
        reverse = _gap_record(
            case,
            {
                "control": _cell("hmm", verdict=False),
                "zwsp": _cell(case["expected"], verdict=True),
            },
        )
        headline = ic.aggregate_candidate([reverse], [case])["headline"]["zwsp"]
        self.assertEqual(headline["n_reverse"], 1)

    def test_aggregate_flaky_cells_continue(self):
        """Aggregate."""
        case = _gap_case()
        record = _gap_record(
            case,
            {
                "control": _cell(case["expected"], verdict=True),
                "zwsp": _cell(case["expected"], verdict=True),
            },
        )
        record["cells"] = _FlakyCells(record["cells"])
        aggregates = ic.aggregate_candidate([record], [case])
        self.assertEqual(aggregates["headline"]["zwsp"]["n_population"], 1)
        self.assertEqual(aggregates["headline"]["zwsp"]["control_passes"], 0)

    def test_stage_formats_validation(self):
        """Stage."""
        backend = _WordBackend()
        with self.assertRaises(ValueError):
            experiments.invisible_chars_stage(backend, (), boundary=(1,), max_context=8192)
        with self.assertRaises(ValueError):
            experiments.invisible_chars_stage(
                backend, (_CHATML, _CHATML), boundary=(1,), max_context=8192
            )

    def test_stage_record_budget(self):
        """Stage."""
        backend = _WordBackend()
        formats = tuple(
            type(_CHATML)(
                f"fmt{i}",
                "",
                "<|im_start|>user\n",
                "<|im_start|>assistant\n",
                "<|im_end|>\n",
            )
            for i in range(25)
        )
        stage = experiments.invisible_chars_stage(backend, formats, boundary=(1,), max_context=8192)
        self.assertEqual(stage["detail"], "invisible_chars_record_budget_exceeded")

    def test_stage_redraw_exhaustion_screens(self):
        """Stage."""
        backend = _WordBackend()
        real_generate = ic.generate_cases
        calls = {"n": 0}

        def flaky_generate(seed):
            """Flaky."""
            calls["n"] += 1
            if calls["n"] > 2:
                raise ic._RedrawExhausted("second")
            return real_generate(seed)

        with mock.patch.object(ic, "generate_cases", side_effect=flaky_generate):
            stage = experiments.invisible_chars_stage(
                backend, (_CHATML,), boundary=(1,), max_context=8192
            )
        self.assertEqual(stage["audit"]["by_reason"], {"redraw_exhausted": 1})
        with mock.patch.object(ic, "protocol_object", side_effect=ic._RedrawExhausted("x")):
            stage = experiments.invisible_chars_stage(
                backend, (_CHATML,), boundary=(1,), max_context=8192
            )
        self.assertEqual(stage["detail"], "invisible_chars_screen_failed")
        self.assertEqual(stage["protocol"], {})

    def test_stage_arm_population_screen(self):
        """Stage."""
        backend = _WordBackend()
        with mock.patch.object(ic, "POPULATION_AUDIT", {"ok": False, "arms": {}}):
            stage = experiments.invisible_chars_stage(
                backend, (_CHATML,), boundary=(1,), max_context=8192
            )
        self.assertEqual(stage["audit"]["by_reason"].get("arm_population"), 1)
        self.assertEqual(stage["detail"], "invisible_chars_screen_failed")

    def test_stage_special_token_screen(self):
        """Stage."""
        backend = _WordBackend()
        with mock.patch(
            "model_protocol_lab.fixtures.audit_special_token_literals",
            return_value=["x"],
        ):
            stage = experiments.invisible_chars_stage(
                backend, (_CHATML,), boundary=(1,), max_context=8192
            )
        self.assertEqual(stage["audit"]["by_reason"], {"special_token_literal": 1})

    def test_stage_boundary_containment_screen(self):
        """Stage."""
        backend = _WordBackend()
        stage = experiments.invisible_chars_stage(
            backend,
            (_CHATML,),
            boundary=tuple(backend.encode("The")),
            max_context=8192,
        )
        self.assertEqual(stage["detail"], "invisible_chars_screen_failed")
        self.assertEqual(stage["audit"]["by_reason"], {})

    def test_stage_undecodable_boundary_proceeds(self):
        """Stage."""

        class _RaisingDecode(_WordBackend):
            def decode(self, token_ids):
                raise RuntimeError("no decode")

        payload = _payload(backend=_RaisingDecode())
        self.assertTrue(payload["available"])
        record = payload["candidates"]["chatml"]["records"][0]
        self.assertEqual(record["cell_details"]["zwsp"], "arm_evaluation_failed")

    def test_analysis_unavailable_candidate(self):
        """Analysis."""
        payload = _payload()
        stage = {
            "available": True,
            "protocol": payload["protocol"],
            "candidates": {"ghost": {"status": "unavailable", "detail": "gone", "records": []}},
        }
        out = experiments.invisible_chars_analysis(stage, ("ghost",))
        self.assertEqual(out["candidates"]["ghost"]["status"], "unavailable")
        self.assertEqual(out["candidates"]["ghost"]["detail"], "gone")

    def test_section_legend_fallback(self):
        """Reporting."""
        result = {"invisible_chars": {"available": True, "candidates": {}, "protocol": {}}}
        section = reporting._invisible_chars_section(result)
        self.assertIn("The frozen arms: zwsp, vs16, tag, softhyphen, wj, visible_hyphen.", section)


class TestMirrorDeepTampers(unittest.TestCase):
    """Deep mirror tampers: forgeries that pass the shallow
    bindings and fail in the recomputed aggregates."""

    def test_deep_tampers_rejected(self):
        """Mirror."""
        base = _payload(_WordBackend("echo"))

        def tamper_j(payload):
            payload["candidates"]["chatml"]["headline"]["zwsp"]["n_population"] += 1

        def tamper_pair(payload):
            payload["candidates"]["chatml"]["headline"]["zwsp"]["pair_decomposition"][
                "exact_only"
            ] += 1

        def tamper_same_ids(payload):
            payload["candidates"]["chatml"]["ids"]["zwsp"]["n_ids_identical"] += 1

        def tamper_control_rate(payload):
            payload["candidates"]["chatml"]["headline"]["zwsp"]["control_rate"] = 0.5

        for tamper in (tamper_j, tamper_pair, tamper_same_ids, tamper_control_rate):
            forged = copy.deepcopy(base)
            tamper(forged)
            self.assertFalse(diffreports._valid_invisible_chars(forged), tamper.__name__)
