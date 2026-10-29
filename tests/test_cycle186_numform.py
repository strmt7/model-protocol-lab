"""Cycle 186: numeric surface-form probe (``run --numform``).

Wave-7 rank 7, shipped as amended by the adversarial round
(A: DO NOT BUILD AS SPECIFIED, F1--F17; B: BUILDABLE AS
AMENDED, GB2--GB5). The probe asks whether the surface form
of a number -- plain digits, comma-grouped, narrow-no-break-
space-grouped, scientific -- changes exact answers on
generated integer tasks with the rendering held constant:
copy, compare, add, one value set per task rendered into
all four forms, canonical values stored in the clear so the
diff validator can ground every recomputation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import unittest
from threading import Event
from unittest import mock

from model_protocol_lab import diffreports as dr
from model_protocol_lab import numform as nf
from model_protocol_lab.cli import _run_numform_probe
from model_protocol_lab.confidentiality import mcnemar_exact_p
from model_protocol_lab.core import Format, Generation
from model_protocol_lab.diffreports import _valid_numform, diff_numform, diff_reports
from model_protocol_lab.experiments import numform_analysis, numform_stage
from model_protocol_lab.reporting import render_html

BOUNDARY_WORD = "<|im_end|>"
SPACE = "\u202f"


def _digits_of(value: int) -> str:
    return str(value)


def _carry_oracle(a: int, b: int) -> str:
    """Independent column-addition simulation (test oracle).

    Adds by hand, column by column, and reports the frozen
    vocabulary: no column carries -> ``none``; every column of
    the wider operand carries -> ``full_chain``; else ``some``.
    """
    da = _digits_of(a)[::-1]
    db = _digits_of(b)[::-1]
    width = max(len(da), len(db))
    carry = 0
    carried = []
    for i in range(width):
        x = int(da[i]) if i < len(da) else 0
        y = int(db[i]) if i < len(db) else 0
        total = x + y + carry
        carried.append(total >= 10)
        carry = 1 if total >= 10 else 0
    if not any(carried):
        return "none"
    if all(carried):
        return "full_chain"
    return "some"


def _margin_oracle(a: int, b: int) -> str:
    """Independent margin classifier (test oracle).

    Encodes the definition in ``margin_class``'s docstring by a
    different route: the first differing position from the left
    is in the leading half when twice its index is below the
    length; the final position is its own class.
    """
    if a == b:
        return "equal"
    da, db = _digits_of(a), _digits_of(b)
    if len(da) != len(db):
        return "different_length"
    # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
    index = next(i for i, (x, y) in enumerate(zip(da, db, strict=True)) if x != y)
    if index == len(da) - 1:
        return "same_length_last_digit"
    if 2 * index < len(da):
        return "same_length_early"
    return "same_length_middle"


class TestRenderers(unittest.TestCase):
    def test_battery(self):
        cases = {
            0: ("0", "0", "0", "0e0"),
            7: ("7", "7", "7", "7e0"),
            42: ("42", "42", "42", "4.2e1"),
            100: ("100", "100", "100", "1e2"),
            999: ("999", "999", "999", "9.99e2"),
            1000: ("1000", "1,000", f"1{SPACE}000", "1e3"),
            1234567: (
                "1234567",
                "1,234,567",
                f"1{SPACE}234{SPACE}567",
                "1.234567e6",
            ),
            999000000000: (
                "999000000000",
                "999,000,000,000",
                f"999{SPACE}000{SPACE}000{SPACE}000",
                "9.99e11",
            ),
            10**12: (
                "1000000000000",
                "1,000,000,000,000",
                f"1{SPACE}000{SPACE}000{SPACE}000{SPACE}000",
                "1e12",
            ),
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(tuple(nf.render_form(value, f) for f in nf.FORMS), expected)

    def test_sweep_round_trip_and_injectivity(self):
        # Injectivity is per form: below 1000 the grouped forms
        # coincide with plain by design (that is what the
        # form_identical flag records), so cross-form string
        # reuse is expected there -- and harmless, because the
        # form is always known from context and parse is a
        # function (one string, at most one value).
        seen: dict[str, dict[str, int]] = {form: {} for form in nf.FORMS}
        probe = list(
            dict.fromkeys(
                list(range(0, 2001))
                + [10**k for k in range(1, 13)]
                + [10**12 - 1, 999_999_999_999, 123_456_789_012]
            )
        )
        for value in probe:
            for form in nf.FORMS:
                text = nf.render_form(value, form)
                self.assertEqual(nf.parse_value(text), value, (value, form, text))
                self.assertNotIn(text, seen[form], (text, seen[form].get(text), value))
                seen[form][text] = value

    def test_sweep_matches_redteam_battery(self):
        # Red-team A's own examples land on B's expected strings.
        self.assertEqual(nf.render_form(9_990_000, "scientific"), "9.99e6")
        total = 9_990_000 + 9_990_000
        self.assertEqual(nf.render_form(total, "scientific"), "1.998e7")
        self.assertEqual(nf.parse_value("2.500000e6"), 2_500_000)

    def test_render_rejects(self):
        with self.assertRaises(ValueError):
            nf.render_form(-1, "plain")
        with self.assertRaises(ValueError):
            nf.render_form(True, "plain")
        with self.assertRaises(ValueError):
            nf.render_form(5, "words")


class TestParser(unittest.TestCase):
    def test_acceptances(self):
        self.assertEqual(nf.parse_value("0"), 0)
        self.assertEqual(nf.parse_value("0123"), 123)
        self.assertEqual(nf.parse_value("1,234,567"), 1234567)
        self.assertEqual(nf.parse_value(f"1{SPACE}234"), 1234)
        self.assertEqual(nf.parse_value("1E6"), 1000000)
        self.assertEqual(nf.parse_value("1e+6"), 1000000)
        self.assertEqual(nf.parse_value("2.500000e6"), 2500000)
        self.assertEqual(nf.parse_value("0e0"), 0)
        self.assertEqual(nf.parse_value("0.0e5"), 0)

    def test_rejections(self):
        for text in (
            "",
            " ",
            " 123",
            "123 ",
            "-5",
            "+5",
            "1_000",
            "1,23,456",
            "12,34",
            "1,2345",
            f"1{SPACE}23{SPACE}456",
            "١٢٣",  # Arabic-Indic digits are not ASCII digits
            "NaN",
            "inf",
            "1.5",
            "1.5e0",  # scientific shape, non-integer value
            "1e",
            "e5",
            ".5e3",
            "1.e3",
            "1,234e5",  # grouping never mixes with scientific
            "12a3",
            "first",
            "1e-0",  # the frozen grammar signs exponents with + only
            "1e-3",
        ):
            with self.subTest(text=text):
                self.assertIsNone(nf.parse_value(text))

    def test_caps_fire_before_conversion(self):
        # The exponent cap is the DoS guard: these must classify,
        # never allocate a giant integer.
        self.assertIsNone(nf.parse_value("1e1000"))
        self.assertIsNone(nf.parse_value("1e999999999"))
        self.assertIsNone(nf.parse_value("9" * 21))
        self.assertEqual(nf.parse_value("9" * 20), int("9" * 20))
        self.assertIsNone(nf.parse_value(None))
        self.assertIsNone(nf.parse_value(123))


class TestClassifiers(unittest.TestCase):
    def test_carry_truth_table(self):
        self.assertEqual(nf.carry_class(100, 200), "none")
        self.assertEqual(nf.carry_class(104, 202), "none")
        self.assertEqual(nf.carry_class(105, 206), "some")
        self.assertEqual(nf.carry_class(150, 260), "some")
        self.assertEqual(nf.carry_class(999, 1), "full_chain")
        self.assertEqual(nf.carry_class(55, 55), "full_chain")
        self.assertEqual(nf.carry_class(500, 500), "some")
        self.assertEqual(nf.carry_class(199, 1), "some")
        self.assertEqual(nf.carry_class(1299, 1), "some")

    def test_carry_matches_oracle_on_sweep(self):
        import itertools

        values = [0, 1, 5, 9, 10, 55, 99, 100, 500, 999, 1000, 5999, 9999, 123456]
        for a, b in itertools.product(values, values):
            self.assertEqual(nf.carry_class(a, b), _carry_oracle(a, b), (a, b))

    def test_margin_truth_table(self):
        self.assertEqual(nf.margin_class(5, 5), "equal")
        self.assertEqual(nf.margin_class(99, 100), "different_length")
        self.assertEqual(nf.margin_class(200, 100), "same_length_early")
        self.assertEqual(nf.margin_class(129, 121), "same_length_last_digit")
        self.assertEqual(nf.margin_class(121, 191), "same_length_early")
        self.assertEqual(nf.margin_class(1211, 1291), "same_length_middle")

    def test_margin_matches_oracle_on_sweep(self):
        import itertools

        values = [1, 9, 10, 99, 100, 121, 129, 191, 999, 1000, 123456]
        for a, b in itertools.product(values, values):
            self.assertEqual(nf.margin_class(a, b), _margin_oracle(a, b), (a, b))


class TestGeneration(unittest.TestCase):
    def test_shape_and_ids(self):
        value_set = nf.generate_value_set(0)
        self.assertEqual(sorted(value_set), sorted(nf.TASKS))
        for task in nf.TASKS:
            records = value_set[task]
            self.assertEqual(len(records), nf.VALUES_PER_TASK)
            self.assertEqual(
                [r["value_id"] for r in records],
                [f"nf-{task}-{i:02d}" for i in range(16)],
            )
            for record in records:
                # The recorded band labels the magnitude on the
                # record: the value itself for copy, the larger
                # operand for pairs (a different-length compare
                # pair is drawn in one band and labelled by its
                # larger side).
                magnitude = max(record["operands"])
                digits = len(str(magnitude))
                expected_band = next(i for i, (lo, hi) in enumerate(nf.BANDS) if lo <= digits <= hi)
                self.assertEqual(record["band"], expected_band, record["value_id"])
            if task == "copy":
                for band in range(4):
                    self.assertEqual(sum(1 for r in records if r["band"] == band), 4, (task, band))

    def test_strata_verified_by_oracles(self):
        for seed in range(8):
            value_set = nf.generate_value_set(seed)
            for record in value_set["copy"]:
                value = record["operands"][0]
                lo, hi = nf.BANDS[record["band"]]
                self.assertTrue(lo <= len(str(value)) <= hi)
                self.assertEqual(record["stratum"] == "round", nf._is_round_value(value))
                self.assertEqual(record["expected_value"], value)
            margins = set()
            for record in value_set["compare"]:
                a, b = record["operands"]
                self.assertNotEqual(a, b)
                self.assertEqual(record["stratum"], _margin_oracle(a, b))
                self.assertNotEqual(record["stratum"], "same_length_middle")
                self.assertEqual(record["expected_value"], max(a, b))
                margins.add(record["stratum"])
            self.assertEqual(
                margins,
                {"different_length", "same_length_early", "same_length_last_digit"},
            )
            carries = set()
            for record in value_set["add"]:
                a, b = record["operands"]
                self.assertEqual(record["stratum"], _carry_oracle(a, b))
                self.assertEqual(record["expected_value"], a + b)
                self.assertLess(a, 10**12)
                self.assertLessEqual(a + b, 10**13)
                carries.add(record["stratum"])
            self.assertEqual(carries, {"none", "some", "full_chain"})

    def test_determinism(self):
        self.assertEqual(nf.generate_value_set(7), nf.generate_value_set(7))
        self.assertNotEqual(nf.generate_value_set(7), nf.generate_value_set(8))
        rows_a = nf.canonical_generated_rows(3)
        rows_b = nf.canonical_generated_rows(3)
        self.assertEqual(rows_a, rows_b)
        digest = hashlib.sha256(
            json.dumps(rows_a, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        self.assertEqual(nf.generator_sha256(3), digest)
        self.assertNotEqual(nf.generator_sha256(3), nf.generator_sha256(4))

    def test_fixture_id_pairing_across_forms(self):
        for seed in (0, 5):
            fixtures = nf.build_fixtures(seed)
            value_set = nf.generate_value_set(seed)
            self.assertEqual(len(fixtures), 12)
            for task in nf.TASKS:
                id_sets = {
                    form: tuple(case.id for case in fixtures[(task, form)]) for form in nf.FORMS
                }
                self.assertEqual(len(set(id_sets.values())), 1)
                entries = {entry["value_id"]: entry for entry in value_set[task]}
                for form in nf.FORMS:
                    for case in fixtures[(task, form)]:
                        entry = entries[case.id]
                        self.assertEqual(case.expected, nf.expected_answer(entry, form))
                        self.assertEqual(case.messages[-1].content, nf.prompt_text(entry, form))

    def test_expected_answers(self):
        value_set = nf.generate_value_set(0)
        for record in value_set["copy"]:
            self.assertEqual(
                nf.expected_answer(record, "comma"),
                nf.render_form(record["operands"][0], "comma"),
            )
        for record in value_set["compare"]:
            a, b = record["operands"]
            self.assertEqual(
                nf.expected_answer(record, "space"),
                "first" if a > b else "second",
            )
        for record in value_set["add"]:
            self.assertEqual(
                nf.expected_answer(record, "scientific"),
                nf.render_form(record["expected_value"], "scientific"),
            )

    def test_prompts_carry_no_digit(self):
        for task in nf.TASKS:
            text = nf.PROMPT_TEMPLATES[task]
            self.assertFalse(any(ch.isdigit() for ch in text), task)
        canonical = json.dumps(
            nf.PROMPT_TEMPLATES, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        self.assertEqual(
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            nf.PROMPT_TEMPLATES_SHA256,
        )

    def test_hash_seed_independence(self):
        script = (
            "import json,sys;"
            "sys.path.insert(0, '.');"
            "from model_protocol_lab import numform as nf;"
            "print(json.dumps(nf.generate_value_set(11), sort_keys=True))"
        )
        outputs = set()
        for hash_seed in ("1", "2"):
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                env={"PYTHONHASHSEED": hash_seed, "PATH": "/usr/bin:/bin"},
                cwd=".",
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            outputs.add(proc.stdout)
        self.assertEqual(len(outputs), 1)

    def test_previously_crashing_seeds_generate(self):
        # Cycle 187 (wave-8 F1): seeds whose band-0 "some" slot
        # redrew a = 0 raised ZeroDivisionError in _add_partner
        # (empty pivot set). The construction now proposes 0,
        # the classifier rejects it, and the loop redraws.
        for seed in (26, 109, 116, 127, 167, 229, 249, 273, 517, 528):
            value_set = nf.generate_value_set(seed)
            for record in value_set["add"]:
                a, b = record["operands"]
                self.assertEqual(record["stratum"], _carry_oracle(a, b), (seed, record))
            for record in value_set["compare"]:
                a, b = record["operands"]
                self.assertEqual(record["stratum"], _margin_oracle(a, b), (seed, record))

    def test_seed_sweep_generation_is_total(self):
        # Generation must succeed for every seed, not just the
        # ones the stage tests happen to use: the CLI accepts
        # any int as --seed, including negatives.
        for seed in list(range(1000)) + list(range(-100, 0)):
            value_set = nf.generate_value_set(seed)
            self.assertEqual(
                sum(len(entries) for entries in value_set.values()), 3 * nf.VALUES_PER_TASK
            )
            for record in value_set["add"]:
                a, b = record["operands"]
                self.assertEqual(record["stratum"], _carry_oracle(a, b), (seed, record))
                self.assertEqual(record["expected_value"], a + b)

    def test_healthy_seed_hashes_pinned(self):
        # The F1 fix must be set-preserving: seeds that already
        # generated valid sets keep byte-identical instruments,
        # or every existing report becomes incomparable for no
        # benefit. Hashes captured from the pre-fix generator.
        pinned = {
            0: "8337b8fa804de3ba23419a4fcde9d04cb55003a24a162a1aec2c271d874269d3",
            3: "87d4f04dcb0d145401e48cee612ff9bd0545d23facc69dc278e0e750b9bb8c9e",
            7: "81353b240e7af9b47f073211ac94742fb35c2f83c0b33e8ad01e2382674f65b5",
            42: "c08ea0f2828ce2d5cd60e0de83b5a8433ed1bb7600d397ea6b4394792371d79a",
            100: "ddd68b1866352067d1cc164c399c491d4b4b0c79b85258450da15ca098481fdf",
            999: "bdd11fbdd6f81f643d70369f37ad0f5131e9c7f6b2a0f1bf3d45b32f35cbe626",
            1234: "b379ba0d40d3741cfc702ea1d2c58ecc679f820e32dd58d69a6f8e12f65cd8d8",
            -5: "7e4ff951cf3985e972cd6376744b7aac678694ba950ed7374ce5503a868ccf59",
        }
        for seed, digest in pinned.items():
            self.assertEqual(nf.generator_sha256(seed), digest, seed)


class TestGroupingClassifier(unittest.TestCase):
    @staticmethod
    def _evidence(counts):
        probes = nf.probe_strings()
        evidence = {}
        for probe in probes:
            if probe.isascii() and probe.isdigit():
                evidence[probe] = counts[len(probe)]
            else:
                evidence[probe] = 3
        return evidence

    def test_single_digit(self):
        evidence = self._evidence({n: n for n in range(1, 13)})
        self.assertEqual(nf.classify_grouping(evidence), "single_digit")

    def test_chunked(self):
        evidence = self._evidence({n: -(-n // 3) for n in range(1, 13)})
        self.assertEqual(nf.classify_grouping(evidence), "chunked_up_to_3")

    def test_irregular(self):
        counts = {n: -(-n // 3) for n in range(1, 13)}
        counts[12] = 11  # within the envelope, matching neither rule
        counts[2] = 2
        evidence = self._evidence(counts)
        self.assertEqual(nf.classify_grouping(evidence), "irregular_or_mixed")

    def test_unknown_cases(self):
        self.assertEqual(nf.classify_grouping({}), "unknown")
        evidence = self._evidence({n: n for n in range(1, 13)})
        del evidence["1234567"]
        self.assertEqual(nf.classify_grouping(evidence), "unknown")
        expansion = self._evidence({n: n + 1 for n in range(1, 13)})
        self.assertEqual(nf.classify_grouping(expansion), "unknown")
        below_floor = self._evidence(dict.fromkeys(range(1, 13), 1))
        self.assertEqual(nf.classify_grouping(below_floor), "unknown")

    def test_measure_with_stub_encoders(self):
        value_set = nf.generate_value_set(0)
        char = nf.measure_grouping(tuple, value_set)
        self.assertEqual(char["classification"], "single_digit")
        self.assertEqual(char["token_deltas_vs_plain"]["plain"], 0.0)
        self.assertIsNotNone(char["form_token_means"]["comma"])

        def chunk_encode(text):
            return tuple(text[i : i + 3] for i in range(0, len(text), 3)) or (0,)

        chunked = nf.measure_grouping(chunk_encode, value_set)
        self.assertEqual(chunked["classification"], "chunked_up_to_3")

        def boom(text):
            raise RuntimeError("no encoder")

        failed = nf.measure_grouping(boom, value_set)
        self.assertEqual(failed["classification"], "unknown")
        self.assertEqual(failed["evidence"], {})
        self.assertIsNone(failed["form_token_means"]["plain"])


class TestContrastClassification(unittest.TestCase):
    def test_truth_table(self):
        cases = [
            ((3, 5, 0), "insufficient"),
            ((6, 0, 0), "no_difference_observed"),
            ((16, 1, 0), "below_threshold"),
            ((16, 5, 1), "a_advantage"),
            ((16, 1, 5), "b_advantage"),
            ((16, 4, 4), "mixed_discordance"),
            ((16, 2, 0), "a_advantage"),
        ]
        for (n, a_only, b_only), label in cases:
            with self.subTest(n=n, a=a_only, b=b_only):
                self.assertEqual(nf.classify_contrast(n, a_only, b_only), label)

    @staticmethod
    def _contrast_records(verdicts_a, verdicts_b):
        records = []
        for form, verdicts in (("plain", verdicts_a), ("comma", verdicts_b)):
            for index, verdict in enumerate(verdicts):
                records.append(
                    {
                        "task": "copy",
                        "form": form,
                        "value_id": f"nf-copy-{index:02d}",
                        "cell": {"verdict": verdict} if verdict is not None else None,
                    }
                )
        return records

    def test_paired_contrast_counts(self):
        verdicts_a = [i < 10 for i in range(16)]
        verdicts_b = [i < 6 or i >= 12 for i in range(16)]
        contrast = nf.paired_contrast(
            self._contrast_records(verdicts_a, verdicts_b), "copy", "plain", "comma"
        )
        self.assertEqual(contrast["n_paired"], 16)
        self.assertEqual(contrast["n_pass_a_only"], 4)
        self.assertEqual(contrast["n_pass_b_only"], 4)
        self.assertEqual(contrast["n_both_pass"], 6)
        self.assertEqual(contrast["n_both_fail"], 2)
        self.assertEqual(contrast["classification"], "mixed_discordance")
        self.assertEqual(contrast["mcnemar_p"], mcnemar_exact_p(4, 4))
        partial = nf.paired_contrast(
            self._contrast_records([True] + [None] * 15, [True] * 16),
            "copy",
            "plain",
            "comma",
        )
        self.assertEqual(partial["n_paired"], 1)
        self.assertEqual(partial["classification"], "insufficient")

    def test_mcnemar_none_without_discordance(self):
        contrast = nf.paired_contrast(
            self._contrast_records([True] * 16, [True] * 16), "copy", "plain", "comma"
        )
        self.assertIsNone(contrast["mcnemar_p"])
        self.assertEqual(contrast["classification"], "no_difference_observed")


class TestDecomposition(unittest.TestCase):
    @staticmethod
    def _record(task, verdict, answer, expected, boundary=True, value=None, expected_value=None):
        return {
            "task": task,
            "expected_answer": expected,
            "expected_value": expected_value,
            "answer_value": value,
            "cell": {
                "verdict": verdict,
                "boundary_found": boundary,
                "answer_digest": hashlib.sha256(answer.encode()).hexdigest()
                if answer is not None
                else "0" * 64,
            },
        }

    def test_classes(self):
        self.assertEqual(
            nf.cell_decomposition(self._record("copy", True, "1,000", "1,000", value=1000)),
            "exact_pass",
        )
        self.assertEqual(
            nf.cell_decomposition(
                self._record("copy", False, "1000", "1,000", value=1000, expected_value=1000)
            ),
            "value_pass_form_fail",
        )
        self.assertEqual(
            nf.cell_decomposition(self._record("add", False, "5", "6", value=5, expected_value=6)),
            "value_fail",
        )
        self.assertEqual(
            nf.cell_decomposition(self._record("compare", False, "first", "second")),
            "wrong_label",
        )
        self.assertEqual(
            nf.cell_decomposition(self._record("compare", False, "banana", "second")),
            "non_label",
        )
        no_cell = self._record("copy", False, "x", "y")
        no_cell["cell"] = None
        with self.assertRaises(ValueError):
            nf.cell_decomposition(no_cell)


FMT_A = Format(
    name="nf-a",
    bos="<|im_start|>system\nYou are helpful.<|im_end|>\n",
    user_prefix="<|im_start|>user\n",
    user_turn_end="<|im_end|>\n",
    assistant_prefix="<|im_start|>assistant\n",
    turn_end="<|im_end|>",
)
FMT_B = Format(
    name="nf-b",
    bos="",
    user_prefix="USER: ",
    assistant_prefix="ASSISTANT: ",
    turn_end="~",
)
FORMATS = (FMT_A, FMT_B)


def _form_of_token(token: str) -> str:
    if "," in token:
        return "comma"
    if SPACE in token:
        return "space"
    if "e" in token and any(ch.isdigit() for ch in token):
        return "scientific"
    return "plain"


class _NumformBackend:
    """Word-token backend answering the generated tasks by rule.

    Modes: ``faithful`` (answer in the requested form),
    ``plain`` (always plain digits -- the normalising model),
    ``wrong_value`` (off by one, in the requested form),
    ``flip_label`` (compare labels swapped, other tasks
    faithful). ``fail_tasks`` makes generate raise for those
    tasks; ``cancel_on_task`` sets the cancel event when that
    task is first generated.
    """

    fingerprint = "sha256:numform-test"

    def __init__(self, mode="faithful", fail_tasks=()):
        self.vocab: dict[str, int] = {}
        self.rev: dict[int, str] = {}
        self.mode = mode
        self.fail_tasks = set(fail_tasks)
        self.generate_calls = 0
        self.cancel_on_task: str | None = None
        self.cancel_event: Event | None = None

    def encode(self, text):
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
        return " ".join(self.rev.get(int(i), "?") for i in ids)

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        self.generate_calls += 1
        text = self.decode(list(input_ids))
        if "Copy this number" in text:
            task = "copy"
        elif "Which of these" in text:
            task = "compare"
        else:
            task = "add"
        if task in self.fail_tasks:
            raise RuntimeError("scripted failure")
        if self.cancel_on_task == task and self.cancel_event is not None:
            self.cancel_event.set()

        # The last operand token carries the format's turn-end
        # glued to it ("100<|im_end|>", "100~"); strip the glue
        # before parsing.
        def clean(token: str) -> str:
            for sep in ("<|", "~"):
                token = token.split(sep)[0]
            return token

        tokens = []
        # Split on ASCII space and newline only: str.split()
        # would also split on U+202F, the space form's own
        # grouping separator, and shred those operands.
        for raw_token in text.replace("\n", " ").split(" "):
            token = clean(raw_token)
            if any(ch.isdigit() for ch in token) and nf.parse_value(token) is not None:
                tokens.append(token)
        values = [nf.parse_value(token) for token in tokens]
        if task == "copy":
            value = values[-1]
            form = _form_of_token(tokens[-1])
            if self.mode == "plain":
                answer = str(value)
            elif self.mode == "wrong_value":
                answer = nf.render_form(value + 1, form)
            else:
                answer = nf.render_form(value, form)
        elif task == "compare":
            first, second = values[-2], values[-1]
            label = "first" if first > second else "second"
            if self.mode == "flip_label":
                label = "second" if label == "first" else "first"
            answer = label
        else:
            first, second = values[-2], values[-1]
            form = _form_of_token(tokens[-2])
            total = first + second
            if self.mode == "plain":
                answer = str(total)
            elif self.mode == "wrong_value":
                answer = nf.render_form(total + 1, form)
            else:
                answer = nf.render_form(total, form)
        # The boundary is the whole encoded sequence: one id
        # for the word tokenizer, several for the char variant.
        boundary_ids = self.encode(BOUNDARY_WORD)
        out = list(self.encode(answer)) + list(boundary_ids)
        out = (out + [boundary_ids[-1]] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


def _run_stage(backend, formats=FORMATS, **kwargs):
    kwargs.setdefault("boundary", tuple(backend.encode(BOUNDARY_WORD)))
    kwargs.setdefault("max_new_tokens", 64)
    kwargs.setdefault("max_context", 8192)
    kwargs.setdefault("seed", 0)
    return numform_stage(backend, formats, **kwargs)


def _payload(backend, stage):
    return numform_analysis(stage, tuple(fmt.name for fmt in FORMATS))


class TestStage(unittest.TestCase):
    def test_happy_path_faithful(self):
        backend = _NumformBackend()
        stage = _run_stage(backend)
        self.assertTrue(stage["available"], stage["detail"])
        self.assertEqual(stage["grouping"]["classification"], "unknown")
        for name in ("nf-a", "nf-b"):
            records = stage["candidates"][name]["records"]
            self.assertEqual(len(records), 192)
            self.assertTrue(all(r["cell"] is not None for r in records))
            self.assertTrue(all(r["detail"] == "" for r in records))
        payload = _payload(backend, stage)
        self.assertTrue(_valid_numform(payload))
        candidate = payload["candidates"]["nf-a"]
        for task in nf.TASKS:
            for form in nf.FORMS:
                aggregate = candidate["aggregates"][task][form]
                if task == "add" and form in ("comma", "space"):
                    # The backend reads the form off the first
                    # operand's string; where that string is
                    # plain-looking (sub-1000) but the arm's form
                    # is grouped, it answers plain and fails the
                    # grouped cells whose sum shows the grouping.
                    # The count below restates that scaffolding
                    # rule over the generated set.
                    value_set = nf.generate_value_set(0)
                    expected_exact = 0
                    for entry in value_set["add"]:
                        first = entry["operands"][0]
                        detected = _form_of_token(nf.render_form(first, form))
                        if nf.render_form(entry["expected_value"], detected) == (
                            nf.render_form(entry["expected_value"], form)
                        ):
                            expected_exact += 1
                    self.assertEqual(aggregate["n_exact"], expected_exact, form)
                else:
                    self.assertEqual(aggregate["exact_rate"], 1.0, (task, form))
        contrast = candidate["contrasts"]["add"]["plain__scientific"]
        self.assertEqual(contrast["classification"], "no_difference_observed")

    def test_add_grouping_can_be_invisible_in_the_prompt(self):
        # Design property, pinned: an add pair whose operands
        # are both sub-1000 renders byte-identically in the
        # plain, comma and space arms, yet the expected answers
        # differ once the sum crosses 1000. The decomposition
        # is what keeps that honest -- the failing grouped cell
        # is a right-value/wrong-form cell, not a wrong value.
        fixtures = nf.build_fixtures(0)
        value_set = nf.generate_value_set(0)
        ambiguous = [
            entry
            for entry in value_set["add"]
            if max(entry["operands"]) < 1000 <= entry["expected_value"]
        ]
        self.assertTrue(ambiguous)
        for entry in ambiguous:
            renders = {
                form: FMT_B.render(fixtures[("add", form)][int(entry["value_id"][-2:])].messages)
                for form in ("plain", "comma", "space")
            }
            self.assertEqual(len(set(renders.values())), 1)
            expecteds = {nf.expected_answer(entry, form) for form in ("plain", "comma", "space")}
            self.assertGreater(len(expecteds), 1)

    def test_plain_model_decomposition(self):
        backend = _NumformBackend(mode="plain")
        payload = _payload(backend, _run_stage(backend))
        self.assertTrue(_valid_numform(payload))
        candidate = payload["candidates"]["nf-a"]
        value_set = nf.generate_value_set(0)
        identical = sum(
            1
            for entry in value_set["copy"]
            if nf.render_form(entry["operands"][0], "comma")
            == nf.render_form(entry["operands"][0], "plain")
        )
        comma = candidate["aggregates"]["copy"]["comma"]
        self.assertEqual(comma["n_exact"], identical)
        self.assertEqual(comma["decomposition"]["value_pass_form_fail"], 16 - identical)
        self.assertEqual(comma["decomposition"].get("value_fail", 0), 0)
        plain = candidate["aggregates"]["copy"]["plain"]
        self.assertEqual(plain["n_exact"], 16)
        contrast = candidate["contrasts"]["add"]["plain__scientific"]
        self.assertEqual(contrast["classification"], "plain_advantage")
        self.assertGreaterEqual(contrast["n_pass_a_only"], 2)

    def test_wrong_value_model(self):
        backend = _NumformBackend(mode="wrong_value")
        payload = _payload(backend, _run_stage(backend))
        candidate = payload["candidates"]["nf-b"]
        for task in ("copy", "add"):
            aggregate = candidate["aggregates"][task]["plain"]
            self.assertEqual(aggregate["n_exact"], 0)
            self.assertEqual(aggregate["decomposition"]["value_fail"], 16)
        compare = candidate["aggregates"]["compare"]["plain"]
        self.assertEqual(compare["n_exact"], 16)  # labels unaffected

    def test_flip_label_model(self):
        backend = _NumformBackend(mode="flip_label")
        payload = _payload(backend, _run_stage(backend))
        candidate = payload["candidates"]["nf-a"]
        compare = candidate["aggregates"]["compare"]["comma"]
        self.assertEqual(compare["n_exact"], 0)
        self.assertEqual(compare["decomposition"].get("wrong_label", 0), 16)
        self.assertEqual(compare["decomposition"].get("non_label", 0), 0)

    def test_context_exceeded(self):
        backend = _NumformBackend()
        stage = _run_stage(backend, max_context=8)
        self.assertTrue(stage["available"])
        records = stage["candidates"]["nf-a"]["records"]
        self.assertTrue(all(r["cell"] is None for r in records))
        self.assertTrue(all(r["detail"] == "arm_context_exceeded" for r in records))
        payload = _payload(backend, stage)
        self.assertTrue(_valid_numform(payload))
        aggregate = payload["candidates"]["nf-a"]["aggregates"]["copy"]["plain"]
        self.assertEqual(aggregate["n_completed"], 0)
        self.assertIsNone(aggregate["exact_rate"])

    def test_answer_budget_exceeded(self):
        backend = _NumformBackend()
        stage = _run_stage(backend, max_new_tokens=1)
        records = stage["candidates"]["nf-a"]["records"]
        self.assertTrue(all(r["detail"] == "answer_budget_exceeded" for r in records))
        self.assertTrue(all(r["cell"] is None for r in records))

    def test_form_not_in_render(self):
        # A template that normalises its input (strips grouping
        # separators, upper-cases the scientific exponent) makes
        # the grouped operand strings vanish from the render;
        # those cells must record form_not_in_render while plain
        # cells still run.
        import re

        original_render = Format.render

        def normalising_render(self, messages):
            text = original_render(self, messages)
            text = text.replace(",", "").replace(SPACE, "")
            return re.sub(r"e(?=[0-9])", "E", text)

        backend = _NumformBackend()
        with mock.patch.object(Format, "render", normalising_render):
            stage = _run_stage(backend)
        records = stage["candidates"]["nf-a"]["records"]
        for record in records:
            if record["form"] == "plain":
                self.assertIsNotNone(record["cell"])
            elif record["form"] == "scientific":
                self.assertEqual(record["detail"], "form_not_in_render")
                self.assertIsNone(record["cell"])
            else:
                # Grouped forms: the guard fires exactly when an
                # operand's grouped string is visible enough to
                # be stripped -- sub-1000 operands render as
                # plain strings, which the normaliser leaves
                # alone, so those cells legitimately run.
                if max(record["operands"]) >= 1000:
                    self.assertEqual(record["detail"], "form_not_in_render")
                    self.assertIsNone(record["cell"])
                else:
                    self.assertIsNotNone(record["cell"])

    def test_evaluation_failed(self):
        backend = _NumformBackend(fail_tasks={"add"})
        stage = _run_stage(backend)
        self.assertTrue(stage["available"])
        records = stage["candidates"]["nf-a"]["records"]
        add_records = [r for r in records if r["task"] == "add"]
        self.assertTrue(all(r["detail"] == "arm_evaluation_failed" for r in add_records))
        self.assertTrue(all(r["cell"] is None for r in add_records))
        self.assertTrue(all(r["cell"] is not None for r in records if r["task"] != "add"))

    def test_template_screen(self):
        backend = _NumformBackend()
        with mock.patch.dict(
            nf.PROMPT_TEMPLATES,
            {"copy": "Copy <|im_end|> this number exactly:\n{operand}"},
        ):
            stage = _run_stage(backend)
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "numform_template_screen_failed")
        self.assertEqual(backend.generate_calls, 0)

    def test_cancelled_before_start(self):
        backend = _NumformBackend()
        event = Event()
        event.set()
        stage = _run_stage(backend, cancel=event)
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")
        self.assertEqual(backend.generate_calls, 0)

    def test_cancelled_mid_run(self):
        backend = _NumformBackend()
        event = Event()
        backend.cancel_on_task = "add"
        backend.cancel_event = event
        stage = _run_stage(backend, cancel=event)
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")
        self.assertGreater(backend.generate_calls, 0)

    def test_record_budget(self):
        backend = _NumformBackend()
        with mock.patch.object(nf, "MAX_GROUND_RECORDS", 100):
            stage = _run_stage(backend)
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "numform_record_budget_exceeded")

    def test_empty_formats_raise(self):
        backend = _NumformBackend()
        with self.assertRaises(ValueError):
            numform_stage(backend, (), boundary=(1,), max_context=100)


class TestAnalysis(unittest.TestCase):
    def test_unavailable_stage_passthrough(self):
        stage = {
            "available": False,
            "detail": "cancelled",
            "protocol": {"version": 1},
            "candidates": {},
        }
        payload = numform_analysis(stage, ("nf-a",))
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "cancelled")
        self.assertEqual(payload["protocol"], {"version": 1})

    def test_unavailable_candidate_entry(self):
        backend = _NumformBackend()
        stage = _run_stage(backend)
        stage["candidates"]["nf-b"] = {
            "status": "unavailable",
            "detail": "arm_evaluation_failed",
            "records": [],
        }
        payload = _payload(backend, stage)
        entry = payload["candidates"]["nf-b"]
        self.assertEqual(entry["status"], "unavailable")
        self.assertEqual(entry["records"], [])
        self.assertTrue(_valid_numform(payload))

    def test_aggregates_recomputed_independently(self):
        backend = _NumformBackend(mode="plain")
        payload = _payload(backend, _run_stage(backend))
        records = payload["candidates"]["nf-a"]["records"]
        subset = [r for r in records if r["task"] == "add" and r["form"] == "space"]
        expected = nf.aggregate_records(records, "add", "space")
        self.assertEqual(payload["candidates"]["nf-a"]["aggregates"]["add"]["space"], expected)
        n_exact = sum(1 for r in subset if r["cell"] and r["cell"]["verdict"])
        self.assertEqual(expected["n_exact"], n_exact)


def _base_payload(mode="faithful", seed=0, fingerprint=None):
    backend = _NumformBackend(mode=mode)
    if fingerprint is not None:
        backend.fingerprint = fingerprint
    stage = _run_stage(backend, seed=seed)
    return _payload(backend, stage)


class _CharBackend(_NumformBackend):
    """Char-token variant: digit probes count one token per
    character, so the grouping classification is single_digit.
    """

    fingerprint = "sha256:numform-char"

    @staticmethod
    def encode(text):
        return tuple(ord(ch) for ch in text)

    @staticmethod
    def decode(ids):
        return "".join(chr(int(i)) for i in ids)


class TestValidatorBattery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = _base_payload()

    def _tampered(self, mutate):
        payload = copy.deepcopy(self.base)
        mutate(payload)
        return payload

    def test_base_validates(self):
        self.assertTrue(_valid_numform(self.base))
        self.assertTrue(_valid_numform(_base_payload(mode="plain")))
        self.assertTrue(_valid_numform(_base_payload(seed=3)))

    def test_seed26_payload_validates(self):
        # Cycle 187 (wave-8 F1): seed 26 crashed the generator
        # before the fix, so no honest payload could exist for
        # it; now the full stage + validator path accepts it.
        self.assertTrue(_valid_numform(_base_payload(seed=26)))

    @staticmethod
    def _rehash_from_records(payload):
        """Recompute the protocol hashes the pre-cycle-187 way:
        the generated set's hash derived from the payload's own
        records, then the protocol fingerprint over that. A
        payload carrying these hashes is exactly what the old
        validator vouched for.
        """
        first = payload["candidates"][sorted(payload["candidates"])[0]]["records"]
        by_entry = {}
        for record in first:
            by_entry.setdefault((record["task"], record["value_id"]), {})[record["form"]] = record
        rows = []
        for task in nf.TASKS:
            for index in range(16):
                value_id = f"nf-{task}-{index:02d}"
                forms_map = by_entry[(task, value_id)]
                plain = forms_map["plain"]
                rows.append(
                    [
                        task,
                        value_id,
                        list(plain["operands"]),
                        plain["expected_value"],
                        [forms_map[form]["expected_answer"] for form in nf.FORMS],
                    ]
                )
        payload["protocol"]["generator_sha256"] = hashlib.sha256(
            json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        core = {k: v for k, v in payload["protocol"].items() if k != "fingerprint"}
        payload["protocol"]["fingerprint"] = hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
                "utf-8"
            )
        ).hexdigest()

    @staticmethod
    def _recompute_candidate(entry):
        pairs = (
            ("plain", "comma"),
            ("plain", "space"),
            ("plain", "scientific"),
            ("comma", "space"),
            ("comma", "scientific"),
            ("space", "scientific"),
        )
        entry["aggregates"] = {
            task: {form: nf.aggregate_records(entry["records"], task, form) for form in nf.FORMS}
            for task in nf.TASKS
        }
        entry["contrasts"] = {
            task: {f"{a}__{b}": nf.paired_contrast(entry["records"], task, a, b) for a, b in pairs}
            for task in nf.TASKS
        }

    def test_fabricated_set_rejected(self):
        # Cycle 187 (wave-8 F2): the validator regenerates the
        # set from the seed. Swapping one value for a different
        # locally-valid one in every candidate -- with aggregates
        # recomputed and both protocol hashes recomputed the old
        # records-derived way -- validated before the fix and
        # must not validate now.
        def swap(payload):
            for entry in payload["candidates"].values():
                for record in entry["records"]:
                    if record["task"] == "copy" and record["value_id"] == "nf-copy-00":
                        record["operands"] = [123]
                        record["expected_value"] = 123
                        record["stratum"] = "random"
                        record["expected_answer"] = nf.render_form(123, record["form"])
                        record["form_identical"] = nf.render_form(
                            123, record["form"]
                        ) == nf.render_form(123, "plain")
                        record["cell"] = None
                        record["detail"] = "arm_context_exceeded"
                        record["answer_value"] = None
                self._recompute_candidate(entry)
            self._rehash_from_records(payload)

        self.assertFalse(_valid_numform(self._tampered(swap)))

    def test_seed_relabel_rejected(self):
        # The seed is part of the instrument binding: genuine
        # seed-0 records relabelled as seed 999 (with the
        # relabelled seed's true generator hash and a recomputed
        # fingerprint) are not seed 999's payload.
        def relabel(payload):
            payload["protocol"]["seed"] = 999
            payload["protocol"]["generator_sha256"] = nf.generator_sha256(999)
            core = {k: v for k, v in payload["protocol"].items() if k != "fingerprint"}
            payload["protocol"]["fingerprint"] = hashlib.sha256(
                json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
                    "utf-8"
                )
            ).hexdigest()

        self.assertFalse(_valid_numform(self._tampered(relabel)))

    def test_grouping_delta_relation(self):
        # Cycle 187 (wave-8 F4): each stored delta must equal
        # its form's mean minus the plain mean (the producer's
        # paired-difference mean over the same operand multiset),
        # not merely be a plausible number.
        def inconsistent(payload):
            payload["grouping"]["form_token_means"] = {
                "plain": 10.0,
                "comma": 20.0,
                "space": 30.0,
                "scientific": 40.0,
            }
            payload["grouping"]["token_deltas_vs_plain"] = {
                "plain": 0.0,
                "comma": 999.0,
                "space": -555.0,
                "scientific": 0.5,
            }

        self.assertFalse(_valid_numform(self._tampered(inconsistent)))

        def consistent(payload):
            payload["grouping"]["form_token_means"] = {
                "plain": 10.0,
                "comma": 20.0,
                "space": 30.0,
                "scientific": 40.0,
            }
            payload["grouping"]["token_deltas_vs_plain"] = {
                "plain": 0.0,
                "comma": 10.0,
                "space": 20.0,
                "scientific": 30.0,
            }

        self.assertTrue(_valid_numform(self._tampered(consistent)))

    def test_mirror_matches_producer_sweep(self):
        # The validator's generator mirror must reproduce the
        # producer's set exactly, seed for seed -- including
        # seed 4, whose copy draw hits the round-redraw loop.
        for seed in range(100):
            _construction, rows = dr._numform_generated(seed)
            digest = hashlib.sha256(
                json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            ).hexdigest()
            self.assertEqual(digest, nf.generator_sha256(seed), seed)

    def test_mirror_defensive_paths(self):
        # The mirror's fail-closed guards, proven directly the
        # way the producer's identical guards are proven above:
        # the out-of-band raise, the partner==0 nudge (both
        # branches), and the out-of-range partner retry.
        with self.assertRaises(ValueError):
            dr._numform_band_of(10**12)
        self.assertEqual(dr._numform_add_partner(9, "none", 10), 10)
        self.assertEqual(dr._numform_add_partner(8, "none", 10), 1)

        # The retry redraws with the next counter, so the flaky
        # set differs from the unpatched one; the property is
        # that mirror and producer agree under the same patch.
        def make_flaky(original):
            calls = {"n": 0}

            def flaky(a, target, derived):
                calls["n"] += 1
                if calls["n"] == 1:
                    return -1
                return original(a, target, derived)

            return flaky

        with mock.patch.object(
            dr, "_numform_compare_partner", make_flaky(dr._numform_compare_partner)
        ):
            _construction, rows = dr._numform_generated(0)
        digest = hashlib.sha256(
            json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        with mock.patch.object(nf, "_compare_partner", make_flaky(nf._compare_partner)):
            produced = nf.canonical_generated_rows(0)
        produced_digest = hashlib.sha256(
            json.dumps(produced, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        self.assertEqual(digest, produced_digest)

    def test_cross_candidate_answer_tokens(self):
        # Cycle 187 (wave-8 F4): expected_answer_tokens is
        # measured once per run, so candidates must agree on it
        # per record; a one-candidate edit (with that candidate's
        # aggregates honestly recomputed) is a construction
        # disagreement, not a per-candidate fact.
        def mismatch(payload):
            names = sorted(payload["candidates"])
            entry = payload["candidates"][names[1]]
            entry["records"][0]["expected_answer_tokens"] += 5
            self._recompute_candidate(entry)

        self.assertFalse(_valid_numform(self._tampered(mismatch)))

    def test_aggregate_tampers(self):
        def agg(payload):
            return payload["candidates"]["nf-a"]["aggregates"]["copy"]["comma"]

        tampers = [
            lambda p: agg(p).__setitem__("n_exact", agg(p)["n_exact"] + 1),
            lambda p: agg(p).__setitem__("exact_rate", 0.5),
            lambda p: agg(p)["decomposition"].__setitem__(
                "value_pass_form_fail_boundary_found", 99
            ),
            lambda p: agg(p)["by_band"]["1"].__setitem__("n_exact", 99),
            lambda p: agg(p).__setitem__("mean_input_tokens", 1.0),
            lambda p: agg(p).__setitem__("n_completed", 3),
        ]
        for i, mutate in enumerate(tampers):
            with self.subTest(tamper=i):
                self.assertFalse(_valid_numform(self._tampered(mutate)))

    def test_contrast_tampers(self):
        def contrast(payload):
            return payload["candidates"]["nf-a"]["contrasts"]["add"]["plain__comma"]

        tampers = [
            lambda p: contrast(p).__setitem__("n_paired", 3),
            lambda p: contrast(p).__setitem__("classification", "comma_advantage"),
            lambda p: contrast(p).__setitem__("mcnemar_p", 0.25),
            lambda p: contrast(p).__setitem__("n_both_pass", 1),
        ]
        for i, mutate in enumerate(tampers):
            with self.subTest(tamper=i):
                self.assertFalse(_valid_numform(self._tampered(mutate)))

    def test_record_tampers(self):
        def first(payload, task, form="plain"):
            # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
            return next(
                r
                for r in payload["candidates"]["nf-a"]["records"]
                if r["task"] == task and r["form"] == form
            )

        tampers = [
            lambda p: first(p, "copy")["cell"].__setitem__(
                "verdict", not first(p, "copy")["cell"]["verdict"]
            ),
            lambda p: first(p, "copy")["cell"].__setitem__(
                "boundary_found", not first(p, "copy")["cell"]["boundary_found"]
            ),
            lambda p: first(p, "copy")["cell"].__setitem__("answer_digest", "f" * 64),
            lambda p: first(p, "copy").__setitem__("detail", "arm_context_exceeded"),
            lambda p: first(p, "add").__setitem__(
                "stratum", "some" if first(p, "add")["stratum"] != "some" else "none"
            ),
            lambda p: first(p, "compare").__setitem__("stratum", "same_length_middle"),
            lambda p: first(p, "copy").__setitem__("band", (first(p, "copy")["band"] + 1) % 4),
            lambda p: first(p, "add").__setitem__(
                "expected_value", first(p, "add")["expected_value"] + 1
            ),
            lambda p: first(p, "copy").__setitem__(
                "expected_answer", first(p, "copy")["expected_answer"] + "x"
            ),
            lambda p: first(p, "copy")["operands"].__setitem__(
                0, first(p, "copy")["operands"][0] + 1
            ),
            lambda p: first(p, "copy").__setitem__(
                "form_identical", not first(p, "copy")["form_identical"]
            ),
            lambda p: first(p, "copy").__setitem__(
                "expected_answer_tokens", first(p, "copy")["expected_answer_tokens"] + 1
            ),
            lambda p: first(p, "copy").__setitem__(
                "answer_value", (first(p, "copy")["expected_value"] or 0) + 5
            ),
            lambda p: first(p, "compare").__setitem__("answer_value", 5),
            lambda p: first(p, "copy").__setitem__("round", True),
            lambda p: first(p, "copy").__setitem__("extra_key", 1),
            lambda p: first(p, "add").__setitem__("operands", [1]),
            lambda p: p["candidates"]["nf-a"]["records"].append(copy.deepcopy(first(p, "copy"))),
            lambda p: p["candidates"]["nf-a"]["records"].pop(0),
            lambda p: first(p, "copy")["cell"].__setitem__("input_tokens", -1),
            lambda p: first(p, "copy")["cell"].__setitem__("answer_digest", "zz"),
        ]
        for i, mutate in enumerate(tampers):
            with self.subTest(tamper=i):
                self.assertFalse(_valid_numform(self._tampered(mutate)))

    def test_protocol_tampers(self):
        def proto(payload):
            return payload["protocol"]

        tampers = [
            lambda p: proto(p).__setitem__("version", 2),
            lambda p: proto(p).__setitem__("seed", 9),
            lambda p: proto(p).__setitem__("prompt_templates_sha256", "0" * 64),
            lambda p: proto(p).__setitem__("fingerprint", "0" * 64),
            lambda p: proto(p).__setitem__("generator_sha256", "0" * 64),
            lambda p: proto(p).__setitem__("generation_budget", 1),
            lambda p: proto(p).__setitem__("space_codepoint", "U+0020"),
            lambda p: proto(p).__setitem__("backend_fingerprint", "sha256:other"),
            lambda p: proto(p)["prompt_templates"].__setitem__("copy", "tampered"),
            lambda p: proto(p).__setitem__("min_paired_evaluable", 1),
        ]
        for i, mutate in enumerate(tampers):
            with self.subTest(tamper=i):
                self.assertFalse(_valid_numform(self._tampered(mutate)))

    def test_grouping_tampers(self):
        def grouping(payload):
            return payload["grouping"]

        tampers = [
            lambda p: grouping(p).__setitem__("classification", "single_digit"),
            lambda p: grouping(p)["token_deltas_vs_plain"].__setitem__("plain", 0.5),
            lambda p: grouping(p)["form_token_means"].__setitem__("comma", None),
            lambda p: grouping(p)["evidence"].__setitem__(next(iter(grouping(p)["evidence"])), -1),
            lambda p: grouping(p).__setitem__("extra", 1),
        ]
        for i, mutate in enumerate(tampers):
            with self.subTest(tamper=i):
                self.assertFalse(_valid_numform(self._tampered(mutate)))

    def test_top_level_tampers(self):
        tampers = [
            lambda p: p.__setitem__("n_candidates", 7),
            lambda p: p.__setitem__("pooled", {"exact_rate": 0.5}),
            lambda p: p["candidates"]["nf-a"].__setitem__("winner", "nf-a"),
            lambda p: p.__setitem__("available", False),
            lambda p: p.pop("grouping"),
        ]
        for i, mutate in enumerate(tampers):
            with self.subTest(tamper=i):
                self.assertFalse(_valid_numform(self._tampered(mutate)))

    def test_records_bytes_cap(self):
        # 24 candidates at the record cap, each cell carrying a
        # 3000-digit input-token count (legal per-record: the
        # count is just an int >= 0), push the records' JSON
        # past the 12 MiB payload cap while every other rule
        # still holds -- the byte cap is the only refusal left.
        payload = copy.deepcopy(self.base)
        entry = payload["candidates"]["nf-a"]
        for i in range(22):
            payload["candidates"][f"nf-c{i:02d}"] = copy.deepcopy(entry)
        payload["n_candidates"] = 24
        self.assertTrue(_valid_numform(payload))
        giant = int("9" * 3000)
        for candidate in payload["candidates"].values():
            for record in candidate["records"]:
                record["cell"]["input_tokens"] = giant
        self.assertFalse(_valid_numform(payload))

    def test_record_count_cap(self):
        payload = copy.deepcopy(self.base)
        records = payload["candidates"]["nf-a"]["records"]
        records.extend(copy.deepcopy(records[:8]))
        self.assertFalse(_valid_numform(payload))

    def test_char_backend_payload_validates_with_grouping(self):
        backend = _CharBackend()
        stage = _run_stage(backend)
        payload = _payload(backend, stage)
        self.assertEqual(payload["grouping"]["classification"], "single_digit")
        self.assertTrue(_valid_numform(payload))


class TestDiffNumform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.faithful = _base_payload()
        cls.plain = _base_payload(mode="plain")

    def test_identical_payloads_unchanged(self):
        diff = diff_numform(self.faithful, copy.deepcopy(self.faithful))
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])
        self.assertTrue(diff["comparable"])
        self.assertFalse(diff["grouping_changed"])
        self.assertEqual(diff["rate_drift"], [])

    def test_rate_and_contrast_drift(self):
        diff = diff_numform(self.faithful, self.plain)
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["comparable"])
        metrics = {entry["metric"] for entry in diff["rate_drift"]}
        self.assertIn("copy_comma_exact_rate", metrics)
        contrast_metrics = {entry["metric"] for entry in diff["contrast_drift"]}
        self.assertTrue(any("classification" in metric for metric in contrast_metrics))

    def test_generator_drift_not_comparable(self):
        other = _base_payload(seed=1)
        diff = diff_numform(self.faithful, other)
        self.assertTrue(diff["generator_drift"])
        self.assertTrue(diff["protocol_drift"])
        self.assertFalse(diff["comparable"])
        self.assertTrue(diff["changed"])

    def test_backend_drift_not_comparable(self):
        other = _base_payload(fingerprint="sha256:someone-else")
        diff = diff_numform(self.faithful, other)
        self.assertTrue(diff["backend_drift"])
        self.assertFalse(diff["comparable"])

    def test_grouping_drift_flags_but_stays_comparable(self):
        backend = _CharBackend()
        stage = _run_stage(backend)
        char_payload = _payload(backend, stage)
        # Same backend fingerprint family is required for
        # comparability of the rates; the char payload has its
        # own fingerprint, so compare grouping drift on a pair
        # that shares the protocol except for grouping: reuse
        # the faithful payload and swap only its grouping block
        # for the char one (validation is diff-internal, and
        # _valid_numform would refuse the swap -- use two real
        # payloads instead and accept backend drift coexisting).
        diff = diff_numform(self.faithful, char_payload)
        self.assertTrue(diff["grouping_changed"])
        word = _base_payload()
        self.assertFalse(diff_numform(word, copy.deepcopy(word))["grouping_changed"])

    def test_status_drift(self):
        degraded = copy.deepcopy(self.plain)
        degraded["candidates"]["nf-b"] = {
            "status": "unavailable",
            "detail": "arm_evaluation_failed",
            "records": [],
        }
        diff = diff_numform(self.plain, degraded)
        self.assertTrue(diff["changed"])
        status_entries = [entry for entry in diff["rate_drift"] if entry["metric"] == "status"]
        self.assertEqual(len(status_entries), 1)
        self.assertEqual(status_entries[0]["candidate"], "nf-b")

    def test_candidates_added(self):
        backend = _NumformBackend()
        three = (
            FMT_A,
            FMT_B,
            Format(name="nf-c", bos="", user_prefix="U: ", assistant_prefix="A: ", turn_end="~"),
        )
        stage = numform_stage(
            backend,
            three,
            boundary=tuple(backend.encode(BOUNDARY_WORD)),
            max_new_tokens=64,
            max_context=8192,
            seed=0,
        )
        bigger = numform_analysis(stage, tuple(fmt.name for fmt in three))
        self.assertTrue(_valid_numform(bigger))
        diff = diff_numform(self.faithful, bigger)
        self.assertEqual(diff["candidates_added"], ["nf-c"])
        self.assertEqual(diff["candidates_removed"], [])
        reverse = diff_numform(bigger, self.faithful)
        self.assertEqual(reverse["candidates_removed"], ["nf-c"])

    def test_one_sided_availability(self):
        diff = diff_numform(self.faithful, None)
        self.assertEqual(diff["available"], "old")
        self.assertFalse(diff["changed"])
        diff = diff_numform(None, self.faithful)
        self.assertEqual(diff["available"], "new")
        diff = diff_numform(None, {"available": True})
        self.assertEqual(diff["available"], "neither")

    def test_diff_reports_integration(self):
        old = {"numform": self.faithful, "decision": "x", "selected_candidate": None}
        new = {"numform": self.plain, "decision": "x", "selected_candidate": None}
        report = diff_reports(old, new)
        self.assertEqual(report["numform"]["available"], "both")
        self.assertTrue(report["summary"]["numform_changed"])
        self.assertFalse(report["summary"]["identical"])
        same = diff_reports(old, copy.deepcopy(old))
        self.assertTrue(same["summary"]["identical"])
        self.assertFalse(same["summary"]["numform_changed"])


class TestReportingSection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = _base_payload(mode="plain")

    def test_section_renders(self):
        html = render_html({"numform": self.payload})
        self.assertIn('id="numform"', html)
        self.assertIn("Numeric surface form", html)
        self.assertIn("plain_advantage", html)

    def test_unknown_grouping_line(self):
        # The word-token backend's grouping classification is
        # unknown, so the section must say no grouping reading
        # is available rather than inventing one.
        html = render_html({"numform": self.payload})
        self.assertIn("unknown", html)

    def test_unavailable_candidate_row(self):
        payload = copy.deepcopy(self.payload)
        payload["candidates"]["nf-b"] = {
            "status": "unavailable",
            "detail": "arm_evaluation_failed",
            "records": [],
        }
        html = render_html({"numform": payload})
        self.assertIn('id="numform"', html)
        self.assertIn("nf-b", html)

    def test_section_absent_without_payload(self):
        self.assertNotIn('id="numform"', render_html({}))
        self.assertNotIn('id="numform"', render_html({"numform": None}))

    def test_malformed_shapes_do_not_raise(self):
        for bad in (
            {"available": True},
            {"available": True, "candidates": "junk"},
            {"available": True, "candidates": {"x": {"status": "available"}}},
            {
                "available": True,
                "candidates": {"x": {"status": "available", "aggregates": {"copy": "junk"}}},
            },
            {"available": False, "detail": "cancelled"},
        ):
            with self.subTest(bad=str(bad)[:40]):
                html = render_html({"numform": bad})
                self.assertIsInstance(html, str)


class TestCliWrapper(unittest.TestCase):
    def test_happy_path(self):
        backend = _NumformBackend()
        result: dict = {}
        _run_numform_probe(
            backend,
            FORMATS,
            result,
            boundary=tuple(backend.encode(BOUNDARY_WORD)),
            boundary_map=None,
            max_new_tokens=64,
            max_context=8192,
            seed=0,
            cancel=None,
        )
        self.assertIn("numform", result)
        self.assertTrue(_valid_numform(result["numform"]))

    def test_stage_failure_becomes_unavailable(self):
        backend = _NumformBackend()
        result: dict = {}
        with mock.patch(
            "model_protocol_lab.experiments.numform_stage",
            side_effect=RuntimeError("boom"),
        ):
            _run_numform_probe(
                backend,
                FORMATS,
                result,
                boundary=(1,),
                boundary_map=None,
                max_new_tokens=64,
                max_context=8192,
                seed=0,
                cancel=None,
            )
        self.assertEqual(result["numform"], {"available": False, "detail": "numform_unavailable"})


class TestGapCoverage(unittest.TestCase):
    """Targeted lines the first battery did not reach, found by
    intersecting the full-run coverage with the new code (the
    cycle-174 pattern): parser cap branches, generator retry
    paths, validator mirror branches, stage screen paths, and
    the reporting section's defensive lines.
    """

    # -- numform.py -------------------------------------------------

    def test_parse_digit_caps_per_grammar(self):
        grouped22 = "1,000,000,000,000,000,000,000"  # 22 digits
        self.assertIsNone(nf.parse_value(grouped22))
        self.assertIsNone(nf.parse_value(grouped22.replace(",", SPACE)))
        self.assertIsNone(nf.parse_value("123456789012345678901e5"))  # 21-digit mantissa
        self.assertEqual(nf.parse_value("1e999"), 10**999)  # grammar cap boundary

    def test_parse_negative_shift_quotient(self):
        self.assertEqual(nf.parse_value("1.250e2"), 125)
        self.assertEqual(nf.parse_value("12.50e1"), 125)
        self.assertIsNone(nf.parse_value("1.25e1"))  # 12.5 is not an integer

    def test_band_of_outside_bands(self):
        with self.assertRaises(ValueError):
            nf._band_of(10**12)

    def test_compare_partner_length_clamp(self):
        partner = nf._compare_partner(123456789012, "different_length", 2)
        self.assertEqual(len(str(partner)), 11)
        self.assertEqual(nf.margin_class(123456789012, partner), "different_length")
        partner = nf._compare_partner(7, "different_length", 3)
        self.assertEqual(len(str(partner)), 2)
        self.assertEqual(nf.margin_class(7, partner), "different_length")

    def test_generation_retries_out_of_range_partner(self):
        original = nf._compare_partner
        calls = {"n": 0}

        def flaky(a, target, derived):
            calls["n"] += 1
            if calls["n"] == 1:
                return -1
            return original(a, target, derived)

        with mock.patch.object(nf, "_compare_partner", flaky):
            value_set = nf.generate_value_set(0)
        self.assertEqual(len(value_set["compare"]), 16)
        for record in value_set["compare"]:
            self.assertEqual(record["stratum"], _margin_oracle(*record["operands"]))

    def test_generation_seed_type(self):
        for bad_seed in ("0", True, 0.5, None):
            with self.subTest(seed=bad_seed), self.assertRaises(ValueError):
                nf.generate_value_set(bad_seed)

    def test_grouping_zero_count_is_unknown(self):
        evidence = {
            probe: (len(probe) if (probe.isascii() and probe.isdigit()) else 3)
            for probe in nf.probe_strings()
        }
        evidence["1234567"] = 0
        self.assertEqual(nf.classify_grouping(evidence), "unknown")

    def test_paired_contrast_ignores_unpaired_values(self):
        records = [
            {
                "task": "copy",
                "form": "plain",
                "value_id": "nf-copy-00",
                "cell": {"verdict": True},
            },
            {
                "task": "copy",
                "form": "plain",
                "value_id": "nf-copy-01",
                "cell": {"verdict": True},
            },
            {
                "task": "copy",
                "form": "comma",
                "value_id": "nf-copy-01",
                "cell": {"verdict": False},
            },
        ]
        contrast = nf.paired_contrast(records, "copy", "plain", "comma")
        self.assertEqual(contrast["n_paired"], 1)
        self.assertEqual(contrast["n_pass_a_only"], 1)

    # -- diffreports mirror parity (direct calls) -------------------

    def test_validator_mirror_parity(self):
        import itertools

        values = [0, 1, 7, 42, 999, 1000, 1234567, 10**12 - 1, 999000000000]
        for value in values:
            for form in nf.FORMS:
                self.assertEqual(dr._numform_render(value, form), nf.render_form(value, form))
        atoms = [0, 1, 5, 9, 10, 55, 99, 100, 121, 129, 191, 500, 999, 1211, 1291]
        for a, b in itertools.product(atoms, atoms):
            self.assertEqual(dr._numform_carry(a, b), nf.carry_class(a, b), (a, b))
            self.assertEqual(dr._numform_margin(a, b), nf.margin_class(a, b), (a, b))
        self.assertEqual(dr._numform_margin(5, 5), "equal")
        self.assertEqual(dr._numform_margin(1211, 1291), "same_length_middle")

    def test_validator_grouping_mirror_parity(self):
        def evidence(counts):
            return {
                probe: counts(len(probe)) if (probe.isascii() and probe.isdigit()) else 3
                for probe in nf.probe_strings()
            }

        cases = [
            {},
            evidence(lambda n: n),
            evidence(lambda n: -(-n // 3)),
            evidence(lambda n: n + 1),
            evidence(lambda n: 1),
        ]
        irregular = evidence(lambda n: -(-n // 3))
        irregular["123456789012"] = 11
        cases.append(irregular)
        zeroed = evidence(lambda n: n)
        zeroed["12345"] = 0
        cases.append(zeroed)
        missing = evidence(lambda n: n)
        del missing["123456"]
        cases.append(missing)
        for item in cases:
            with self.subTest(head=str(item)[:30]):
                self.assertEqual(dr._numform_classify_grouping(item), nf.classify_grouping(item))

    def test_validator_contrast_mirror(self):
        for n_paired in (0, 3, 6, 16):
            for a_only in (0, 1, 2, 4):
                for b_only in (0, 1, 2, 4):
                    self.assertEqual(
                        dr._numform_classify_contrast(n_paired, a_only, b_only),
                        nf.classify_contrast(n_paired, a_only, b_only),
                        (n_paired, a_only, b_only),
                    )
        records = []
        for index in range(16):
            for form, verdict in (
                ("plain", index < 10),
                ("comma", index < 6 or index >= 12),
            ):
                records.append(
                    {
                        "task": "copy",
                        "form": form,
                        "value_id": f"nf-copy-{index:02d}",
                        "cell": {"verdict": verdict},
                    }
                )
        mirrored = dr._numform_contrast(records, "copy", "plain", "comma")
        self.assertEqual(mirrored, nf.paired_contrast(records, "copy", "plain", "comma"))
        self.assertEqual(mirrored["classification"], "mixed_discordance")
        unpaired = records + [
            {
                "task": "copy",
                "form": "plain",
                "value_id": "nf-copy-99",
                "cell": {"verdict": True},
            }
        ]
        self.assertEqual(
            dr._numform_contrast(unpaired, "copy", "plain", "comma"),
            nf.paired_contrast(unpaired, "copy", "plain", "comma"),
        )

    # -- validator tamper battery, round 2 --------------------------

    @classmethod
    def setUpClass(cls):
        cls.base = _base_payload()

    def _tampered(self, mutate):
        payload = copy.deepcopy(self.base)
        mutate(payload)
        return payload

    def test_record_guards_round_two(self):
        def first(payload, task, form="plain"):
            # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
            return next(
                r
                for r in payload["candidates"]["nf-a"]["records"]
                if r["task"] == task and r["form"] == form
            )

        def middle_compare(payload):
            # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
            record = next(
                r
                for r in payload["candidates"]["nf-a"]["records"]
                if r["task"] == "compare" and r["form"] == "plain" and r["band"] == 1
            )
            record["operands"] = [1211, 1291]
            record["expected_value"] = 1291
            record["expected_answer"] = "second"
            record["stratum"] = "same_length_middle"
            record["form_identical"] = True

        def wrong_compare_stratum(payload):
            record = first(payload, "compare")
            true_class = nf.margin_class(*record["operands"])
            # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
            record["stratum"] = next(
                c
                for c in (
                    "different_length",
                    "same_length_early",
                    "same_length_last_digit",
                )
                if c != true_class
            )

        def wrong_add_stratum(payload):
            record = first(payload, "add")
            true_class = nf.carry_class(*record["operands"])
            # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
            record["stratum"] = next(c for c in ("none", "some", "full_chain") if c != true_class)

        tampers = [
            lambda p: p["candidates"]["nf-a"]["records"].__setitem__(0, "junk"),
            lambda p: first(p, "copy").__setitem__("task", "bogus"),
            lambda p: first(p, "copy").__setitem__("value_id", "nf-copy-99"),
            lambda p: first(p, "copy").__setitem__("operands", "123"),
            lambda p: first(p, "copy")["operands"].__setitem__(0, -5),
            lambda p: first(p, "copy").__setitem__(
                "expected_value", first(p, "copy")["expected_value"] + 1
            ),
            lambda p: first(p, "compare").__setitem__(
                "expected_value", min(first(p, "compare")["operands"])
            ),
            lambda p: first(p, "copy").__setitem__("band", "0"),
            lambda p: first(p, "copy").__setitem__("stratum", "random"),
            wrong_compare_stratum,
            wrong_add_stratum,
            middle_compare,
            lambda p: first(p, "copy").__setitem__("form_identical", 1),
            # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
            lambda p: next(
                r
                for r in p["candidates"]["nf-a"]["records"]
                if r["form"] == "comma" and r["form_identical"] is False
            ).__setitem__("form_identical", True),
            lambda p: first(p, "copy").__setitem__("expected_answer_tokens", -1),
            lambda p: first(p, "copy").__setitem__("answer_value", -3),
            lambda p: first(p, "copy")["cell"].__setitem__("verdict", "yes"),
            lambda p: first(p, "copy")["cell"].__setitem__("boundary_text_match", "yes"),
        ]
        for i, mutate in enumerate(tampers):
            with self.subTest(tamper=i):
                self.assertFalse(_valid_numform(self._tampered(mutate)))

    def test_payload_guards_round_two(self):
        tampers = [
            lambda p: p.__setitem__("protocol", []),
            lambda p: p["protocol"].__setitem__("seed", "0"),
            lambda p: p["protocol"].__setitem__("backend_fingerprint", ""),
            lambda p: p.__setitem__("candidates", []),
            lambda p: p["candidates"].__setitem__("nf-a", "junk"),
            lambda p: p["candidates"].__setitem__(
                "nf-b",
                {"status": "unavailable", "detail": "x", "records": [{}]},
            ),
            lambda p: p["candidates"]["nf-a"].__setitem__("status", "weird"),
            lambda p: p["candidates"]["nf-a"].__setitem__("records", {}),
            lambda p: p["grouping"]["form_token_means"].__setitem__("plain", "x"),
            lambda p: p["grouping"]["token_deltas_vs_plain"].__setitem__("comma", "x"),
            lambda p: p["grouping"].__setitem__("classification", "chunked_up_to_3"),
            lambda p: p["candidates"]["nf-a"]["aggregates"]["copy"]["plain"].pop("decomposition"),
        ]
        for i, mutate in enumerate(tampers):
            with self.subTest(tamper=i):
                self.assertFalse(_valid_numform(self._tampered(mutate)))

    # -- experiments stage paths ------------------------------------

    def test_boundary_decode_failure_screens_nothing(self):
        class DecodeRaisesBackend(_NumformBackend):
            def decode(self, ids):
                if self.encode(BOUNDARY_WORD)[0] in ids:
                    raise RuntimeError("undecodable boundary")
                return super().decode(ids)

        backend = DecodeRaisesBackend()
        stage = _run_stage(backend)
        self.assertTrue(stage["available"], stage["detail"])
        records = stage["candidates"]["nf-a"]["records"]
        # The screen decoded nothing, so it screened nothing and
        # the run proceeded: every record still evaluated (the
        # answer span excludes the boundary token itself, so the
        # raising decode is never hit during evaluation).
        self.assertTrue(all(r["cell"] is not None for r in records))

    def test_boundary_text_in_template_screens(self):
        backend = _NumformBackend()
        stage = _run_stage(backend, boundary=tuple(backend.encode("number")))
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "numform_template_screen_failed")
        self.assertEqual(backend.generate_calls, 0)

    def test_stage_records_too_large(self):
        backend = _NumformBackend()
        with mock.patch.object(nf, "MAX_RECORDS_BYTES", 10):
            stage = _run_stage(backend)
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "numform_records_too_large")

    # -- reporting paths ---------------------------------------------

    def test_section_known_grouping_and_formatted_rates(self):
        backend = _CharBackend()
        payload = _payload(backend, _run_stage(backend))
        html = render_html({"numform": payload})
        self.assertIn("single_digit", html)
        self.assertIn("(1.000)", html)

    def test_section_rate_not_available(self):
        backend = _NumformBackend()
        stage = _run_stage(backend, max_context=8)
        payload = _payload(backend, stage)
        html = render_html({"numform": payload})
        self.assertIn("(n/a)", html)

    def test_section_defensive_continues(self):
        payload = {
            "available": True,
            "detail": "",
            "protocol": {},
            "grouping": {},
            "candidates": {
                "junk-entry": "not-a-dict",
                "bad-agg": {
                    "status": "available",
                    "detail": "",
                    "records": [],
                    "aggregates": {"copy": {"plain": "junk"}},
                    "contrasts": {"copy": {"plain__comma": "junk"}},
                },
            },
            "n_candidates": 2,
        }
        html = render_html({"numform": payload})
        self.assertIn('id="numform"', html)


class TestGapCoverageRoundThree(unittest.TestCase):
    """The validator guards the first two batteries still did not
    reach, mapped line-by-line from the full-run coverage
    intersection (each tamper is aimed at one named guard).
    """

    @classmethod
    def setUpClass(cls):
        cls.base = _base_payload()

    def _tampered(self, mutate):
        payload = copy.deepcopy(self.base)
        mutate(payload)
        return payload

    def test_construction_guards_round_three(self):
        def first(payload, task, form="plain", name="nf-a"):
            # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
            return next(
                r
                for r in payload["candidates"][name]["records"]
                if r["task"] == task and r["form"] == form
            )

        def equal_compare(payload):
            record = first(payload, "compare")
            record["operands"] = [500, 500]
            record["expected_value"] = 500

        def swapped_label(payload):
            record = first(payload, "compare")
            record["expected_answer"] = (
                "second" if record["expected_answer"] == "first" else "first"
            )

        tampers = [
            lambda p: first(p, "copy").__setitem__("expected_value", "5"),
            lambda p: first(p, "copy").__setitem__("expected_answer", 5),
            equal_compare,
            swapped_label,
            lambda p: first(p, "copy").__setitem__("detail", "not_a_detail"),
            lambda p: first(p, "copy").__setitem__("cell", {"verdict": True}),
        ]
        for i, mutate in enumerate(tampers):
            with self.subTest(tamper=i):
                self.assertFalse(_valid_numform(self._tampered(mutate)))

    def test_recompute_guards_round_three(self):
        def duplicate_within_candidate(payload):
            records = payload["candidates"]["nf-a"]["records"]
            records[5] = copy.deepcopy(records[0])

        def cross_candidate_construction_drift(payload):
            # The same (task, value, form) key carrying a
            # different -- individually consistent -- value in
            # the second candidate: construction tuples differ.
            # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
            record = next(
                r
                for r in payload["candidates"]["nf-b"]["records"]
                if r["value_id"] == "nf-copy-00" and r["form"] == "plain"
            )
            record["operands"] = [101]
            record["expected_value"] = 101
            record["expected_answer"] = "101"
            record["stratum"] = "random"
            record["answer_value"] = 101
            record["cell"]["verdict"] = False

        def no_available_candidates(payload):
            for name in ("nf-a", "nf-b"):
                payload["candidates"][name] = {
                    "status": "unavailable",
                    "detail": "numform_stage_unavailable",
                    "records": [],
                }

        def drop_one_form_of_one_value(payload):
            for name in ("nf-a", "nf-b"):
                entry = payload["candidates"][name]
                entry["records"] = [
                    r
                    for r in entry["records"]
                    if not (r["value_id"] == "nf-copy-00" and r["form"] == "comma")
                ]

        tampers = [
            duplicate_within_candidate,
            cross_candidate_construction_drift,
            no_available_candidates,
            drop_one_form_of_one_value,
            lambda p: p["grouping"].__setitem__("evidence", "x"),
            lambda p: p["grouping"].__setitem__("evidence", {"only-one": 3}),
            lambda p: p["grouping"].__setitem__("form_token_means", []),
            lambda p: p["grouping"].__setitem__("token_deltas_vs_plain", []),
        ]
        for i, mutate in enumerate(tampers):
            with self.subTest(tamper=i):
                self.assertFalse(_valid_numform(self._tampered(mutate)))
