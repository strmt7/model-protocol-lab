"""Core contract tests using a scripted, non-model backend."""

import unittest
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from threading import Event
from unittest.mock import patch

from model_protocol_lab.core import (
    EvaluationReport,
    Fixture,
    Format,
    Generation,
    Message,
    Split,
    evaluate,
    first_boundary,
    local_repairs,
)


def ids(text):
    """Unicode-codepoint test codec; no genuine tokenizer semantics."""
    return tuple(map(ord, text))


BASE = Format("base", "<B>", "<U>", "<A>", "<E>")
RAW = ids("ok|tail")
GOOD = Generation(RAW, "limit")
PRIVATE = "synthetic-private-diagnostic-do-not-export"


def case(name="one", *, expected="ok", group=None, messages=None):
    return Fixture(
        name,
        (Message("user", "question:" + name),) if messages is None else messages,
        expected,
        "group:" + name if group is None else group,
    )


class FakeBackend:
    fingerprint = "scripted-fixture-not-a-model"

    def __init__(
        self,
        generation=GOOD,
        *,
        encode_script=(),
        generate_script=(),
        decode_script=(),
        cancel_at=None,
        event=None,
    ):
        self.generation = generation
        self.encode_script = list(encode_script)
        self.generate_script = list(generate_script)
        self.decode_script = list(decode_script)
        self.cancel_at, self.event = cancel_at, event
        self.encoded, self.generated, self.decoded = [], [], []

    @staticmethod
    def take(script, fallback):
        value = script.pop(0) if script else fallback
        if isinstance(value, BaseException):
            raise value
        return value

    def trigger(self, phase, event=None):
        if self.cancel_at == phase:
            target = event if event is not None else self.event
            if target is None:
                target = self.generated[-1][3]
            target.set()

    def encode(self, text):
        self.encoded.append(text)
        value = self.take(self.encode_script, ids(text))
        self.trigger("encode")
        return value

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        self.generated.append((input_ids, max_new_tokens, seed, cancel))
        value = self.take(self.generate_script, self.generation)
        self.trigger("generate", cancel)
        return value

    def decode(self, output_ids):
        self.decoded.append(output_ids)
        value = self.take(self.decode_script, "".join(map(chr, output_ids)))
        self.trigger("decode")
        return value


def run_case(backend, *, cases=None, candidate=BASE, **kwargs):
    options = {"boundary": ids("|"), "max_new_tokens": len(RAW), "seed": 0}
    options.update(kwargs)
    return evaluate(backend, candidate, (case(),) if cases is None else cases, **options)


class ValueTests(unittest.TestCase):
    def test_message_accepts_supported_roles_and_literal_text(self):
        for role in ("user", "assistant"):
            for text in ("", " \ntext\t", "café", "<B><A>literal"):
                with self.subTest(role=role, text=text):
                    self.assertEqual(Message(role, text).content, text)

    def test_message_rejects_bad_roles_and_content(self):
        for role in ("system", "tool", "", "User", None, 1):
            with self.subTest(role=role), self.assertRaises(ValueError):
                Message(role, "q")
        for text in (None, 1, True, b"q", ["q"]):
            with self.subTest(text=text), self.assertRaises(ValueError):
                Message("user", text)

    def test_format_validates_field_types(self):
        for field in ("name", "bos", "user_prefix", "assistant_prefix", "turn_end"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(BASE, **{field: None})
        for name in ("", " \n\t"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                replace(BASE, name=name)
        for value in (0, 1, None, "yes"):
            with self.subTest(repeat=value), self.assertRaises(ValueError):
                replace(BASE, repeat_bos=value)

    def test_fixture_validation(self):
        for field in ("id", "group"):
            for value in ("", " \n", None, 1):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(case(), **{field: value})
        for expected in (None, 1, False, b"ok"):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                case(expected=expected)
        for messages in ((), [], [Message("user", "q")], (object(),)):
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                case(messages=messages)
        self.assertEqual(case(expected="").expected, "")

    def test_value_records_are_frozen(self):
        report = run_case(FakeBackend())
        pairs = (
            (Message("user", "q"), "content"),
            (BASE, "bos"),
            (case(), "expected"),
            (GOOD, "token_ids"),
            (Split((case("s"),), (case("h"),)), "search"),
            (report.observations[0], "answer"),
            (report, "status"),
        )
        for value, field in pairs:
            with self.subTest(field=field), self.assertRaises(FrozenInstanceError):
                setattr(value, field, None)


class RenderTests(unittest.TestCase):
    def test_single_and_multi_turn_exact_rendering(self):
        self.assertEqual(BASE.render((Message("user", "q"),)), "<B><U>q<E><A>")
        history = (Message("user", "q"), Message("assistant", "a"), Message("user", "r"))
        self.assertEqual(BASE.render(history), "<B><U>q<E><A>a<E><U>r<E><A>")

    def test_repeat_bos_includes_final_open_assistant(self):
        fmt = replace(BASE, repeat_bos=True)
        self.assertEqual(fmt.render((Message("user", "q"),)), "<B><U>q<E><B><A>")
        history = (Message("user", "q"), Message("assistant", "a"), Message("user", "r"))
        self.assertEqual(fmt.render(history), "<B><U>q<E><B><A>a<E><B><U>r<E><B><A>")

    def test_content_is_preserved_without_interpretation(self):
        text = " \n<B><A>literal\t"
        self.assertEqual(BASE.render((Message("user", text),)), "<B><U>" + text + "<E><A>")
        empty = Format("empty", "", "", "", "", True)
        history = (Message("user", "q "), Message("assistant", "a\n"), Message("user", "r"))
        self.assertEqual(empty.render(history), "q a\nr")

    def test_invalid_message_orders_and_containers(self):
        invalid = (
            (),
            None,
            [],
            [Message("user", "q")],
            ("q",),
            (Message("assistant", "a"),),
            (Message("user", "q"), Message("assistant", "a")),
            (Message("user", "q"), Message("user", "r")),
            (Message("assistant", "a"), Message("user", "r")),
        )
        for messages in invalid:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                BASE.render(messages)

    def test_repairs_are_bounded_ordered_and_do_not_mutate_base(self):
        base = replace(BASE, assistant_prefix="<A> \t")
        before = replace(base)
        variants = local_repairs(base)
        self.assertEqual(base, before)
        self.assertEqual(
            [v.name for v in variants],
            ["base", "base/toggle-bos", "base/no-bos", "base/prefix-newline", "base/prefix-trim"],
        )
        self.assertIs(variants[0], base)
        self.assertTrue(variants[1].repeat_bos)
        self.assertEqual(variants[2].bos, "")
        self.assertEqual(variants[3].assistant_prefix, "<A> \t\n")
        self.assertEqual(variants[4].assistant_prefix, "<A>")
        reverse = local_repairs(replace(base, repeat_bos=True))
        self.assertFalse(reverse[1].repeat_bos)

    def test_repairs_deduplicate_noops(self):
        self.assertEqual(len(local_repairs(BASE)), 4)
        for repeat in (True, False):
            with self.subTest(repeat=repeat):
                base = replace(BASE, bos="", repeat_bos=repeat)
                result = local_repairs(base)
                self.assertEqual([v.name for v in result], ["base", "base/prefix-newline"])
                self.assertIs(result[0], base)


class SplitTests(unittest.TestCase):
    def test_valid_groups_and_same_answers(self):
        split = Split((case("a", group="s"), case("b", group="s")), (case("c"),))
        self.assertEqual(len(split.search), 2)
        self.assertEqual(split.search[0].expected, split.held_out[0].expected)

    def test_invalid_split_containers_and_items(self):
        a, b = (case("a"),), (case("b"),)
        for left, right in (
            ((), b),
            (a, ()),
            (list(a), b),
            (a, list(b)),
            ((object(),), b),
            (a, (object(),)),
        ):
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                Split(left, right)

    def test_duplicate_ids_rejected_everywhere(self):
        a, b, c = case("a"), replace(case("b"), id="a"), case("c")
        for left, right in (((a, b), (c,)), ((c,), (a, b)), ((a,), (b,))):
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                Split(left, right)

    def test_group_and_exact_input_leakage_rejected(self):
        with self.assertRaises(ValueError):
            Split((case("s", group="shared"),), (case("h", group="shared"),))
        s = case("s")
        h = case("h", messages=s.messages, expected="different label")
        with self.assertRaises(ValueError):
            Split((s,), (h,))

    def test_duplicate_inputs_within_partition_are_not_cross_split_leakage(self):
        a = case("a")
        split = Split((a, case("b", messages=a.messages)), (case("h"),))
        self.assertEqual(len(split.search), 2)


class BoundaryTests(unittest.TestCase):
    def test_matching_and_nonmatching_boundaries(self):
        samples = (
            ((1,), (1,), 0),
            ((1, 2), (1, 2), 0),
            ((9, 1, 2), (1, 2), 1),
            ((1, 2, 1, 2), (1, 2), 0),
            ((1, 1, 1), (1, 1), 0),
            ((1, 1, 2), (1, 2), 1),
            ((), (1,), None),
            ((1,), (1, 2), None),
            ((1, 2), (1, 2, 3), None),
            ((9, 1), (1, 2), None),
            ((1, 2), (3,), None),
        )
        for tokens, stop, expected in samples:
            with self.subTest(tokens=tokens, stop=stop):
                self.assertEqual(first_boundary(tokens, stop), expected)

    def test_empty_boundary_rejected(self):
        for tokens in ((), (1, 2)):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                first_boundary(tokens, ())


class EvaluationTests(unittest.TestCase):
    def test_completed_report_preserves_full_evidence(self):
        backend = FakeBackend()
        fixture = case()
        report = run_case(backend, cases=(fixture,))
        obs = report.observations[0]
        self.assertEqual(
            (report.status, report.planned_count, report.attempted_count), ("completed", 1, 1)
        )
        self.assertTrue(report.complete and report.aggregate_eligible)
        self.assertEqual((obs.candidate, obs.fixture), ("base", "one"))
        self.assertEqual(obs.input_ids, ids(BASE.render(fixture.messages)))
        self.assertEqual(obs.output_ids, RAW)
        self.assertEqual(
            (obs.answer, obs.raw_reason, obs.error, obs.status), ("ok", "limit", None, "completed")
        )
        self.assertTrue(obs.exact_match and obs.boundary_found)
        # decode calls: answer ids, boundary ids, retained ids (diagnostic).
        self.assertEqual(backend.decoded, [ids("ok"), ids("|"), ids("ok|")])
        self.assertEqual(backend.encoded, [BASE.render(fixture.messages)])

    def test_exact_match_boundary_and_suffix_are_separate(self):
        samples = (
            ("ok", "ok", "|", True, False),
            ("ok|ignored|suffix", "ok", "|", True, True),
            ("|suffix", "", "|", True, True),
            ("ok|", "ok", "|", True, True),
            ("ok|", "ok|", "||", True, False),
        )
        for raw, expected, stop, match, found in samples:
            with self.subTest(raw=raw, stop=stop):
                backend = FakeBackend(Generation(ids(raw), "limit"))
                obs = run_case(
                    backend,
                    cases=(case(expected=expected),),
                    max_new_tokens=len(raw),
                    boundary=ids(stop),
                ).observations[0]
                self.assertEqual(obs.exact_match, match)
                self.assertEqual(obs.boundary_found, found)
                self.assertEqual(obs.output_ids, ids(raw))

    def test_exact_matching_performs_no_normalization(self):
        for actual, expected in (
            ("OK", "ok"),
            (" ok", "ok"),
            ("ok ", "ok"),
            ("ok\n", "ok"),
            ("a  b", "a b"),
            ("e\u0301", "é"),
        ):
            with self.subTest(actual=actual):
                raw = ids(actual + "|")
                obs = run_case(
                    FakeBackend(Generation(raw, "limit")),
                    cases=(case(expected=expected),),
                    max_new_tokens=len(raw),
                ).observations[0]
                self.assertFalse(obs.exact_match)
                self.assertEqual(obs.answer, actual)
                self.assertEqual(obs.status, "completed")

    def test_explicit_boundary_ids_are_not_reencoded(self):
        stop = (987654321, 987654322)
        raw = ids("ok") + stop + (999999999,)
        backend = FakeBackend(Generation(raw, "limit"))
        obs = run_case(backend, boundary=stop, max_new_tokens=len(raw)).observations[0]
        self.assertTrue(obs.exact_match and obs.boundary_found)
        self.assertEqual(backend.decoded, [ids("ok"), stop])
        self.assertEqual(len(backend.encoded), 1)
        # The oversized boundary IDs are not valid text: the text cross-check
        # degrades to None instead of failing the run.
        self.assertIsNone(obs.boundary_text_match)

    def test_multi_fixture_order_and_output_cap_endpoints(self):
        report = run_case(FakeBackend(), cases=(case("c"), case("a"), case("b")))
        self.assertEqual([o.fixture for o in report.observations], ["c", "a", "b"])
        self.assertEqual(report.attempted_count, 3)
        for cap in (1, 4096):
            with self.subTest(cap=cap):
                text = "x" * cap
                backend = FakeBackend(Generation(ids(text), "limit"))
                obs = run_case(
                    backend, cases=(case(expected=text),), max_new_tokens=cap
                ).observations[0]
                self.assertEqual(backend.generated[0][1], cap)
                self.assertTrue(obs.exact_match)

    def test_default_cap_is_64(self):
        backend = FakeBackend(Generation(ids("x" * 64), "limit"))
        report = evaluate(backend, BASE, (case(expected="x" * 64),), boundary=ids("|"))
        self.assertEqual(backend.generated[0][1], 64)
        self.assertTrue(report.complete)

    def test_invalid_arguments_fail_before_backend_calls(self):
        bad = [
            {"candidate": None},
            {"cases": ()},
            {"cases": [case()]},
            {"cases": (object(),)},
            {"cases": (case(), case())},
        ]
        bad += [{"max_new_tokens": v} for v in (-1, 0, 4097, True, False, 1.0, None, "2")]
        bad += [{"boundary": v} for v in ((), [], [1], None, 1, (True,), (-1,), (1.0,), ("1",))]
        bad += [{"seed": v} for v in (True, False, 1.0, None, "1", -1, 2**64, 1 << 20000)]
        for index, options in enumerate(bad):
            with self.subTest(index=index):
                backend = FakeBackend()
                with self.assertRaises(ValueError):
                    run_case(backend, **options)
                self.assertEqual(
                    (backend.encoded, backend.generated, backend.decoded), ([], [], [])
                )

    def test_invalid_encoded_tokens_are_not_retained(self):
        for payload in (None, [], [1, 2], (True,), (-1,), (1.0,), ("1",)):
            with self.subTest(payload=payload):
                backend = FakeBackend(encode_script=[payload])
                obs = run_case(backend).observations[0]
                self.assertEqual(obs.error, "invalid_input_ids")
                self.assertEqual((obs.input_ids, obs.output_ids), ((), ()))
                self.assertFalse(obs.exact_match or obs.boundary_found)
                self.assertEqual(backend.generated, [])

    def test_empty_encoded_tuple_is_permitted(self):
        backend = FakeBackend(encode_script=[()])
        report = run_case(backend)
        self.assertEqual(backend.generated[0][0], ())
        self.assertTrue(report.complete)

    def test_malformed_generation_fails_closed(self):
        raw = ids("ok|")
        bad = (
            None,
            object(),
            {"token_ids": raw, "reason": "limit"},
            Generation(list(raw), "limit"),
            Generation((True, 2, 3), "limit"),
            Generation((-1, 2, 3), "limit"),
            Generation((1.0, 2, 3), "limit"),
            Generation(("1", 2, 3), "limit"),
            Generation(raw, "unknown"),
            Generation(raw, None),
            Generation(raw, "limit", PRIVATE),
            Generation(raw, "cancelled", PRIVATE),
            Generation(raw, "error", 123),
            Generation((), "limit"),
            Generation(ids("ok"), "limit"),
        )
        for index, result in enumerate(bad):
            with self.subTest(index=index):
                backend = FakeBackend(result)
                report = run_case(backend, max_new_tokens=3)
                obs = report.observations[0]
                self.assertEqual((report.status, obs.error), ("failed", "invalid_generation"))
                self.assertEqual(obs.output_ids, ())
                self.assertIsNone(obs.raw_reason)
                self.assertFalse(obs.exact_match or obs.boundary_found)
                self.assertFalse(report.aggregate_eligible)
                self.assertEqual(backend.decoded, [])
                self.assertNotIn(PRIVATE, repr(report))

    def test_all_reasons_must_respect_output_cap(self):
        for reason in ("limit", "cancelled", "error"):
            with self.subTest(reason=reason):
                obs = run_case(
                    FakeBackend(Generation(ids("over"), reason)), max_new_tokens=3
                ).observations[0]
                self.assertEqual(obs.error, "invalid_generation")
                self.assertEqual(obs.output_ids, ())

    def test_nonstring_decode_results_are_failures_with_raw_evidence(self):
        for value in (None, 7, b"ok", ["ok"]):
            with self.subTest(value=value):
                obs = run_case(FakeBackend(decode_script=[value])).observations[0]
                self.assertEqual((obs.error, obs.answer), ("decode_failed", ""))
                self.assertEqual(obs.output_ids, RAW)
                self.assertTrue(obs.boundary_found)
                self.assertFalse(obs.exact_match)

    def test_reported_backend_errors_keep_boundary_but_hide_diagnostics(self):
        for error in (None, "", PRIVATE):
            with self.subTest(error=error):
                backend = FakeBackend(Generation(ids("ok|"), "error", error))
                report = run_case(backend, cases=(case("first"), case("later")))
                obs = report.observations[0]
                self.assertEqual(
                    (report.status, report.planned_count, report.attempted_count), ("failed", 2, 1)
                )
                self.assertEqual(obs.error, "backend_reported_error")
                self.assertEqual(obs.raw_reason, "error")
                self.assertTrue(obs.boundary_found)
                self.assertFalse(obs.exact_match or report.aggregate_eligible)
                self.assertNotIn(PRIVATE, repr(report))
                self.assertEqual(backend.decoded, [])

    def test_each_failure_stage_retains_prior_records_and_stops(self):
        stages = (
            ("render", "render_failed"),
            ("encode", "encode_failed"),
            ("input", "invalid_input_ids"),
            ("generate", "generate_failed"),
            ("generation", "invalid_generation"),
            ("decode", "decode_failed"),
        )
        for stage, code in stages:
            with self.subTest(stage=stage):
                first, second, third = case("first"), case("second"), case("third")
                error, options = RuntimeError(PRIVATE), {}
                if stage == "render":
                    second = replace(second, messages=(Message("assistant", "bad"),))
                elif stage == "encode":
                    options["encode_script"] = [ids(BASE.render(first.messages)), error]
                elif stage == "input":
                    options["encode_script"] = [ids(BASE.render(first.messages)), [1, 2]]
                elif stage == "generate":
                    options["generate_script"] = [GOOD, error]
                elif stage == "generation":
                    options["generate_script"] = [GOOD, Generation((), "limit")]
                elif stage == "decode":
                    # Per-case decode sequence: answer ids, boundary ids,
                    # retained ids (diagnostic); the failure lands on the
                    # second case's answer decode.
                    options["decode_script"] = ["ok", "|", "ok|", error]
                report = run_case(FakeBackend(**options), cases=(first, second, third))
                before, failed = report.observations
                self.assertEqual(
                    (report.status, report.planned_count, report.attempted_count), ("failed", 3, 2)
                )
                self.assertTrue(before.exact_match)
                self.assertEqual(
                    (failed.fixture, failed.status, failed.error), ("second", "failed", code)
                )
                self.assertFalse(failed.exact_match or report.aggregate_eligible)
                self.assertNotIn(PRIVATE, repr(report))
                self.assertEqual(
                    failed.input_ids,
                    ()
                    if stage in ("render", "encode", "input")
                    else ids(BASE.render(second.messages)),
                )
                self.assertEqual(failed.output_ids, RAW if stage == "decode" else ())
                self.assertEqual(failed.boundary_found, stage == "decode")

    def test_failures_take_precedence_over_concurrent_cancellation(self):
        for result, code in (
            (Generation((), "limit"), "invalid_generation"),
            (Generation(RAW, "error", PRIVATE), "backend_reported_error"),
        ):
            with self.subTest(code=code):
                event = Event()
                report = run_case(FakeBackend(result, cancel_at="generate"), cancel=event)
                self.assertTrue(event.is_set())
                self.assertEqual(report.status, "failed")
                self.assertEqual(report.observations[0].error, code)
        event = Event()
        report = run_case(
            FakeBackend(decode_script=[None], cancel_at="decode", event=event), cancel=event
        )
        self.assertTrue(event.is_set())
        self.assertEqual(report.observations[0].error, "decode_failed")

    def test_last_fixture_failure_is_not_complete(self):
        backend = FakeBackend(generate_script=[GOOD, RuntimeError(PRIVATE)])
        report = run_case(backend, cases=(case("a"), case("b")))
        self.assertEqual(report.attempted_count, report.planned_count)
        self.assertFalse(report.complete or report.aggregate_eligible)
        self.assertEqual(report.status, "failed")


class SeedTests(unittest.TestCase):
    def test_uint64_seed_derivation_including_unicode_and_surrogates(self):
        for seed in (0, 17, 2**64 - 1):
            for name in ("one", "unicode-é", "nul-\0-id", "surrogate-\ud800"):
                with self.subTest(seed=seed, name=name):
                    backend = FakeBackend()
                    run_case(backend, cases=(case(name),), seed=seed)
                    material = f"{seed}\0{name}".encode("utf-8", errors="surrogatepass")
                    expected = int.from_bytes(sha256(material).digest()[:4], "big")
                    self.assertEqual(backend.generated[0][2], expected)

    def test_fixture_order_does_not_change_seeds(self):
        cases = (case("alpha"), case("beta"), case("gamma"))
        first, second = FakeBackend(), FakeBackend()
        run_case(first, cases=cases, seed=41)
        run_case(second, cases=tuple(reversed(cases)), seed=41)
        a = {c.id: call[2] for c, call in zip(cases, first.generated, strict=True)}
        b = {c.id: call[2] for c, call in zip(reversed(cases), second.generated, strict=True)}
        self.assertEqual(a, b)

    def test_candidates_share_fixture_seed_without_sharing_prompt(self):
        alternate = replace(BASE, name="alternate", assistant_prefix="<OTHER>", repeat_bos=True)
        first, second = FakeBackend(), FakeBackend()
        run_case(first, seed=37)
        run_case(second, candidate=alternate, seed=37)
        self.assertEqual(first.generated[0][2], second.generated[0][2])
        self.assertNotEqual(first.generated[0][0], second.generated[0][0])


class CancellationTests(unittest.TestCase):
    def test_pre_cancelled_request_does_no_work(self):
        event, backend = Event(), FakeBackend()
        event.set()
        report = run_case(backend, cases=(case("a"), case("b")), cancel=event)
        self.assertEqual(
            (report.status, report.planned_count, report.attempted_count), ("cancelled", 2, 0)
        )
        self.assertEqual((backend.encoded, backend.generated, backend.decoded), ([], [], []))
        self.assertFalse(report.aggregate_eligible)

    def test_cancel_after_render_skips_encode(self):
        event, backend, original = Event(), FakeBackend(), BASE.render

        def cancelling_render(messages):
            text = original(messages)
            event.set()
            return text

        with patch.object(Format, "render", side_effect=cancelling_render):
            report = run_case(backend, cancel=event)
        obs = report.observations[0]
        self.assertEqual((report.status, report.attempted_count), ("cancelled", 1))
        self.assertEqual((obs.input_ids, obs.output_ids), ((), ()))
        self.assertFalse(obs.exact_match)
        self.assertEqual(backend.encoded, [])

    def test_cancel_at_encode_generate_and_decode_checkpoints(self):
        for phase in ("encode", "generate", "decode"):
            with self.subTest(phase=phase):
                event = Event()
                backend = FakeBackend(cancel_at=phase, event=event)
                report = run_case(backend, cases=(case("a"), case("later")), cancel=event)
                obs = report.observations[0]
                self.assertEqual((report.status, report.attempted_count), ("cancelled", 1))
                self.assertEqual(obs.status, "cancelled")
                self.assertFalse(obs.exact_match or report.aggregate_eligible)
                self.assertIsNone(obs.error)
                self.assertEqual(obs.input_ids, ids(BASE.render(case("a").messages)))
                self.assertEqual(obs.output_ids, () if phase == "encode" else RAW)
                self.assertEqual(obs.raw_reason, None if phase == "encode" else "limit")
                self.assertEqual(obs.boundary_found, phase != "encode")
                self.assertEqual(obs.answer, "ok" if phase == "decode" else "")
                if phase == "encode":
                    self.assertEqual(backend.generated, [])
                if phase != "decode":
                    self.assertEqual(backend.decoded, [])

    def test_backend_cancellation_stops_even_without_event(self):
        for raw, found in (((), False), (ids("ok|"), True)):
            with self.subTest(raw=raw):
                event = Event()
                backend = FakeBackend(Generation(raw, "cancelled"))
                report = run_case(backend, cases=(case("a"), case("b")), cancel=event)
                obs = report.observations[0]
                self.assertFalse(event.is_set())
                self.assertEqual((report.status, report.attempted_count), ("cancelled", 1))
                self.assertEqual(obs.output_ids, raw)
                self.assertEqual(obs.boundary_found, found)
                self.assertFalse(obs.exact_match)
                self.assertIsNone(obs.error)
                self.assertEqual(backend.decoded, [])

    def test_events_shared_within_run_and_fresh_across_runs(self):
        backend = FakeBackend()
        run_case(backend, cases=(case("a"), case("b")))
        run_case(backend, cases=(case("c"),))
        self.assertIs(backend.generated[0][3], backend.generated[1][3])
        self.assertIsNot(backend.generated[0][3], backend.generated[2][3])
        event, explicit = Event(), FakeBackend()
        run_case(explicit, cases=(case("a"), case("b")), cancel=event)
        self.assertTrue(all(call[3] is event for call in explicit.generated))
        self.assertFalse(event.is_set())

    def test_last_attempt_cancelled_is_not_complete(self):
        backend = FakeBackend(generate_script=[GOOD, Generation(ids("ok|"), "cancelled")])
        report = run_case(backend, cases=(case("a"), case("b")))
        self.assertEqual(report.attempted_count, report.planned_count)
        self.assertTrue(report.observations[0].exact_match)
        self.assertFalse(report.observations[1].exact_match)
        self.assertFalse(report.complete or report.aggregate_eligible)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.good = run_case(FakeBackend()).observations[0]
        self.cancelled = replace(self.good, status="cancelled", exact_match=False)
        self.failed = replace(self.good, status="failed", exact_match=False, error="decode_failed")

    def test_valid_complete_and_cancelled_prefix_reports(self):
        self.assertTrue(EvaluationReport((self.good,), 1, "completed").aggregate_eligible)
        for observations, planned in (((), 2), ((self.good,), 2), ((self.cancelled,), 1)):
            with self.subTest(planned=planned, count=len(observations)):
                report = EvaluationReport(observations, planned, "cancelled")
                self.assertFalse(report.complete or report.aggregate_eligible)
                self.assertEqual(report.attempted_count, len(observations))

    def test_invalid_reports_rejected(self):
        g, c, f = self.good, self.cancelled, self.failed
        invalid = (
            ((g,), 0, "completed"),
            ((g,), -1, "completed"),
            ((g,), True, "completed"),
            ((g,), 1.0, "completed"),
            ([g], 1, "completed"),
            ((object(),), 1, "completed"),
            ((g,), 1, "unknown"),
            ((g, g), 1, "completed"),
            ((), 1, "completed"),
            ((g,), 2, "completed"),
            ((c,), 1, "completed"),
            ((), 1, "failed"),
            ((g,), 1, "failed"),
            ((f, g), 2, "failed"),
            ((c,), 1, "failed"),
            ((f,), 1, "cancelled"),
            ((replace(g, status="unknown"),), 1, "cancelled"),
        )
        for index, args in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                EvaluationReport(*args)


if __name__ == "__main__":
    unittest.main()
