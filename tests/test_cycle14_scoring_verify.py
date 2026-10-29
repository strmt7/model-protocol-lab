"""Cycle 14: differential verification of the teacher-forced scoring engine.

Applies the same differential rigor previously used for the three backend
stop emulations to ``score_target_logprobs`` / ``score_span_from_logits``
(the numerical heart of the logprob diagnostic) and to ``check_alignment``
(the §9.3 boundary guard).

Method: every expectation below comes from an INDEPENDENT reference written
from the math definition -- plain-Python naive log-softmax
(``x_t - log(sum(exp(x)))``, deliberately *not* the max-shifted formulation
the engine uses), ``math.fsum`` accumulation, and ``struct`` float32
rounding -- never a port of, or wrapper around, the code under test. The
alignment oracle is an independently written predicate over an independently
written greedy piece-table codec.

Properties under test
---------------------
1. Causal shift: target token ``i`` is scored from logit row
   ``prompt_len - 1 + i`` (off-by-one at the prompt/answer boundary would
   fail the seeded fuzz with probability ~1).
2. Log-softmax: correct normalization and numerical stability (extreme
   logits ±1e30, exact ties, single-token vocabularies).
3. Gather: the correct target id is read from each predicting row.
4. Aggregation: per-token values float32-rounded (locked exactly:
   every reported value is float32-representable); the total accumulates
   float64 *unrounded* values (a 20k-token test separates float64 from
   float32 accumulation; a constructive 50k-token identical-rows test
   separates unrounded from rounded-first accumulation); the mean divides
   the unrounded total; the empty span yields ``((), 0.0, None)``.
5. Masking: rows that predict prompt tokens (``0 .. prompt_len-2``) and the
   final row (predicts past the span) are never read -- randomizing them
   cannot move the score.
6. ``check_alignment``: differential agreement with the oracle on genuine
   splits (liveness: genuine splits of a greedy encoding MUST verify),
   adversarial id sequences, and constructed straddles; plus the safety
   direction -- whenever the engine reports ok, the concatenated text
   re-encodes to the concatenated ids, and whenever a re-parsed piece
   strictly contains the character boundary the engine MUST refuse.
7. ``score_target_logprobs`` orchestration: exactly one forward pass over
   ``prompt_ids + target_ids``, field-for-field agreement with the
   reference, fixed reason codes, no forward call on refusal paths.
"""

from __future__ import annotations

import math
import random
import struct
import unittest

import torch

from model_protocol_lab.scoring import (
    STATUS_OK,
    STATUS_OK_EMPTY,
    STATUS_UNAVAILABLE,
    check_alignment,
    score_span_from_logits,
    score_target_logprobs,
)

# ---------------------------------------------------------------------------
# Independent reference (math definition, no torch, no max-shift).
# ---------------------------------------------------------------------------


def ref_f32(value: float) -> float:
    """Independent float32 rounding: the definition of binary32 rounding."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def ref_logprob_row(row: list[float], token_id: int) -> float:
    """Naive log-softmax from the definition: x_t - log(sum_v exp(x_v)).

    Deliberately avoids the max-shift formulation used by the engine so the
    cross-check exercises a different floating-point path. Only used with
    moderate-magnitude logits where the naive formula is safe; extremes get
    hand-derived closed forms instead.
    """
    log_z = math.log(math.fsum(math.exp(x) for x in row))
    return row[token_id] - log_z


def ref_score_span(
    logit_rows: list[list[float]],
    prompt_len: int,
    target_ids: tuple[int, ...],
) -> tuple[tuple[float, ...], float, float | None]:
    """Independent (per-token, total, mean) triple from the math definition.

    Gathers row ``prompt_len - 1 + i`` for target token ``i`` (the causal
    shift), float32-rounds each per-token value, accumulates the *unrounded*
    float64 values with ``math.fsum``, float32-rounds the total, and divides
    the unrounded total for the mean (None for an empty span).
    """
    per_token_unrounded = [
        ref_logprob_row(logit_rows[prompt_len - 1 + i], tid) for i, tid in enumerate(target_ids)
    ]
    per_token = tuple(ref_f32(v) for v in per_token_unrounded)
    total_unrounded = math.fsum(per_token_unrounded)
    total = ref_f32(total_unrounded)
    mean = ref_f32(total_unrounded / len(target_ids)) if target_ids else None
    return per_token, total, mean


def assert_close(
    case: unittest.TestCase,
    got: float,
    want: float,
    *,
    rel: float = 1e-6,
    abs_tol: float = 1e-9,
    label: str = "value",
) -> None:
    """Assert close."""
    if not math.isfinite(got) or not math.isfinite(want):
        case.fail(f"{label}: non-finite comparison {got!r} vs {want!r}")
    if not math.isclose(got, want, rel_tol=rel, abs_tol=abs_tol):
        case.fail(f"{label}: {got!r} != {want!r} (rel_tol={rel})")


# ---------------------------------------------------------------------------
# Independent codec + alignment oracle (fresh implementations).
# ---------------------------------------------------------------------------


class FuzzTokenizer:
    """Greedy longest-match piece-table codec, written independently."""

    def __init__(self, pieces: dict[str, int]) -> None:
        self._pieces = dict(pieces)
        self._by_id = {i: p for p, i in pieces.items()}
        self._by_len = sorted(pieces, key=len, reverse=True)

    @property
    def vocab_size(self) -> int:
        return len(self._pieces)

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode."""
        out: list[int] = []
        pos = 0
        while pos < len(text):
            for piece in self._by_len:
                if text.startswith(piece, pos):
                    out.append(self._pieces[piece])
                    pos += len(piece)
                    break
            else:
                raise ValueError(f"unencodable@{pos}")
        return tuple(out)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        """Decode."""
        try:
            return "".join(self._by_id[i] for i in token_ids)
        except KeyError as exc:
            raise ValueError("unknown_token_id") from exc

    def parse_spans(self, text: str) -> list[tuple[int, int, int]]:
        """Greedy parse recording (token_id, char_start, char_end) spans."""
        spans: list[tuple[int, int, int]] = []
        pos = 0
        while pos < len(text):
            for piece in self._by_len:
                if text.startswith(piece, pos):
                    spans.append((self._pieces[piece], pos, pos + len(piece)))
                    pos += len(piece)
                    break
            else:
                raise ValueError(f"unencodable@{pos}")
        return spans


def oracle_alignment(
    tok: FuzzTokenizer,
    prompt_ids: tuple[int, ...],
    target_ids: tuple[int, ...],
) -> tuple[bool, str | None]:
    """Independent §9.3 predicate: (ok, reason), mirroring the documented
    contract rather than the implementation's code.
    """
    try:
        prompt_text = tok.decode(prompt_ids)
    except Exception:  # noqa: BLE001 -- oracle mirrors the sanitization boundary
        return False, "decode_failed"
    try:
        target_text = tok.decode(target_ids)
    except Exception:  # noqa: BLE001 -- oracle mirrors the sanitization boundary
        return False, "decode_failed"
    try:
        re_prompt = tok.encode(prompt_text)
    except Exception:  # noqa: BLE001 -- oracle mirrors the sanitization boundary
        return False, "encode_failed"
    if re_prompt != tuple(prompt_ids):
        return False, "prompt_roundtrip_mismatch"
    try:
        re_target = tok.encode(target_text)
    except Exception:  # noqa: BLE001 -- oracle mirrors the sanitization boundary
        return False, "encode_failed"
    if re_target != tuple(target_ids):
        return False, "target_roundtrip_mismatch"
    try:
        combined = tok.encode(prompt_text + target_text)
    except Exception:  # noqa: BLE001 -- oracle mirrors the sanitization boundary
        return False, "encode_failed"
    if combined != tuple(prompt_ids) + tuple(target_ids):
        return False, "boundary_merge_mismatch"
    return True, None


class FuzzBackend:
    """Weight-free backend: independent codec plus a scripted logit tensor."""

    fingerprint = "fuzz-scoring-backend-not-a-model"

    def __init__(
        self,
        tok: FuzzTokenizer,
        logits: torch.Tensor | None = None,
        *,
        error: BaseException | None = None,
    ) -> None:
        self._tok = tok
        self._logits = logits
        self._error = error
        self.vocab_size = tok.vocab_size
        self.forward_calls: list[tuple[int, ...]] = []

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode."""
        return self._tok.encode(text)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        """Decode."""
        return self._tok.decode(token_ids)

    def forward_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        """Forward logits."""
        self.forward_calls.append(tuple(input_ids))
        if self._error is not None:
            raise self._error
        if self._logits is None:
            raise RuntimeError("no_logits_scripted")
        return self._logits


def make_piece_table(rng: random.Random) -> dict[str, int]:
    """Random piece table over {a,b,c}; single chars always covered, at
    least one multi-char piece guaranteed (merge-sensitive boundaries).
    """
    pieces: dict[str, int] = {"a": 0, "b": 1, "c": 2}
    alphabet = ["a", "b", "c"]
    pairs = [x + y for x in alphabet for y in alphabet]
    rng.shuffle(pairs)
    for pair in pairs[: rng.randint(1, 5)]:
        pieces.setdefault(pair, len(pieces))
    if rng.random() < 0.5:
        tri = "".join(rng.choice(alphabet) for _ in range(3))
        pieces.setdefault(tri, len(pieces))
    ids = list(range(len(pieces)))
    rng.shuffle(ids)
    names = list(pieces)
    return {name: ids[k] for k, name in enumerate(names)}


def random_text(rng: random.Random, tok: FuzzTokenizer, pieces: int) -> str:
    """Random text."""
    names = list(tok._pieces)  # noqa: SLF001 -- test-local access to the table
    return "".join(rng.choice(names) for _ in range(pieces))


# ---------------------------------------------------------------------------
# score_span_from_logits: seeded differential fuzz.
# ---------------------------------------------------------------------------


class ScoreSpanDifferentialTests(unittest.TestCase):
    """Seeded differential fuzz: random logits, random prompt/target splits."""

    N_SEEDS = 60

    @staticmethod
    def _trial_tensors(seed: int) -> tuple[torch.Tensor, int, tuple[int, ...]]:
        """Trial tensors."""
        rng = random.Random(10_000 + seed)  # noqa: S311 -- deterministic test seed
        vocab = rng.choice([1, 2, 3, 5, 8, 13])
        total_len = rng.randint(2, 40)
        prompt_len = rng.randint(1, total_len)  # target may be empty
        target_ids = tuple(rng.randrange(vocab) for _ in range(total_len - prompt_len))
        scale = rng.choice([0.25, 2.0, 30.0])
        gen = torch.Generator().manual_seed(20_000 + seed)
        logits = torch.randn(total_len, vocab, generator=gen) * scale
        if rng.random() < 0.25:
            logits = logits.to(torch.float64)  # also exercise float64 input
        if rng.random() < 0.15:
            logits = torch.zeros(total_len, vocab)  # exact ties
        return logits, prompt_len, target_ids

    def test_seeded_differential_fuzz(self):
        """Test seeded differential fuzz."""
        for seed in range(self.N_SEEDS):
            logits, prompt_len, target_ids = self._trial_tensors(seed)
            with self.subTest(seed=seed):
                per_token, total, mean = score_span_from_logits(
                    logits, prompt_len=prompt_len, target_ids=target_ids
                )
                rows = logits.to(torch.float64).tolist()
                exp_per, exp_total, exp_mean = ref_score_span(rows, prompt_len, target_ids)
                self.assertEqual(len(per_token), len(target_ids))
                for i, (got, want) in enumerate(zip(per_token, exp_per, strict=True)):
                    assert_close(self, got, want, label=f"seed={seed} per_token[{i}]")
                    # Reporting policy, locked exactly: every per-token value
                    # is exactly float32-representable (a forgotten rounding
                    # would leave float64 residue and fail here).
                    self.assertEqual(got, ref_f32(got), f"seed={seed} per_token[{i}] not f32")
                self.assertEqual(total, ref_f32(total), f"seed={seed} total not f32")
                assert_close(self, total, exp_total, label=f"seed={seed} total")
                if mean is None:
                    self.assertIsNone(exp_mean)
                    self.assertEqual(target_ids, ())
                else:
                    assert exp_mean is not None
                    assert_close(self, mean, exp_mean, label=f"seed={seed} mean")

    def test_non_predicting_rows_are_masked_fuzz(self):
        """Test non predicting rows are masked fuzz."""
        # Randomizing every row that never predicts a target token (prompt
        # rows 0..prompt_len-2 and the trailing row T-1) must not move the
        # score; the predicting rows prompt_len-1..T-2 are gathered.
        for seed in range(30):
            rng = random.Random(30_000 + seed)  # noqa: S311 -- deterministic test seed
            vocab = rng.choice([2, 3, 7])
            prompt_len = rng.randint(1, 8)
            n_target = rng.randint(1, 6)
            total_len = prompt_len + n_target
            target_ids = tuple(rng.randrange(vocab) for _ in range(n_target))
            gen = torch.Generator().manual_seed(40_000 + seed)
            base = torch.randn(total_len, vocab, generator=gen) * 4.0
            masked = base.clone()
            gen2 = torch.Generator().manual_seed(40_001 + seed)
            garbage = torch.randn(total_len, vocab, generator=gen2) * 100.0
            for r in list(range(prompt_len - 1)) + [total_len - 1]:
                masked[r] = garbage[r]
            with self.subTest(seed=seed):
                self.assertEqual(
                    score_span_from_logits(base, prompt_len=prompt_len, target_ids=target_ids),
                    score_span_from_logits(masked, prompt_len=prompt_len, target_ids=target_ids),
                )

    def test_off_by_one_would_fail(self):
        """Test off by one would fail."""
        # Sensitivity control: scoring from the WRONG row (shifted by one)
        # must disagree with the reference on random logits, proving the
        # fuzz above is actually sensitive to the causal-shift index.
        gen = torch.Generator().manual_seed(7)
        logits = torch.randn(6, 4, generator=gen) * 3.0
        target_ids = (3, 1, 0)
        _, total, _ = score_span_from_logits(logits, prompt_len=3, target_ids=target_ids)
        rows = logits.tolist()
        wrong = ref_score_span(rows, 4, target_ids)  # shifted reference
        self.assertFalse(math.isclose(total, wrong[1], rel_tol=1e-6, abs_tol=1e-9))

    def test_deterministic_bit_identical(self):
        """Test deterministic bit identical."""
        gen = torch.Generator().manual_seed(99)
        logits = torch.randn(7, 6, generator=gen) * 5.0
        first = score_span_from_logits(logits, prompt_len=4, target_ids=(5, 0, 2))
        second = score_span_from_logits(logits, prompt_len=4, target_ids=(5, 0, 2))
        self.assertEqual(first, second)


class ExtremeLogitTests(unittest.TestCase):
    """Hand-derived closed forms where the naive reference would overflow."""

    def test_single_token_vocab_is_exact_zero(self):
        """Test single token vocab is exact zero."""
        per_token, total, mean = score_span_from_logits(
            torch.tensor([[123.456], [-987.0]]), prompt_len=1, target_ids=(0,)
        )
        self.assertEqual(per_token, (0.0,))
        self.assertEqual(total, 0.0)
        self.assertEqual(mean, 0.0)

    def test_uniform_row_is_minus_log_vocab(self):
        """Test uniform row is minus log vocab."""
        # Uniform row: peak-shifted exps are exactly 1, the sum is exactly V,
        # and torch's float64 log matches math.log bit-for-bit -- so the
        # float32-reported value is exactly f32(-log(V)).
        for vocab in (2, 5, 13):
            rows = [[3.25] * vocab, [-40.0] * vocab]
            per_token, total, mean = score_span_from_logits(
                torch.tensor(rows), prompt_len=1, target_ids=(vocab - 1,)
            )
            want = ref_f32(-math.log(vocab))
            self.assertEqual(per_token, (want,))
            self.assertEqual(total, want)
            self.assertEqual(mean, want)

    def test_dominant_logit_exact(self):
        """Test dominant logit exact."""
        # peak=1000: exp(0)=1, exp(-2000)=0 exactly in float64 -> exact 0/-2000.
        logits = torch.tensor([[1000.0, -1000.0, -1000.0], [0.0, 0.0, 0.0]])
        per_token, _, _ = score_span_from_logits(logits, prompt_len=1, target_ids=(0,))
        self.assertEqual(per_token, (0.0,))
        per_token, _, _ = score_span_from_logits(logits, prompt_len=1, target_ids=(1,))
        self.assertEqual(per_token, (-2000.0,))

    def test_extreme_1e30_exact(self):
        """Test extreme 1e30 exact."""
        # NOTE: torch.tensor defaults to float32, and 1e30 is not exactly
        # representable there -- the engine faithfully computes on the
        # float32-quantized inputs. Use float64 inputs for the exactness
        # check; the float32-input path is covered by the fuzz.
        logits = torch.tensor([[1e30, -1e30], [0.0, 0.0]], dtype=torch.float64)
        per_token, _, _ = score_span_from_logits(logits, prompt_len=1, target_ids=(0,))
        self.assertEqual(per_token, (0.0,))
        # exp(-2e30) underflows to 0 exactly; log(1) == 0 exactly; the only
        # rounding is the final float32 report of (-1e30) - (1e30).
        want = ref_f32((-1e30) - (1e30))
        per_token, _, _ = score_span_from_logits(logits, prompt_len=1, target_ids=(1,))
        self.assertEqual(per_token, (want,))

    def test_large_but_finite_logits_stable(self):
        """Test large but finite logits stable."""
        # |logits| ~ 100: float32 exp would overflow; the engine upcasts to
        # float64 first, so this must stay finite and near the closed form.
        logits = torch.tensor([[100.0, 0.0, -100.0], [1.0, 2.0, 3.0]])
        per_token, total, mean = score_span_from_logits(logits, prompt_len=1, target_ids=(0,))
        self.assertTrue(math.isfinite(per_token[0]))
        self.assertLessEqual(per_token[0], 0.0)
        self.assertGreater(per_token[0], -1e-15)  # ~ -exp(-100)
        self.assertEqual(total, per_token[0])
        self.assertEqual(mean, per_token[0])

    def test_mixed_sign_large_gap(self):
        """Test mixed sign large gap."""
        logits = torch.tensor([[-500.0, 500.0], [0.0, 0.0]])
        per_token, _, _ = score_span_from_logits(logits, prompt_len=1, target_ids=(1,))
        self.assertEqual(per_token, (0.0,))
        per_token, _, _ = score_span_from_logits(logits, prompt_len=1, target_ids=(0,))
        self.assertEqual(per_token, (-1000.0,))


class AccumulationPolicyTests(unittest.TestCase):
    """The total must accumulate float64 *unrounded* per-token values.

    Part 1 separates float64 from float32 accumulation: with 20k random
    rows, float32 accumulation errs by ~0.5 while float64 errs by ~1e-9, so
    a 1e-4 absolute tolerance through the float32 reporting rounding tells
    them apart. Part 2 separates unrounded from rounded-first accumulation:
    identical rows make the per-term float32 rounding error *systematic*,
    so 200k terms amplify it past the final reporting ulp (a rounded-first
    policy would miss by ~1e-2).
    """

    def test_total_accumulates_in_float64(self):
        """Test total accumulates in float64."""
        n = 20_000
        gen = torch.Generator().manual_seed(1234)
        logits = torch.randn(n + 1, 8, generator=gen) * 2.0
        target_ids = tuple(int(v) for v in torch.randint(0, 8, (n,), generator=gen).tolist())
        _, total, mean = score_span_from_logits(logits, prompt_len=1, target_ids=target_ids)
        rows = logits.to(torch.float64).tolist()
        unrounded = [ref_logprob_row(rows[i], t) for i, t in enumerate(target_ids)]
        want_total = ref_f32(math.fsum(unrounded))
        # A float32 accumulator would be off by ~0.5 here; float64 by ~1e-9.
        self.assertLess(abs(total - want_total), 1e-4)
        want_mean = ref_f32(math.fsum(unrounded) / n)
        self.assertIsNotNone(mean)
        assert mean is not None
        assert_close(self, mean, want_mean, rel=1e-9, abs_tol=1e-12, label="mean")

    def test_total_accumulates_unrounded_values(self):
        """Test total accumulates unrounded values."""
        # The per-term float32 rounding error is systematic for identical
        # rows, so a large enough n amplifies it past the final reporting
        # ulp -- but only for favorable bit patterns. Search a fixed
        # candidate list for a construction where f32(n*v) and
        # f32(n*f32(v)) genuinely differ; the search is deterministic and
        # fails loudly if no candidate separates the policies.
        n = 50_000
        candidates = [
            ([0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 2.0, -2.0], 0),
            ([1.0, 0.0, -1.0, 0.5, -0.5, 0.125, -0.125, 0.0], 2),
            ([3.0, -3.0, 1.5, -1.5, 0.75, -0.75, 0.0, 0.0], 4),
            ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 7),
            ([-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 0.5, -1.5], 1),
            ([4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0], 5),
        ]
        found = None
        for row, tid in candidates:
            v = ref_logprob_row(row, tid)
            # v safely mid-bin: engine/reference float64 disagreement
            # (~1e-16) cannot masquerade as a policy difference...
            if ref_f32(v * (1 + 1e-9)) != ref_f32(v * (1 - 1e-9)):
                continue
            total_unrounded = math.fsum([v] * n)
            # ...and the total safely mid-bin too.
            if ref_f32(total_unrounded * (1 + 1e-9)) != ref_f32(total_unrounded * (1 - 1e-9)):
                continue
            if ref_f32(total_unrounded) != ref_f32(n * ref_f32(v)):
                found = (row, tid, v, total_unrounded)
                break
        self.assertIsNotNone(found, "no separating construction found")
        assert found is not None
        row, tid, v, total_unrounded = found
        logits = torch.tensor([row] * n + [[0.0] * 8])
        per_token, total, _ = score_span_from_logits(logits, prompt_len=1, target_ids=(tid,) * n)
        self.assertEqual(len(per_token), n)
        # A rounded-first accumulator would report f32(n*f32(v)); the
        # engine must report f32 of the unrounded sum, exactly.
        self.assertEqual(total, ref_f32(total_unrounded))
        self.assertNotEqual(total, ref_f32(n * ref_f32(v)))


# ---------------------------------------------------------------------------
# check_alignment: differential verification against the oracle.
# ---------------------------------------------------------------------------


class AlignmentDifferentialTests(unittest.TestCase):
    """Alignment differential tests."""

    N_TABLES = 12

    def _tables(self):
        """Tables."""
        for s in range(self.N_TABLES):
            rng = random.Random(50_000 + s)  # noqa: S311 -- deterministic test seed
            pieces = make_piece_table(rng)
            yield s, FuzzTokenizer(pieces), rng

    def _assert_matches_oracle(self, tok, backend, prompt_ids, target_ids):
        """Assert matches oracle."""
        check = check_alignment(backend, prompt_ids, target_ids)
        want_ok, want_reason = oracle_alignment(tok, prompt_ids, target_ids)
        self.assertEqual(check.ok, want_ok, f"ok mismatch for {prompt_ids} / {target_ids}")
        self.assertEqual(
            check.reason, want_reason, f"reason mismatch for {prompt_ids} / {target_ids}"
        )
        return check

    def test_genuine_splits_always_verify(self):
        """Test genuine splits always verify."""
        # Liveness theorem for greedy codecs: any split of a genuine
        # encoding at a token boundary MUST verify (per-span roundtrips and
        # the concatenation property hold by construction).
        for s, tok, rng in self._tables():
            backend = FuzzBackend(tok)
            for trial in range(25):
                text = random_text(rng, tok, rng.randint(1, 10))
                ids = tok.encode(text)
                cut = rng.randint(0, len(ids))  # incl. empty spans
                prompt_ids, target_ids = ids[:cut], ids[cut:]
                with self.subTest(table=s, trial=trial):
                    check = self._assert_matches_oracle(tok, backend, prompt_ids, target_ids)
                    self.assertTrue(check.ok)
                    self.assertIsNone(check.reason)
                    # Safety direction, verified directly: an ok verdict
                    # means the concatenated text re-encodes to the
                    # concatenated ids -- the invariant scoring relies on.
                    self.assertEqual(
                        tok.encode(check.prompt_text + check.target_text),
                        tuple(prompt_ids) + tuple(target_ids),
                    )

    def test_adversarial_sequences_match_oracle(self):
        """Test adversarial sequences match oracle."""
        # Arbitrary id tuples (unknown ids, unencodable decodes, broken
        # roundtrips, boundary merges): implementation and oracle must agree
        # exactly, on both the verdict and the fixed reason code.
        for s, tok, rng in self._tables():
            backend = FuzzBackend(tok)
            vocab = tok.vocab_size
            for trial in range(30):
                n, m = rng.randint(0, 5), rng.randint(0, 5)
                prompt_ids = tuple(rng.randrange(vocab + 2) for _ in range(n))
                target_ids = tuple(rng.randrange(vocab + 2) for _ in range(m))
                with self.subTest(table=s, trial=trial):
                    check = self._assert_matches_oracle(tok, backend, prompt_ids, target_ids)
                    if check.ok:
                        self.assertEqual(
                            tok.encode(check.prompt_text + check.target_text),
                            tuple(prompt_ids) + tuple(target_ids),
                        )

    def test_straddling_token_must_refuse(self):
        """Test straddling token must refuse."""
        # Construct splits where a vocabulary piece's characters straddle
        # the prompt/answer boundary; whenever the re-parse uses a piece
        # strictly containing the character boundary, refusal is mandatory.
        # The oracle arbitrates the verdict; the span check asserts the
        # safety direction independently.
        for s, tok, rng in self._tables():
            multi = [p for p in tok._pieces if len(p) >= 2]  # noqa: SLF001
            self.assertTrue(multi)
            backend = FuzzBackend(tok)
            for trial in range(12):
                piece = rng.choice(multi)
                cut = rng.randint(1, len(piece) - 1)
                left = random_text(rng, tok, rng.randint(0, 3))
                right = random_text(rng, tok, rng.randint(0, 3))
                prompt_text, target_text = left + piece[:cut], piece[cut:] + right
                prompt_ids, target_ids = tok.encode(prompt_text), tok.encode(target_text)
                with self.subTest(table=s, trial=trial):
                    check = self._assert_matches_oracle(tok, backend, prompt_ids, target_ids)
                    spans = tok.parse_spans(prompt_text + target_text)
                    boundary = len(prompt_text)
                    straddled = any(a < boundary < b for _, a, b in spans)
                    if straddled:
                        self.assertFalse(check.ok)
                        self.assertEqual(check.reason, "boundary_merge_mismatch")

    def test_codec_failures_become_fixed_reasons(self):
        """Test codec failures become fixed reasons."""
        tok = FuzzTokenizer({"a": 0, "b": 1})
        backend = FuzzBackend(tok)
        # Unknown id in prompt / target -> decode_failed.
        check = check_alignment(backend, (9,), (0,))
        self.assertEqual((check.ok, check.reason), (False, "decode_failed"))
        self.assertEqual((check.prompt_text, check.target_text), ("", ""))
        check = check_alignment(backend, (0,), (9,))
        self.assertEqual((check.ok, check.reason), (False, "decode_failed"))
        self.assertEqual((check.prompt_text, check.target_text), ("a", ""))

        # decode() returning a non-str -> decode_failed, never an exception.
        class BadDecode(FuzzBackend):
            """Bad decode."""

            @staticmethod
            def decode(token_ids: tuple[int, ...]) -> str:
                """Decode."""
                # type: ignore[override]
                return ["not", "a", "str"]  # type: ignore[return-value]

        check = check_alignment(BadDecode(tok), (0,), (1,))
        self.assertEqual((check.ok, check.reason), (False, "decode_failed"))

        # decode() raising -> decode_failed.
        class RaisingDecode(FuzzBackend):
            """Raising decode."""

            @staticmethod
            def decode(token_ids: tuple[int, ...]) -> str:
                """Decode."""
                raise RuntimeError("tokenizer exploded")

        check = check_alignment(RaisingDecode(tok), (0,), (1,))
        self.assertEqual((check.ok, check.reason), (False, "decode_failed"))
        # A codec whose decode emits text it cannot re-encode -> encode_failed.
        holey = FuzzTokenizer({"a": 0})
        holey._by_id[7] = "z"  # noqa: SLF001 -- decode-only id, unencodable
        check = check_alignment(FuzzBackend(holey), (7,), (0,))
        self.assertEqual((check.ok, check.reason), (False, "encode_failed"))

        # encode() returning a non-tuple -> encode_failed.
        class BadEncode(FuzzBackend):
            """Bad encode."""

            @staticmethod
            def encode(text: str) -> tuple[int, ...]:
                """Encode."""
                # type: ignore[override]
                return [0]  # type: ignore[return-value]

        check = check_alignment(BadEncode(tok), (0,), (1,))
        self.assertEqual((check.ok, check.reason), (False, "encode_failed"))
        # Lossy per-span roundtrips -> the specific mismatch codes.
        tricky = FuzzTokenizer({"ab": 0, "a": 1, "b": 2})
        tricky_backend = FuzzBackend(tricky)
        check = check_alignment(tricky_backend, (1, 2), (0,))
        self.assertEqual((check.ok, check.reason), (False, "prompt_roundtrip_mismatch"))
        check = check_alignment(tricky_backend, (0,), (1, 2))
        self.assertEqual((check.ok, check.reason), (False, "target_roundtrip_mismatch"))
        # Empty spans verify ("" roundtrips through the empty tuple).
        check = check_alignment(backend, (), ())
        self.assertEqual((check.ok, check.reason), (True, None))


# ---------------------------------------------------------------------------
# score_target_logprobs: end-to-end differential verification.
# ---------------------------------------------------------------------------


class EndToEndDifferentialTests(unittest.TestCase):
    """End to end differential tests."""

    N_SEEDS = 40

    @staticmethod
    def _trial(seed: int):
        """Trial."""
        rng = random.Random(60_000 + seed)  # noqa: S311 -- deterministic test seed
        tok = FuzzTokenizer(make_piece_table(rng))
        # Genuine split of one encoding at a token boundary: alignment is
        # guaranteed to pass (liveness theorem), isolating the math layer.
        for _ in range(100):
            full_ids = tok.encode(random_text(rng, tok, rng.randint(2, 11)))
            if len(full_ids) >= 2:
                break
        else:
            raise AssertionError("could not build a multi-token encoding")
        cut = rng.randint(1, len(full_ids) - 1)
        prompt_ids, target_ids = full_ids[:cut], full_ids[cut:]
        target_text = tok.decode(target_ids)
        vocab = tok.vocab_size
        gen = torch.Generator().manual_seed(70_000 + seed)
        scale = rng.choice([0.5, 3.0, 20.0])
        logits = torch.randn(len(full_ids), vocab, generator=gen) * scale
        return tok, prompt_ids, target_ids, logits, target_text

    def test_end_to_end_matches_reference(self):
        """Test end to end matches reference."""
        for seed in range(self.N_SEEDS):
            tok, prompt_ids, target_ids, logits, target_text = self._trial(seed)
            backend = FuzzBackend(tok, logits)
            with self.subTest(seed=seed):
                score = score_target_logprobs(backend, prompt_ids, target_ids)
                rows = logits.to(torch.float64).tolist()
                exp_per, exp_total, exp_mean = ref_score_span(rows, len(prompt_ids), target_ids)
                self.assertEqual(score.status, STATUS_OK)
                self.assertIsNone(score.reason)
                self.assertEqual(len(score.per_token_logprobs), len(target_ids))
                for i, (got, want) in enumerate(
                    zip(score.per_token_logprobs, exp_per, strict=True)
                ):
                    assert_close(self, got, want, label=f"seed={seed} per_token[{i}]")
                assert score.total_logprob is not None
                assert_close(self, score.total_logprob, exp_total, label=f"seed={seed} total")
                assert score.mean_logprob is not None
                assert exp_mean is not None
                assert_close(self, score.mean_logprob, exp_mean, label=f"seed={seed} mean")
                self.assertEqual(score.scored_token_count, len(target_ids))
                self.assertEqual(score.prompt_token_count, len(prompt_ids))
                self.assertEqual(score.target_byte_length, len(target_text.encode("utf-8")))
                # Exactly one forward pass over prompt_ids + target_ids.
                self.assertEqual(backend.forward_calls, [tuple(prompt_ids) + tuple(target_ids)])

    def test_empty_target_is_defined(self):
        """Test empty target is defined."""
        tok = FuzzTokenizer(make_piece_table(random.Random(1)))  # noqa: S311 -- deterministic test seed
        backend = FuzzBackend(tok, torch.zeros(2, tok.vocab_size))
        score = score_target_logprobs(backend, (0, 1), ())
        self.assertEqual(score.status, STATUS_OK_EMPTY)
        self.assertIsNone(score.reason)
        self.assertEqual(score.per_token_logprobs, ())
        self.assertEqual(score.total_logprob, 0.0)
        self.assertIsNone(score.mean_logprob)
        self.assertEqual(score.scored_token_count, 0)
        self.assertEqual(score.target_byte_length, 0)
        self.assertEqual(score.prompt_token_count, 2)
        self.assertEqual(backend.forward_calls, [])

    def test_single_token_target_at_sequence_start(self):
        """Test single token target at sequence start."""
        # prompt_len=1: the single target token is predicted by row 0, the
        # sharpest off-by-one probe (row -1 would be out of bounds).
        tok = FuzzTokenizer({"a": 0, "b": 1, "c": 2})
        gen = torch.Generator().manual_seed(5)
        logits = torch.randn(2, 3, generator=gen)
        backend = FuzzBackend(tok, logits)
        score = score_target_logprobs(backend, (0,), (2,))
        exp_per, exp_total, exp_mean = ref_score_span(logits.to(torch.float64).tolist(), 1, (2,))
        self.assertEqual(score.status, STATUS_OK)
        assert_close(self, score.per_token_logprobs[0], exp_per[0], label="per_token")
        assert score.total_logprob is not None and exp_mean is not None
        assert_close(self, score.total_logprob, exp_total, label="total")
        assert score.mean_logprob is not None
        assert_close(self, score.mean_logprob, exp_mean, label="mean")

    def test_very_long_prompt(self):
        """Test very long prompt."""
        tok = FuzzTokenizer({"a": 0, "b": 1})
        prompt_ids = tuple([0, 1] * 1000)  # 2000 prompt tokens
        target_ids = (0, 1, 0)
        gen = torch.Generator().manual_seed(11)
        logits = torch.randn(len(prompt_ids) + len(target_ids), 2, generator=gen)
        backend = FuzzBackend(tok, logits)
        score = score_target_logprobs(backend, prompt_ids, target_ids)
        exp_per, exp_total, exp_mean = ref_score_span(
            logits.to(torch.float64).tolist(), len(prompt_ids), target_ids
        )
        self.assertEqual(score.status, STATUS_OK)
        self.assertEqual(score.prompt_token_count, 2000)
        self.assertEqual(score.scored_token_count, 3)
        for got, want in zip(score.per_token_logprobs, exp_per, strict=True):
            assert_close(self, got, want, label="per_token")
        assert score.total_logprob is not None and exp_mean is not None
        assert_close(self, score.total_logprob, exp_total, label="total")
        assert score.mean_logprob is not None
        assert_close(self, score.mean_logprob, exp_mean, label="mean")

    def test_single_token_vocab_end_to_end(self):
        """Test single token vocab end to end."""
        tok = FuzzTokenizer({"a": 0})
        logits = torch.tensor([[4.0], [-2.0], [0.5]])
        backend = FuzzBackend(tok, logits)
        score = score_target_logprobs(backend, (0,), (0, 0))
        self.assertEqual(score.status, STATUS_OK)
        self.assertEqual(score.per_token_logprobs, (0.0, 0.0))
        self.assertEqual(score.total_logprob, 0.0)
        self.assertEqual(score.mean_logprob, 0.0)

    def test_refusal_paths_make_no_forward_call(self):
        """Test refusal paths make no forward call."""
        tok = FuzzTokenizer({"a": 0, "b": 1, "ab": 2})
        backend = FuzzBackend(tok, torch.zeros(3, 3))
        # Boundary straddle: "a"(0) prompt, "b"(1) target, "ab" re-merges.
        score = score_target_logprobs(backend, (0,), (1,))
        self.assertEqual(score.status, STATUS_UNAVAILABLE)
        self.assertEqual(score.reason, "boundary_merge_mismatch")
        self.assertEqual(backend.forward_calls, [])
        # Empty prompt.
        score = score_target_logprobs(backend, (), (0,))
        self.assertEqual(score.reason, "empty_prompt")
        self.assertEqual(backend.forward_calls, [])
        # Out-of-range id caught before the forward pass.
        score = score_target_logprobs(backend, (0,), (7,))
        self.assertEqual(score.reason, "token_id_out_of_range")
        self.assertEqual(backend.forward_calls, [])

    def test_provider_math_rejections_become_fixed_reasons(self):
        """Test provider math rejections become fixed reasons."""
        tok = FuzzTokenizer({"a": 0, "b": 1})
        # Wrong row count -> logits_length_mismatch.
        backend = FuzzBackend(tok, torch.zeros(5, 2))
        score = score_target_logprobs(backend, (0,), (1,))
        self.assertEqual(score.status, STATUS_UNAVAILABLE)
        self.assertEqual(score.reason, "logits_length_mismatch")
        # NaN logits -> non_finite_logits.
        bad = torch.zeros(2, 2)
        bad[0, 0] = float("nan")
        score = score_target_logprobs(FuzzBackend(tok, bad), (0,), (1,))
        self.assertEqual(score.reason, "non_finite_logits")
        # Whitelisted provider code passes through verbatim.
        backend = FuzzBackend(tok, error=ValueError("context_limit_exceeded"))
        score = score_target_logprobs(backend, (0,), (1,))
        self.assertEqual(score.reason, "context_limit_exceeded")
        # Non-whitelisted provider text collapses to forward_failed.
        backend = FuzzBackend(tok, error=RuntimeError("disk on fire"))
        score = score_target_logprobs(backend, (0,), (1,))
        self.assertEqual(score.reason, "forward_failed")
        # Non-string provider arg collapses too.
        backend = FuzzBackend(tok, error=RuntimeError(123))
        score = score_target_logprobs(backend, (0,), (1,))
        self.assertEqual(score.reason, "forward_failed")

    def test_end_to_end_deterministic(self):
        """Test end to end deterministic."""
        tok, prompt_ids, target_ids, logits, _ = self._trial(3)
        backend = FuzzBackend(tok, logits)
        first = score_target_logprobs(backend, prompt_ids, target_ids)
        second = score_target_logprobs(backend, prompt_ids, target_ids)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
