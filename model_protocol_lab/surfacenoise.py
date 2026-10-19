"""Surface-noise self-consistency probe.

Deterministic instrument measuring whether a prompt's task success
survives surface-form noise (typos, case flips, whitespace variation,
Unicode quirks, punctuation stripping).

What it is: for a user-supplied prompt text P, task fixtures F, and a
frozen backend B, the probe computes the baseline pass/fail vector
V(P, F, B), then re-runs V over a frozen perturbation-operator plan
applied to P's text span only. Per operator it reports stability (share
of effective variants whose pass/fail vector is unchanged) and the
nearest observed flip (smallest token indel-edit rate among flipping
variants). Fixture message content is never modified.

What it is not (see docs/SURFACE_NOISE_PROBE_DESIGN.md and the
ADAPTER_RULE below): a prompt-quality certificate, a guarantee against
real-world typos (the reported distance is an *upper bound* on the true
minimal-flip distance and errs toward false confidence), or a measure of
semantic, adversarial, or injection robustness.

No model training, no LLM calls beyond the backend under test, no I/O,
no global state. All randomness is seeded and published in the frozen
protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
import threading
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any, Literal

from .core import Backend, Fixture, Generation, Message, first_boundary
from .perturbation import PERTURBATION_OPERATORS, perturb_text

PROBE_NAME = "surface-noise-probe"
PROBE_VERSION = "1.0.0"

# Frozen search plan: seeded operators get exactly these seeds (a sliver
# of the 2**64 seed space -- reported honestly as heuristic coverage);
# whitespace's three modes and punctuation_strip's single outcome are
# enumerated exactly.
SEARCH_PLAN_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
WHITESPACE_MODES: tuple[str, ...] = ("collapse", "double", "newline-split")

# Sweep bounds (documented rationale with each use site).
MAX_PROMPT_CHARS = 1024 * 1024  # 1 MiB: memory-DoS prevention; the sweep
# multiplies input size across variants, so the cap is on the subject.
MAX_FIXTURES = 50  # bounds the variant x fixture forward-pass fan-out.
MAX_FIXTURE_MESSAGE_CHARS = 256 * 1024  # 256 KiB per message: the sweep
# re-encodes every fixture's messages for every variant, so unbounded
# content is a memory-DoS vector (Finding 5).

RENDER_PREFIX = "Instruction: "

Verdict = Literal["pass", "fail", "abstain"]

ADAPTER_RULE = (
    "This probe measures stability of task success under a frozen set of "
    "surface-form text perturbations. A high stability number means the "
    "prompt's pass/fail pattern survived those perturbations on those "
    "fixtures with that backend -- nothing more. It is not a certificate "
    "of prompt quality, not a guarantee against real-world typos (the "
    "searched seeds are a sliver of the space; the reported distance is "
    "an upper bound), and not a measurement of semantic, adversarial, or "
    "injection robustness. Controls validate discrimination within the "
    "operator set's theory of fragility, not against real-world prompt "
    "failure."
)

# Unicode bidi control set (cycle 114): the exact set the lab's
# sanitizers strip. The lint check reports their presence; it does not
# strip them.
_BIDI_CONTROLS = frozenset(
    ["\u200e", "\u200f", "\u2028", "\u2029"]
    + [chr(c) for c in range(0x202A, 0x202F)]
    + [chr(c) for c in range(0x2066, 0x206A)]
)


def _check_prompt(prompt_text: Any) -> str:
    if type(prompt_text) is not str:
        raise ValueError("prompt_text must be a string")
    if len(prompt_text) > MAX_PROMPT_CHARS:
        raise ValueError(f"prompt_text exceeds {MAX_PROMPT_CHARS} characters")
    return prompt_text


def _check_fixtures(fixtures: Any) -> tuple[Fixture, ...]:
    if type(fixtures) is not tuple or not fixtures:
        raise ValueError("fixtures must be a nonempty tuple of Fixture")
    if any(type(case) is not Fixture for case in fixtures):
        raise ValueError("fixtures must be a nonempty tuple of Fixture")
    if len(fixtures) > MAX_FIXTURES:
        raise ValueError(f"at most {MAX_FIXTURES} fixtures per probe run")
    if len({case.id for case in fixtures}) != len(fixtures):
        raise ValueError("fixture ids must be unique")
    for case in fixtures:
        for message in case.messages:
            if len(message.content) > MAX_FIXTURE_MESSAGE_CHARS:
                raise ValueError(
                    f"fixture {case.id!r} message exceeds {MAX_FIXTURE_MESSAGE_CHARS} characters"
                )
    return fixtures


def _check_boundary(boundary: Any) -> tuple[int, ...]:
    if (
        type(boundary) is not tuple
        or not boundary
        or any(type(t) is not int or t < 0 for t in boundary)
    ):
        raise ValueError("boundary must be a nonempty tuple of token ids")
    return boundary


def _check_max_new_tokens(value: Any) -> int:
    if type(value) is not int or isinstance(value, bool) or not 1 <= value <= 4096:
        raise ValueError("max_new_tokens must be an integer between 1 and 4096")
    return value


def _check_seed(seed: Any) -> int:
    if type(seed) is not int or isinstance(seed, bool) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    return seed


def _valid_ids(ids: Any) -> bool:
    return type(ids) is tuple and all(type(t) is int and t >= 0 for t in ids)


def surfacenoise_code_hash() -> str:
    """sha256 of surfacenoise.py (the scorer: flip predicate, stability formula)."""
    path = Path(__file__).with_name("surfacenoise.py")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def perturbation_code_hash() -> str:
    """sha256 of the frozen operator source: the protocol's code pin."""
    path = Path(__file__).with_name("perturbation.py")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_hash(fixtures: tuple[Fixture, ...]) -> str:
    payload = [
        {
            "id": case.id,
            "messages": [
                {"role": message.role, "content": message.content} for message in case.messages
            ],
            "expected": case.expected,
            "group": case.group,
        }
        for case in fixtures
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def frozen_search_plan() -> dict[str, dict[str, Any]]:
    """The frozen per-operator search plan (single source of truth)."""
    return {
        "typo": {"seeds": list(SEARCH_PLAN_SEEDS), "exact": False},
        "case_flip": {"seeds": list(SEARCH_PLAN_SEEDS), "exact": False},
        "unicode_variant": {"seeds": list(SEARCH_PLAN_SEEDS), "exact": False},
        "whitespace": {"modes": list(WHITESPACE_MODES), "exact": True},
        "punctuation_strip": {"outcomes": 1, "exact": True},
    }


def freeze_surface_noise_protocol(
    *, backend_fingerprint: str, fixtures: tuple[Fixture, ...]
) -> dict[str, Any]:
    """Publish the frozen measurement protocol *before* any scoring.

    The returned dict is the complete, self-describing contract: operator
    set, code hash, search plan verbatim, prompt/task split, rendering,
    flip predicate, denominator definition, backend fingerprint, and
    fixture hashes. The probe refuses to run against a tampered or
    mismatched protocol.
    """
    if type(backend_fingerprint) is not str or not backend_fingerprint:
        raise ValueError("backend_fingerprint must be a nonempty string")
    fixtures = _check_fixtures(fixtures)
    return {
        "instrument": PROBE_NAME,
        "protocol_version": PROBE_VERSION,
        "perturbation_operators": list(PERTURBATION_OPERATORS),
        "perturbation_code_sha256": perturbation_code_hash(),
        "search_plan": frozen_search_plan(),
        "prompt_task_split": (
            "Perturbation applies to the prompt text span only; fixture "
            "message content is never modified."
        ),
        "rendering": (
            'Each evaluation renders "Instruction: <P>\\n\\n" followed by '
            "the fixture's messages verbatim as '<role>: <content>' lines."
        ),
        "flip_predicate": (
            "A variant flips when its pass/fail vector differs from the "
            "baseline vector on any mutually-evaluable fixture (evaluable "
            "in both runs); a change in either direction counts."
        ),
        "denominator": (
            "n_effective counts variants whose perturbed prompt span "
            "changed the token-id sequence (via backend.encode); no-op "
            "variants are listed, never counted as stable. n_unevaluable "
            "counts variants with zero mutually-evaluable fixtures; they are "
            "excluded from the stability denominator, never counted as unchanged."
        ),
        "distance": (
            "Token indel-edit rate: Myers O(ND) insertion/deletion distance "
            "between the baseline and variant prompt-span token-id "
            "sequences, divided by their combined length; a substitution "
            "counts as delete + insert. Reported per operator only."
        ),
        "backend_fingerprint": backend_fingerprint,
        "fixture_ids": [case.id for case in fixtures],
        "fixture_sha256": _fixture_hash(fixtures),
        "surfacenoise_code_sha256": surfacenoise_code_hash(),
        "frozen_at": datetime.now(UTC).isoformat(),
    }


def verify_protocol(
    protocol: Any,
    *,
    backend_fingerprint: str,
    fixtures: tuple[Fixture, ...] | None = None,
) -> dict[str, Any]:
    """Fail-closed protocol check; returns the protocol on success.

    When ``fixtures`` is provided, their hash must match the protocol's
    ``fixture_sha256`` — the probe refuses to score a fixture set different
    from the one the protocol was frozen for (cross-run comparability binding).
    """
    if type(protocol) is not dict:
        raise ValueError("protocol must be the dict from freeze_surface_noise_protocol")
    if type(backend_fingerprint) is not str or not backend_fingerprint:
        raise ValueError("backend_fingerprint must be a nonempty string")
    if protocol.get("instrument") != PROBE_NAME:
        raise ValueError("protocol is not a surface-noise-probe protocol")
    if protocol.get("protocol_version") != PROBE_VERSION:
        raise ValueError(
            f"protocol version {protocol.get('protocol_version')!r} != {PROBE_VERSION!r}"
        )
    if protocol.get("perturbation_code_sha256") != perturbation_code_hash():
        raise ValueError("perturbation.py changed since the protocol was frozen")
    if protocol.get("surfacenoise_code_sha256") != surfacenoise_code_hash():
        raise ValueError("surfacenoise.py changed since the protocol was frozen")
    if protocol.get("search_plan") != frozen_search_plan():
        raise ValueError("search plan differs from the frozen protocol")
    if protocol.get("backend_fingerprint") != backend_fingerprint:
        raise ValueError("backend fingerprint differs from the frozen protocol")
    if fixtures is not None and _fixture_hash(fixtures) != protocol.get("fixture_sha256"):
        raise ValueError("fixtures differ from the frozen protocol's fixture set")
    return protocol


def render_probe_input(prompt_text: str, fixture: Fixture) -> str:
    """Frozen rendering: instruction span, then fixture messages verbatim."""
    lines = [f"{RENDER_PREFIX}{prompt_text}", ""]
    for message in fixture.messages:
        lines.append(f"{message.role}: {message.content}")
    return "\n".join(lines)


def _whitespace_mode_pairs(text: str) -> tuple[tuple[str, str], ...]:
    """(name, text) per distinct whitespace mode, order-stable.

    Single source of truth for mode/text alignment: deduplication by text
    keeps the FIRST name for each distinct outcome, so names never
    misalign with texts after dedup.
    """
    if type(text) is not str:
        raise ValueError("text must be a string")
    if not any(ch.isspace() for ch in text):
        return ()
    candidates = (
        ("collapse", " ".join(text.split())),
        ("double", text.replace(" ", "  ")),
        ("newline-split", text.replace(" ", "\n")),
    )
    seen: list[tuple[str, str]] = []
    seen_texts: set[str] = set()
    for name, mode_text in candidates:
        if mode_text not in seen_texts:
            seen_texts.add(mode_text)
            seen.append((name, mode_text))
    return tuple(seen)


def whitespace_modes(text: str) -> tuple[str, ...]:
    """The three documented whitespace modes, deduplicated, order-stable.

    Mirrors perturbation.py's ``_perturb_whitespace`` modes (collapse /
    double / newline-split); returns ``()`` for text without whitespace.
    No-op filtering is the probe's job (token-level), not this function's.
    """
    return tuple(mode_text for _, mode_text in _whitespace_mode_pairs(text))


def token_edit_distance(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    """Indel edit distance via Myers' O(ND) greedy algorithm.

    Counts insertions + deletions; a substitution costs two operations.
    Exact and deterministic. Common prefix/suffix trimming keeps the
    quadratic core small for near-identical sequences (which perturbed
    prompts are). Raises ValueError on non-tuple input.
    """
    if type(a) is not tuple or type(b) is not tuple:
        raise ValueError("token sequences must be tuples")
    # Trim the common prefix and suffix: edits live in the middle.
    start = 0
    while start < len(a) and start < len(b) and a[start] == b[start]:
        start += 1
    end_a, end_b = len(a), len(b)
    while end_a > start and end_b > start and a[end_a - 1] == b[end_b - 1]:
        end_a -= 1
        end_b -= 1
    a, b = a[start:end_a], b[start:end_b]
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    size = n + m
    # v[k] = furthest x reached on diagonal k; diagonals in [-d, d].
    v = [0] * (2 * size + 1)
    off = size
    for d in range(size + 1):
        for k in range(-d, d + 1, 2):
            kk = off + k
            # insertion (down) when hugging the lower edge or the
            # insertion path reached further; otherwise deletion (right).
            x = v[kk + 1] if k == -d or (k != d and v[kk - 1] < v[kk + 1]) else v[kk - 1] + 1
            y = x - k
            while x < n and y < m and a[x] == b[y]:
                x += 1
                y += 1
            v[kk] = x
            if x >= n and y >= m:
                return d
    raise AssertionError(  # pragma: no cover -- Myers distance always terminates
        "unreachable: Myers distance always terminates"
    )


def token_edit_rate(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """Normalized distance in [0, 1]: indel distance over combined length."""
    distance = token_edit_distance(a, b)
    denom = len(a) + len(b)
    return 0.0 if denom == 0 else distance / denom


def _request_seed(seed: int, fixture_id: str) -> int:
    material = f"{seed}\0{fixture_id}".encode("utf-8", errors="surrogatepass")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _evaluate_one(
    prompt_text: str,
    case: Fixture,
    backend: Backend,
    boundary: tuple[int, ...],
    max_new_tokens: int,
    seed: int,
    event: threading.Event,
) -> Verdict:
    """Score one fixture: pass / fail / abstain (fail-closed)."""
    try:
        input_ids = backend.encode(render_probe_input(prompt_text, case))
    except Exception:  # noqa: BLE001 -- backend boundary; abstain, never crash
        return "abstain"
    if not _valid_ids(input_ids):
        return "abstain"
    try:
        gen = backend.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            seed=_request_seed(seed, case.id),
            cancel=event,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:  # noqa: BLE001 -- backend boundary; abstain, never crash
        return "abstain"
    if gen.reason == "cancelled" or event.is_set() or gen.reason == "error":
        return "abstain"
    if not _valid_ids(gen.token_ids):
        return "abstain"
    cut = first_boundary(gen.token_ids, boundary)
    answer_ids = gen.token_ids if cut is None else gen.token_ids[:cut]
    try:
        answer = backend.decode(answer_ids)
    except Exception:  # noqa: BLE001 -- backend boundary; abstain, never crash
        return "abstain"
    if type(answer) is not str:
        return "abstain"
    return "pass" if answer == case.expected else "fail"


def evaluate_prompt(
    prompt_text: str,
    fixtures: tuple[Fixture, ...],
    backend: Backend,
    *,
    boundary: tuple[int, ...],
    max_new_tokens: int = 64,
    seed: int = 0,
    cancel: threading.Event | None = None,
) -> tuple[Verdict, ...]:
    """V(P, F, B): the pass/fail/abstain vector for one prompt.

    Pure with respect to (prompt, fixtures, backend, boundary, seed):
    identical inputs always produce identical vectors (greedy backend
    assumed, as the Backend contract requires).
    """
    prompt_text = _check_prompt(prompt_text)
    fixtures = _check_fixtures(fixtures)
    boundary = _check_boundary(boundary)
    max_new_tokens = _check_max_new_tokens(max_new_tokens)
    seed = _check_seed(seed)
    if cancel is not None and not isinstance(cancel, threading.Event):
        raise ValueError("cancel must be a threading.Event")
    event = cancel if cancel is not None else threading.Event()
    return tuple(
        "abstain"
        if event.is_set()
        else _evaluate_one(prompt_text, case, backend, boundary, max_new_tokens, seed, event)
        for case in fixtures
    )


def _enumerate_variants(prompt_text: str) -> list[tuple[str, str, dict[str, Any], str]]:
    """All (variant_id, operator, detail, perturbed_text) in frozen plan order."""
    variants: list[tuple[str, str, dict[str, Any], str]] = []
    for operator in ("typo", "case_flip", "unicode_variant"):
        for s in SEARCH_PLAN_SEEDS:
            variants.append(
                (
                    f"{operator}/seed={s}",
                    operator,
                    {"seed": s},
                    perturb_text(prompt_text, operator, seed=s),
                )
            )
    for i, (mode_name, mode_text) in enumerate(_whitespace_mode_pairs(prompt_text)):
        variants.append(
            (
                f"whitespace/mode={mode_name}",
                "whitespace",
                {"mode": mode_name, "mode_index": i},
                mode_text,
            )
        )
    variants.append(
        (
            "punctuation_strip/single",
            "punctuation_strip",
            {"outcome": "single"},
            perturb_text(prompt_text, "punctuation_strip", seed=0),
        )
    )
    return variants


def _operator_exact(operator: str) -> bool:
    return operator in ("whitespace", "punctuation_strip")


def _operator_coverage(operator: str, n: int) -> str:
    if operator in ("typo", "case_flip", "unicode_variant"):
        return f"{len(SEARCH_PLAN_SEEDS)} of 2**64 seeds searched (heuristic)"
    if operator == "whitespace":
        return (
            f"{n} of 3 declared modes produced distinct outcomes (exact); "
            "textually-identical modes are listed as no-ops, never counted"
        )
    return "single deterministic outcome (exact)"


def _unavailable(
    protocol: dict[str, Any], prompt_text: str, fixtures: tuple[Fixture, ...], reason: str
) -> dict[str, Any]:
    return {
        "instrument": PROBE_NAME,
        "protocol_version": PROBE_VERSION,
        "status": "unavailable",
        "reason": reason,
        "protocol": protocol,
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "n_fixtures": len(fixtures),
        "adapter_rule": ADAPTER_RULE,
    }


def probe_surface_noise(
    prompt_text: str,
    fixtures: tuple[Fixture, ...],
    backend: Backend,
    protocol: dict[str, Any],
    *,
    boundary: tuple[int, ...],
    max_new_tokens: int = 64,
    seed: int = 0,
    cancel: threading.Event | None = None,
) -> dict[str, Any]:
    """Run the frozen surface-noise protocol on one prompt.

    Returns the per-operator stability / nearest-observed-flip report, or
    ``status: "cancelled"`` (fail-closed, no score) when the cancel event
    fires mid-sweep, or ``status: "unavailable"`` when the baseline cannot
    be evaluated. The protocol is verified before any scoring.
    """
    prompt_text = _check_prompt(prompt_text)
    fixtures = _check_fixtures(fixtures)
    boundary = _check_boundary(boundary)
    max_new_tokens = _check_max_new_tokens(max_new_tokens)
    seed = _check_seed(seed)
    if cancel is not None and not isinstance(cancel, threading.Event):
        raise ValueError("cancel must be a threading.Event")
    protocol = verify_protocol(protocol, backend_fingerprint=backend.fingerprint, fixtures=fixtures)
    event = cancel if cancel is not None else threading.Event()
    if event.is_set():
        return {
            "instrument": PROBE_NAME,
            "protocol_version": PROBE_VERSION,
            "status": "cancelled",
            "reason": "cancelled_before_baseline",
            "completed_variants": 0,
            "total_variants": 0,
            "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
            "protocol": protocol,
        }

    try:
        base_ids = backend.encode(prompt_text)
    except Exception:  # noqa: BLE001 -- measurement failure, not a crash
        return _unavailable(protocol, prompt_text, fixtures, "baseline_encode_failed")
    if not _valid_ids(base_ids):
        return _unavailable(protocol, prompt_text, fixtures, "baseline_encode_invalid")

    baseline = evaluate_prompt(
        prompt_text,
        fixtures,
        backend,
        boundary=boundary,
        max_new_tokens=max_new_tokens,
        seed=seed,
        cancel=event,
    )
    n_base_abstain = sum(1 for v in baseline if v == "abstain")
    if n_base_abstain == len(fixtures):
        return _unavailable(protocol, prompt_text, fixtures, "baseline_all_abstained")

    operators: dict[str, Any] = {}
    variants = _enumerate_variants(prompt_text)
    total_variants = len(variants)
    done = 0
    for operator in PERTURBATION_OPERATORS:
        op_variants = [v for v in variants if v[1] == operator]
        n_total = len(op_variants)
        n_effective = 0
        n_unchanged = 0
        n_unevaluable = 0
        n_abstained = 0
        n_error = 0
        no_ops: list[str] = []
        flips: list[dict[str, Any]] = []
        nearest: tuple[float, str] | None = None
        for variant_id, _op, detail, perturbed in op_variants:
            if event.is_set():
                return {
                    "instrument": PROBE_NAME,
                    "protocol_version": PROBE_VERSION,
                    "status": "cancelled",
                    "completed_variants": done,
                    "total_variants": total_variants,
                    "protocol": protocol,
                    "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                    "n_fixtures": len(fixtures),
                    "adapter_rule": ADAPTER_RULE,
                }
            try:
                var_ids = backend.encode(perturbed)
            except Exception:  # noqa: BLE001 -- variant-level error, counted
                n_error += 1
                done += 1
                continue
            if not _valid_ids(var_ids) or var_ids == base_ids:
                no_ops.append(variant_id)
                done += 1
                continue
            n_effective += 1
            distance = token_edit_rate(base_ids, var_ids)
            vector = evaluate_prompt(
                perturbed,
                fixtures,
                backend,
                boundary=boundary,
                max_new_tokens=max_new_tokens,
                seed=seed,
                cancel=event,
            )
            if event.is_set():
                return {
                    "instrument": PROBE_NAME,
                    "protocol_version": PROBE_VERSION,
                    "status": "cancelled",
                    "completed_variants": done,
                    "total_variants": total_variants,
                    "protocol": protocol,
                    "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                    "n_fixtures": len(fixtures),
                    "adapter_rule": ADAPTER_RULE,
                }
            n_abstained += sum(1 for v in vector if v == "abstain")
            mutually_evaluable = [
                (case.id, base_v, var_v)
                for case, base_v, var_v in zip(fixtures, baseline, vector, strict=True)
                if base_v != "abstain" and var_v != "abstain"
            ]
            changed = [case_id for case_id, base_v, var_v in mutually_evaluable if base_v != var_v]
            if not mutually_evaluable:
                n_unevaluable += 1
            elif changed:
                flips.append(
                    {
                        "variant_id": variant_id,
                        "detail": detail,
                        "distance": distance,
                        "changed_fixtures": changed,
                    }
                )
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, variant_id)
            else:
                n_unchanged += 1
            done += 1
        n_scored = n_effective - n_unevaluable
        stability = (n_unchanged / n_scored) if n_scored else None
        operators[operator] = {
            "n_total": n_total,
            "n_effective": n_effective,
            "n_abstained": n_abstained,
            "n_error": n_error,
            "n_unchanged": n_unchanged,
            "n_unevaluable": n_unevaluable,
            "stability": stability,
            "stability_note": (
                "stability 1.0 with n_unevaluable > 0 means no mutually-evaluable "
                "fixtures observed a change — not evidence of stability"
                if n_unevaluable
                else None
            ),
            "nearest_observed_flip": (
                {"distance": nearest[0], "variant_id": nearest[1]} if nearest is not None else None
            ),
            "exact": _operator_exact(operator),
            "coverage": _operator_coverage(operator, n_total),
            "no_op_variants": no_ops,
            "flips": flips,
        }

    denominators = {
        key: sum(op[key] for op in operators.values())
        for key in (
            "n_total",
            "n_effective",
            "n_abstained",
            "n_error",
            "n_unchanged",
            "n_unevaluable",
        )
    }
    return {
        "instrument": PROBE_NAME,
        "protocol_version": PROBE_VERSION,
        "status": "ok",
        "protocol": protocol,
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "n_fixtures": len(fixtures),
        "baseline": {
            "n_pass": sum(1 for v in baseline if v == "pass"),
            "n_fail": sum(1 for v in baseline if v == "fail"),
            "n_abstain": n_base_abstain,
        },
        "operators": operators,
        "denominators": denominators,
        "static_lint": static_lint(prompt_text),
        "adapter_rule": ADAPTER_RULE,
    }


def _lint_check(name: str, passed: bool, detail: str, rationale: str) -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail, "rationale": rationale}


def static_lint(prompt_text: str) -> dict[str, Any]:
    """Cheap deterministic pre-sweep sanity panel (advisory only).

    Five fully-specified checks. Advisory: never gates the sweep, never
    blended into a scalar. Each check documents its exact predicate.
    """
    if type(prompt_text) is not str:
        raise ValueError("prompt_text must be a string")
    stripped = prompt_text.strip()
    bidi_found = sorted({c for c in prompt_text if c in _BIDI_CONTROLS})
    sentences = [s.strip() for s in re.split(r"[.!?]+", prompt_text) if s.strip()]
    dup_ratio: float | None = None
    if len(sentences) >= 2:
        dup_ratio = (len(sentences) - len(set(sentences))) / len(sentences)
    checks = [
        _lint_check(
            "nonempty",
            bool(stripped),
            "blank/empty prompt" if not stripped else "has non-whitespace content",
            "An empty prompt has no instruction whose stability could be measured.",
        ),
        _lint_check(
            "bounded_length",
            len(prompt_text) <= MAX_PROMPT_CHARS,
            f"{len(prompt_text)} chars (cap {MAX_PROMPT_CHARS})",
            "Oversized prompts multiply sweep cost across variants.",
        ),
        _lint_check(
            "no_bidi_controls",
            not bidi_found,
            (
                "bidi controls present: " + ",".join(f"U+{ord(c):04X}" for c in bidi_found)
                if bidi_found
                else "no bidi controls"
            ),
            "Bidi controls enable trojan-source-style misreading of the prompt.",
        ),
        _lint_check(
            "terminal_punctuation",
            stripped.endswith((".", "?", "!", ":")),
            (
                "ends with terminal punctuation"
                if stripped.endswith((".", "?", "!", ":"))
                else "no terminal punctuation"
            ),
            "Heuristic only: prompts without terminal punctuation are more "
            "often truncated fragments than complete instructions.",
        ),
        _lint_check(
            "sentence_repetition",
            dup_ratio is None or dup_ratio < 0.5,
            (
                "fewer than 2 sentences"
                if dup_ratio is None
                else f"duplicate-sentence ratio {dup_ratio:.2f}"
            ),
            "Heuristic only: heavy sentence repetition wastes context and "
            "can degrade model attention; it also games noise-robustness.",
        ),
    ]
    return {"advisory_only": True, "checks": checks}


def hardening_notes(probe_result: dict[str, Any]) -> dict[str, Any]:
    """Informational counterfactual surfacing from a probe result.

    Reports the weakest operator family, every observed flip (the
    minimal-flip counterfactuals), and failed lint checks. This is
    *information*, not an optimization loop: see the design doc's
    improvement-loop gating (R14) before building any
    generate->score->generate loop on top of it.
    """
    if type(probe_result) is not dict:
        raise ValueError("probe_result must be a probe result dict")
    operators = probe_result.get("operators")
    if type(operators) is not dict:
        raise ValueError("probe_result has no operators report")
    weakest: str | None = None
    weakest_stability: float | None = None
    counterfactuals: list[str] = []
    for name in sorted(operators):
        op = operators[name]
        if type(op) is not dict:
            continue
        stability = op.get("stability")
        if (
            isinstance(stability, (int, float))
            and not isinstance(stability, bool)
            and op.get("n_effective")
            and (weakest_stability is None or stability < weakest_stability)
        ):
            weakest = name
            weakest_stability = stability
        for flip in op.get("flips") or []:
            if type(flip) is not dict:
                continue
            distance = flip.get("distance")
            distance_str = f"{distance:.4f}" if type(distance) in (int, float) else str(distance)
            counterfactuals.append(
                f"Operator {name}: variant {flip.get('variant_id')} "
                f"(token indel-edit rate {distance_str}) flips "
                f"fixtures {flip.get('changed_fixtures')}"
            )
    lint = probe_result.get("static_lint") or {}
    lint_findings = [
        check.get("name")
        for check in lint.get("checks") or []
        if type(check) is dict and not check.get("passed")
    ]
    return {
        "weakest_operator": weakest,
        "weakest_stability": weakest_stability,
        "counterfactuals": counterfactuals,
        "lint_findings": lint_findings,
        "loop_guidance": (
            "Informational only. Any generate->score->generate loop must "
            "validate acceptance on a held-out operator family, keep a "
            "verbosity guardrail, and report full attempt denominators "
            "(see docs/SURFACE_NOISE_PROBE_DESIGN.md section 12)."
        ),
    }


# Built-in controls: a fixed calibration harness. Each control declares the
# backend output it expects; run_surface_noise_controls builds dedicated
# fixtures from them and scores them with _ControlCalibrationBackend
# (below), so the check is self-contained and always meaningful --
# discrimination can never be silently vacuous the way it would be with
# an arbitrary user backend. Provenance is documented per control: these
# are designer-authored and encode the operator set's own theory of
# fragility (see the Adapter Rule). HELDOUT_CONTROLS are never referenced
# by the test suite: CI pins their count and provenance, not their content.
CONTROLS: tuple[dict[str, str], ...] = (
    {
        "name": "sturdy_numbered",
        "text": (
            "Answer each question with exactly one digit: 0 or 1. "
            "Put the digit on its own line. Do not write anything else."
        ),
        "expected_output": "BLUE",
        "expectation": "sturdy",
        "provenance": (
            "designer-authored; the calibration backend answers BLUE to "
            "any non-empty prompt, and perturbations never produce empty "
            "text (fail-closed), so this control is stable by construction "
            "(encodes operator-set theory)"
        ),
    },
    {
        "name": "fragile_exact_echo",
        "text": "Reply with exactly: OK",
        "expectation": "fragile",
        "expected_output": "OK",
        "provenance": (
            "designer-authored; the calibration backend requires the exact "
            "substring 'exactly: OK', which whitespace newline-splitting, "
            "case flips, Unicode variants, typos, and punctuation stripping "
            "all break (encodes operator-set theory)"
        ),
    },
)

HELDOUT_CONTROLS: tuple[dict[str, str], ...] = (
    {
        "name": "heldout_sturdy_list",
        "text": (
            "Respond with a numbered list. Each item goes on its own line, "
            "starting with the number, a dot, and a space."
        ),
        "expected_output": "BLUE",
        "expectation": "sturdy",
        "provenance": (
            "designer-authored held-out; same non-empty mechanism as "
            "sturdy_numbered with different text; never referenced by the "
            "test suite"
        ),
    },
    {
        "name": "heldout_fragile_shout",
        "text": "SAY IT LOUD: HELLO",
        "expected_output": "LOUD",
        "expectation": "fragile",
        "provenance": (
            "designer-authored held-out; the calibration backend requires "
            "the exact uppercase substring, which case_flip noise "
            "deterministically breaks; never referenced by the test suite"
        ),
    },
)


class _ControlCalibrationBackend:
    """Fixed backend for the control harness (not a prompt measurement).

    Rule order: exact "exactly: OK" -> "OK"; exact "SAY IT LOUD: HELLO"
    -> "LOUD"; any other non-empty prompt -> "BLUE". Ord/chr codec, like
    the repo's scripted test backends.
    """

    fingerprint = "surface-noise-control-calibration/v1"

    @staticmethod
    def encode(text: str) -> tuple[int, ...]:
        if type(text) is not str:
            raise ValueError("text must be a string")
        return tuple(map(ord, text))

    @staticmethod
    def decode(ids: tuple[int, ...]) -> str:
        return "".join(map(chr, ids))

    def generate(
        self,
        input_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        seed: int,
        cancel: threading.Event | None,
    ) -> Generation:
        if cancel is not None and cancel.is_set():
            return Generation((), "cancelled")
        prompt = self.decode(input_ids)
        if "exactly: OK" in prompt:
            text = "OK"
        elif "SAY IT LOUD: HELLO" in prompt:
            text = "LOUD"
        elif prompt.strip():
            text = "BLUE"
        else:
            text = "WRONG"
        ids = self.encode(text + chr(7))
        return Generation(ids + (1,) * (max_new_tokens - len(ids)), "limit")


class SelfTestBackend:
    """Deterministic instrument self-test backend (not a prompt measurement).

    Answers from a marker in the prompt span: fixtures expecting "BLUE"
    pass when the marker is present verbatim, "WRONG" otherwise. Lets
    users and agents exercise the full probe loop without a checkpoint.
    The fingerprint includes a hash of the marker (it changes behavior).
    """

    def __init__(self, marker: str = "MAGICWORD"):
        if type(marker) is not str or not marker:
            raise ValueError("marker must be a nonempty string")
        self._marker = marker
        # The marker changes behavior, so it is part of the fingerprint:
        # the comparability contract keys on (fixtures, backend, protocol).
        digest = hashlib.sha256(marker.encode("utf-8")).hexdigest()[:16]
        self.fingerprint = f"surface-noise-selftest/v1+marker:{digest}"

    @staticmethod
    def encode(text: str) -> tuple[int, ...]:
        if type(text) is not str:
            raise ValueError("text must be a string")
        return tuple(map(ord, text))

    @staticmethod
    def decode(ids: tuple[int, ...]) -> str:
        return "".join(map(chr, ids))

    def generate(
        self,
        input_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        seed: int,
        cancel: threading.Event | None,
    ) -> Generation:
        if cancel is not None and cancel.is_set():
            return Generation((), "cancelled")
        prompt = self.decode(input_ids)
        text = ("BLUE" + chr(7)) if self._marker in prompt else ("WRONG" + chr(7))
        ids = self.encode(text)
        return Generation(ids + (1,) * (max_new_tokens - len(ids)), "limit")


def _mean_stability(operators: dict[str, Any]) -> float | None:
    values: list[float] = [
        float(op["stability"])
        for op in operators.values()
        if type(op) is dict
        and isinstance(op.get("stability"), (int, float))
        and not isinstance(op.get("stability"), bool)
        and op.get("n_effective")
    ]
    if not values:
        return None
    return sum(values) / len(values)


def run_surface_noise_controls(
    *,
    boundary: tuple[int, ...],
    include_heldout: bool = False,
    max_new_tokens: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the built-in control harness; check sturdy > fragile.

    Self-contained: dedicated fixtures are built from the controls'
    declared expected outputs and scored with the fixed calibration
    backend under a freshly frozen protocol. Returns per-control mean
    stabilities and the discrimination verdict. Controls validate
    discrimination *within the operator set's theory of fragility* (see
    ADAPTER_RULE), not against real-world prompt failure.
    """
    boundary = _check_boundary(boundary)
    max_new_tokens = _check_max_new_tokens(max_new_tokens)
    seed = _check_seed(seed)
    if type(include_heldout) is not bool:
        raise ValueError("include_heldout must be a bool")
    controls = CONTROLS + HELDOUT_CONTROLS if include_heldout else CONTROLS
    backend = _ControlCalibrationBackend()
    fixtures = tuple(
        Fixture(
            id=f"control-{control['name']}",
            messages=(Message(role="user", content="probe"),),
            expected=control["expected_output"],
            group="controls",
        )
        for control in controls
    )
    protocol = freeze_surface_noise_protocol(
        backend_fingerprint=backend.fingerprint, fixtures=fixtures
    )
    results: list[dict[str, Any]] = []
    for control in controls:
        report = probe_surface_noise(
            control["text"],
            fixtures,
            backend,
            protocol,
            boundary=boundary,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        if report.get("status") != "ok":
            results.append(
                {
                    "name": control["name"],
                    "expectation": control["expectation"],
                    "provenance": control["provenance"],
                    "status": report.get("status"),
                    "mean_stability": None,
                }
            )
            continue
        results.append(
            {
                "name": control["name"],
                "expectation": control["expectation"],
                "provenance": control["provenance"],
                "status": "ok",
                "mean_stability": _mean_stability(report["operators"]),
            }
        )
    sturdy = [
        r["mean_stability"]
        for r in results
        if r["expectation"] == "sturdy" and type(r["mean_stability"]) in (int, float)
    ]
    fragile = [
        r["mean_stability"]
        for r in results
        if r["expectation"] == "fragile" and type(r["mean_stability"]) in (int, float)
    ]
    discriminates: bool | None = None
    if sturdy and fragile:
        discriminates = sum(sturdy) / len(sturdy) > sum(fragile) / len(fragile)
    return {
        "controls": results,
        "discriminates": discriminates,
        "circularity_note": (
            "Controls encode the operator set's own theory of fragility; "
            "a pass validates the implementation against the specification, "
            "not against real-world prompt failure."
        ),
    }


# ---------------------------------------------------------------------------
# Standalone dashboard (self-contained HTML; inline CSS+SVG, no external assets)
# ---------------------------------------------------------------------------


def _regenerate_variant(prompt_text: str, operator: str, detail: dict[str, Any]) -> str | None:
    """Rebuild a variant's perturbed text from the frozen plan (deterministic)."""
    if operator in ("typo", "case_flip", "unicode_variant"):
        seed = detail.get("seed")
        if type(seed) is not int:
            return None
        return perturb_text(prompt_text, operator, seed=seed)
    if operator == "whitespace":
        index = detail.get("mode_index")
        modes = whitespace_modes(prompt_text)
        if type(index) is not int or not 0 <= index < len(modes):
            return None
        return modes[index]
    if operator == "punctuation_strip":
        return perturb_text(prompt_text, "punctuation_strip", seed=0)
    return None


def _excerpt(text: str, limit: int = 200) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def render_surface_noise_dashboard(report: dict[str, Any], prompt_text: str | None = None) -> str:
    """Self-contained HTML dashboard for a probe report.

    Inline CSS + SVG, no external assets, restrictive CSP. Every
    interpolated string is escaped (including inside SVG text nodes).
    The prompt text itself is never rendered raw -- only its sha256;
    minimal-flip counterfactual excerpts are shown only when
    ``prompt_text`` is supplied *and* its sha256 matches the report.
    """
    if type(report) is not dict:
        raise ValueError("report must be a probe result dict")
    if prompt_text is not None and type(prompt_text) is not str:
        raise ValueError("prompt_text must be a string or None")
    show_excerpts = prompt_text is not None and hashlib.sha256(
        prompt_text.encode("utf-8")
    ).hexdigest() == report.get("prompt_sha256")

    operators = report.get("operators") or {}
    baseline = report.get("baseline") or {}
    denominators = report.get("denominators") or {}
    lint = report.get("static_lint") or {}
    protocol = report.get("protocol") or {}

    # Per-operator SVG stability bars (one row per operator).
    bar_rows: list[str] = []
    y = 0
    for name in sorted(operators):
        op = operators[name]
        if type(op) is not dict:
            continue
        stability = op.get("stability")
        if isinstance(stability, (int, float)) and not isinstance(stability, bool):
            width = max(0.0, min(1.0, stability)) * 220.0
            label = f"{stability:.2f}"
        else:
            width = 0.0
            label = "n/a"
        bar_rows.append(
            f'<text x="0" y="{y + 14}" class="lbl">{escape(name)}</text>'
            f'<rect x="150" y="{y}" width="{width:.1f}" height="14" class="bar"/>'
            f'<rect x="150" y="{y}" width="220" height="14" class="frame"/>'
            f'<text x="378" y="{y + 14}" class="lbl">{escape(label)}</text>'
        )
        y += 24
    svg_height = max(y, 24)
    svg = (
        f'<svg width="460" height="{svg_height}" role="img" '
        f'aria-label="per-operator stability bars">' + "".join(bar_rows) + "</svg>"
    )

    # Operator detail table.
    detail_rows: list[str] = []
    for name in sorted(operators):
        op = operators[name]
        if type(op) is not dict:
            continue
        nearest = op.get("nearest_observed_flip")
        if type(nearest) is dict and type(nearest.get("distance")) in (int, float):
            flip_cell = f"{nearest['distance']:.4f} ({escape(str(nearest.get('variant_id')))})"
        else:
            flip_cell = "none observed"
        stability = op.get("stability")
        detail_rows.append(
            "<tr><td>"
            + escape(name)
            + "</td><td>"
            + escape(f"{op.get('n_effective')}/{op.get('n_total')}")
            + "</td><td>"
            + escape(f"{stability:.2f}" if type(stability) in (int, float) else "n/a")
            + "</td><td>"
            + flip_cell
            + "</td><td>"
            + escape("exact" if op.get("exact") else "heuristic")
            + "</td><td>"
            + escape(str(op.get("coverage")))
            + "</td></tr>"
        )

    # Minimal-flip counterfactuals.
    counterfactuals: list[str] = []
    for name in sorted(operators):
        op = operators[name]
        if type(op) is not dict:
            continue
        for flip in op.get("flips") or []:
            if type(flip) is not dict:
                continue
            distance = flip.get("distance")
            distance_str = (
                f"{distance:.4f}" if type(distance) in (int, float) else escape(str(distance))
            )
            line = (
                f"Operator {escape(name)}: variant "
                f"{escape(str(flip.get('variant_id')))} "
                f"(token indel-edit rate {distance_str}) flips "
                f"fixtures {escape(str(flip.get('changed_fixtures')))}"
            )
            if show_excerpts and prompt_text is not None:
                text = _regenerate_variant(prompt_text, name, flip.get("detail") or {})
                if text is not None:
                    line += f'<br><span class="mono">{escape(_excerpt(text))}</span>'
            counterfactuals.append(f"<li>{line}</li>")

    # Static lint panel (advisory).
    lint_rows: list[str] = []
    for check in lint.get("checks") or []:
        if type(check) is not dict:
            continue
        mark = "✓" if check.get("passed") else "✗"
        lint_rows.append(
            "<tr><td>"
            + escape(str(check.get("name")))
            + "</td><td>"
            + escape(mark)
            + "</td><td>"
            + escape(str(check.get("detail")))
            + "</td></tr>"
        )

    not_measured = [
        "semantic paraphrase of instructions",
        "conflicting-instruction override / prompt injection",
        "hierarchy attacks",
        "tool-call schema adherence under noise",
        "multi-turn drift",
        "context-length degradation",
    ]
    search_plan = protocol.get("search_plan") or {}

    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        '<meta http-equiv="Content-Security-Policy"'
        " content=\"default-src 'none'; style-src 'unsafe-inline'\">"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Surface-noise self-consistency probe</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;max-width:70rem;margin:2rem auto;"
        "padding:0 1rem;color:#1a1a1a;background:#fff}"
        ".scope{background:#fff8e1;border-left:4px solid #f9a825;padding:.75rem 1rem}"
        ".card{border:1px solid #ddd;border-radius:.5rem;padding:1rem;margin:1rem 0}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ddd;padding:.4rem .6rem;text-align:left}"
        "th{background:#f5f5f5}.mono{font-family:monospace;font-size:.85em}"
        ".bar{fill:#2563eb}.frame{fill:none;stroke:#999}"
        ".lbl{font-size:12px;fill:#1a1a1a}"
        ".warn{background:#fef2f2;border-left:4px solid #dc2626;padding:.75rem 1rem}"
        "</style></head><body>"
        "<h1>Surface-noise self-consistency probe</h1>"
        f'<div class="scope"><strong>Scope — read first.</strong> {escape(ADAPTER_RULE)}</div>'
        '<div class="card"><h2>Baseline</h2><p>'
        f"Pass: <strong>{escape(str(baseline.get('n_pass')))}</strong> / "
        f"{escape(str(report.get('n_fixtures')))} fixtures "
        f"({escape(str(baseline.get('n_abstain')))} abstained). "
        'Prompt sha256: <span class="mono">'
        f"{escape(str(report.get('prompt_sha256')))}</span>."
        "</p></div>"
        '<div class="card"><h2>Per-operator stability</h2>'
        + svg
        + "<table><tr><th>Operator</th><th>Effective / total</th><th>Stability</th>"
        "<th>Nearest observed flip (token indel-edit rate)</th>"
        "<th>Exact / heuristic</th><th>Search coverage</th></tr>"
        + "".join(detail_rows)
        + "</table><p>Stability = unchanged / effective variants. "
        "“Nearest observed flip” is an <em>upper bound</em> on the true "
        "minimal-flip distance — it errs toward false confidence and is "
        "never a guarantee.</p></div>"
        '<div class="card"><h2>Static lint (advisory — never gates, never blended)</h2>'
        "<table><tr><th>Check</th><th>Pass</th><th>Detail</th></tr>"
        + "".join(lint_rows)
        + "</table></div>"
        '<div class="card"><h2>Minimal-flip counterfactuals</h2><ul>'
        + ("".join(counterfactuals) if counterfactuals else "<li>None observed.</li>")
        + "</ul></div>"
        '<div class="warn"><h2>Not measured</h2><p>This instrument measures '
        "<strong>surface-form noise only</strong>. It says nothing about:</p><ul>"
        + "".join(f"<li>{escape(item)}</li>" for item in not_measured)
        + "</ul></div>"
        '<div class="card"><h2>Denominators &amp; protocol</h2><p>'
        + escape(
            f"n_total={denominators.get('n_total')} "
            f"n_effective={denominators.get('n_effective')} "
            f"n_abstained={denominators.get('n_abstained')} "
            f"n_error={denominators.get('n_error')}"
        )
        + '</p><p>Frozen search plan: <span class="mono">'
        + escape(json.dumps(search_plan, sort_keys=True))
        + '</span></p><p>Backend fingerprint: <span class="mono">'
        + escape(str(protocol.get("backend_fingerprint")))
        + '</span>; fixture sha256: <span class="mono">'
        + escape(str(protocol.get("fixture_sha256")))
        + "</span>.</p>"
        "<p><strong>Comparability:</strong> scores are comparable only under "
        "identical (fixtures, backend, protocol-fingerprint) triples.</p></div>"
        "</body></html>"
    )


# ---------------------------------------------------------------------------
# CLI: `mpl surface-noise probe|diff|protocol`
# ---------------------------------------------------------------------------

MONDAY_MORNING_USE = (
    "You rewrote a prompt over the weekend — the probe tells you whether "
    "a typo, case change, or whitespace shuffle would flip its answers, "
    "before you ship it."
)

_PROMPT_SIZE_CAP = 1 << 20  # 1 MiB


def _read_prompt_file(path: Path) -> str:
    """Read a prompt file defensively: regular file, size-capped, UTF-8."""
    target = Path(path)
    try:
        st = target.stat()
    except OSError as exc:
        raise ValueError(f"cannot read prompt file: {exc.strerror or exc}") from exc
    if not stat.S_ISREG(st.st_mode):
        raise ValueError("prompt path must be a regular file")
    if st.st_size > _PROMPT_SIZE_CAP:
        raise ValueError(f"prompt file exceeds {_PROMPT_SIZE_CAP} bytes")
    try:
        return target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read prompt file: {exc}") from exc


def _parse_boundary_arg(value: str) -> tuple[int, ...]:
    """Argparse type for --boundary-ids (local copy: no cli import cycle)."""
    parts = value.split(",")
    ids: list[int] = []
    for part in parts:
        part = part.strip()
        if not part or not part.isascii() or not part.isdigit():
            raise argparse.ArgumentTypeError("Use ID,ID (for example 7 or 198,2)")
        num = int(part)
        if num >= 2**31:
            raise argparse.ArgumentTypeError("boundary id too large")
        ids.append(num)
    if not ids or len(ids) > 16:
        raise argparse.ArgumentTypeError("Use 1-16 comma-separated IDs")
    return tuple(ids)


def describe_instrument() -> dict[str, Any]:
    """Frozen instrument descriptor (no fixtures, no backend, no scoring)."""
    return {
        "instrument": PROBE_NAME,
        "protocol_version": PROBE_VERSION,
        "monday_morning_use": MONDAY_MORNING_USE,
        "adapter_rule": ADAPTER_RULE,
        "distance": (
            "Token indel-edit rate: Myers O(ND) diff, substitution = "
            "delete+insert, normalized by the combined token-id length. "
            "Unit: fraction of token positions changed, in [0, 1]."
        ),
        "search_plan": frozen_search_plan(),
        "verdict_function": (
            "V(P, F, B): per-fixture pass (decoded answer == expected) / "
            "fail (any other text) / abstain (error); no per-prompt V exists, "
            "so every operator reports a vector, never a scalar."
        ),
        "radius_framing": (
            "The reported 'nearest observed flip' is an UPPER BOUND over "
            "the frozen searched subset S: d_obs >= d*. It errs toward "
            "false confidence and is never a robustness guarantee."
        ),
        "comparability": (
            "Scores comparable only under identical "
            "(fixtures, backend, protocol-fingerprint) triples."
        ),
        "not_measured": [
            "semantic paraphrase of instructions",
            "conflicting-instruction override / prompt injection",
            "hierarchy attacks",
            "tool-call schema adherence under noise",
            "multi-turn drift",
            "context-length degradation",
        ],
    }


def build_surface_noise_parser(sub: Any) -> None:
    """Register the ``surface-noise`` subcommand (called from cli.build_parser)."""
    sn_cmd = sub.add_parser(
        "surface-noise",
        help="Surface-noise self-consistency probe: " + MONDAY_MORNING_USE,
        allow_abbrev=False,
    )
    sn_sub = sn_cmd.add_subparsers(dest="surface_noise_command", required=True)

    probe_cmd = sn_sub.add_parser(
        "probe",
        help="Probe one prompt's verdict stability under frozen surface noise",
        allow_abbrev=False,
    )
    probe_cmd.add_argument(
        "--prompt", required=True, type=Path, help="Text file with the prompt to probe"
    )
    probe_cmd.add_argument(
        "--fixtures",
        required=True,
        type=Path,
        help="Fixture JSON file (its search split is used)",
    )
    probe_cmd.add_argument(
        "--boundary-ids",
        required=True,
        type=_parse_boundary_arg,
        help="Comma-separated boundary token IDs (for example 7)",
    )
    probe_cmd.add_argument(
        "--backend",
        choices=("selftest", "local-hf"),
        default="selftest",
        help="selftest: deterministic marker backend (default); "
        "local-hf: a real checkpoint (needs --model)",
    )
    probe_cmd.add_argument(
        "--model", type=Path, default=None, help="Checkpoint path (local-hf only)"
    )
    probe_cmd.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New output directory (probe.json + dashboard.html); "
        "an existing path is never overwritten",
    )
    probe_cmd.add_argument("--max-new-tokens", type=int, default=64)
    probe_cmd.add_argument("--seed", type=int, default=0)

    diff_cmd = sn_sub.add_parser(
        "diff",
        help="Compare two probe.json files (before/after a prompt edit)",
        allow_abbrev=False,
    )
    diff_cmd.add_argument("old_probe", type=Path, help="Older probe.json")
    diff_cmd.add_argument("new_probe", type=Path, help="Newer probe.json")

    sn_sub.add_parser(
        "protocol",
        help="Print the frozen instrument descriptor (no scoring)",
        allow_abbrev=False,
    )


def _build_cli_backend(args: Any) -> Any:
    if args.backend == "selftest":
        return SelfTestBackend()
    if args.model is None:
        raise ValueError("--model is required for --backend local-hf")
    from .backends import LocalHFBackend

    return LocalHFBackend(args.model)


def _load_probe_json(path: Path) -> Any:
    target = Path(path)
    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if len(raw) > 16 << 20:
        raise ValueError(f"{path} exceeds 16 MiB")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc


def run_surface_noise(args: Any, parser: Any) -> int:
    """Implement ``mpl surface-noise``; returns the process exit code.

    probe: JSON goes to stdout and to <output>/probe.json, the human
    summary to stderr, dashboard.html alongside. Exit 0 when the sweep
    completed, 2 when it was cancelled, 3 when the baseline was
    unavailable (no score produced), 1 for invalid input.
    diff: JSON to stdout, summary to stderr, exit 0/1.
    protocol: descriptor JSON to stdout, exit 0.
    """
    command = args.surface_noise_command
    if command == "protocol":
        print(json.dumps(describe_instrument(), indent=2, ensure_ascii=True, sort_keys=True))
        return 0
    if command == "diff":
        try:
            old = _load_probe_json(args.old_probe)
            new = _load_probe_json(args.new_probe)
        except ValueError as exc:
            print(f"surface-noise diff: {exc}", file=sys.stderr)
            return 1
        from .diffreports import diff_surface_noise

        diff = diff_surface_noise(old, new)
        print(json.dumps(diff, indent=2, ensure_ascii=True, sort_keys=True))
        summary = (
            f"surface-noise diff: available={diff['available']} "
            f"changed={diff['changed']} identical={diff['identical']}"
        )
        if diff.get("protocol_drift"):
            summary += " WARNING: protocol fingerprint changed — not comparable"
        if diff.get("prompt_changed"):
            summary += " (prompt text changed)"
        print(summary, file=sys.stderr)
        return 0
    # command == "probe"
    try:
        prompt_text = _read_prompt_file(args.prompt)
        from .datasets import load_dataset

        split = load_dataset(args.fixtures)  # nosec B615 -- own bounded JSON reader
        fixtures = split.search  # load_dataset guarantees a nonempty search split
        backend = _build_cli_backend(args)
        protocol = freeze_surface_noise_protocol(
            backend_fingerprint=backend.fingerprint, fixtures=fixtures
        )
        import signal as _signal

        cancel = threading.Event()
        _previous_int = _signal.getsignal(_signal.SIGINT)
        _previous_term = _signal.getsignal(_signal.SIGTERM)
        _signal.signal(_signal.SIGINT, lambda _s, _f: cancel.set())
        _signal.signal(_signal.SIGTERM, lambda _s, _f: cancel.set())
        try:
            report = probe_surface_noise(
                prompt_text,
                fixtures,
                backend,
                protocol,
                boundary=args.boundary_ids,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                cancel=cancel,
            )
        finally:
            _signal.signal(_signal.SIGINT, _previous_int)
            _signal.signal(_signal.SIGTERM, _previous_term)
    except ValueError as exc:
        print(f"surface-noise probe: {exc}", file=sys.stderr)
        return 1
    status = report.get("status")
    out_dir = Path(args.output)
    if out_dir.exists():
        print(
            f"surface-noise probe: output path {out_dir} exists; refusing to overwrite",
            file=sys.stderr,
        )
        return 1
    out_dir.mkdir(parents=True)
    (out_dir / "probe.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "dashboard.html").write_text(
        render_surface_noise_dashboard(report, prompt_text=prompt_text),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True))
    operators = report.get("operators") or {}
    stable = sum(1 for op in operators.values() if type(op) is dict and op.get("stability") == 1.0)
    print(
        f"surface-noise probe: status={status} "
        f"{stable}/{len(operators)} operators fully stable; "
        f"wrote {out_dir}/probe.json + dashboard.html",
        file=sys.stderr,
    )
    if status == "cancelled":
        return 2
    if status == "unavailable":
        return 3
    return 0
