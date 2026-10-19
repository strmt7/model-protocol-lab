"""Few-shot demonstration gain x template family (research wave-4 idea 2).

The question: does in-context learning work the same under every chat
template? For a fixed demo set, the zero-shot -> k-shot verdict-change rate
(gain) and the order sensitivity over the demo permutations are measured per
candidate on held-out fixtures, with model, demos, task and decoder fixed --
the template is the only independent variable. The ICL literature
(arXiv:2511.09700 "Order Matters", Nov 2025) varies demo order on one fixed
prompt format per study; nobody varies the chat template as the independent
variable. The lab's 29 families x exact token sequences is the missing
instrument.

Protocol (v1, pre-registered here):

- ``k = 2`` demos sampled (seeded keyed-hash selection for input-order
  independence) from the SEARCH split only -- never held-out. The demo
  pool is the search split; the measurement set is the held-out split,
  so a demo fixture can never evaluate itself (the contamination guard
  the brief requires).
- Both orders of the 2 demos (the two-orders design: the minimal
  order-sensitivity probe). Demos are prepended as real user/assistant turns
  before the fixture's own messages, so every candidate renders them through
  its own template; the fixture's ``expected`` is unchanged, so the
  pipeline's ``exact_match`` already grades the demo-conditioned run (the
  same prefix-cancellation argument as the prefill probe: demos are context,
  not the answer).
- ``gain`` = mean over the two orders of the per-fixture verdict-change
  rate (k-shot vs 0-shot), in [0, 1].
- ``order_sensitivity`` = population standard deviation of the two
  per-order change rates, in [0, 0.5].

Falsifiable claim: gain and order_sensitivity are identical across all 29
template families. Proved wrong if families differ -- i.e. the template, not
just the demos, determines whether in-context learning works.

Honest framing: "gain" is a verdict-CHANGE rate, not a quality judgment --
a high gain means the demos moved the verdicts, not that they helped. The
demos are text-level context; nothing here claims the demos are good, or
that a different k would behave the same (k=2 is the v1 protocol).

Everything here is pure: demo sampling, permutation, fixture building and
the gain/sensitivity arithmetic need no model.
"""

from __future__ import annotations

import hashlib
import itertools
import math

from .core import Fixture, Message

#: v1 protocol: k=2 demos, both orders (the two-orders design).
FEWSHOT_K = 2

#: Fixed demo-sampling seed (v1 pins this; the sampler is pure and
#: deterministic, so the demo set is reproducible across runs).
FEWSHOT_SEED = 20261019

#: Arm identifiers: the 0-shot baseline and the two demo orders.
ARMS = ("baseline", "order_a", "order_b")

#: Fail-closed detail codes (recorded, never raised through the stage).
DETAIL_INSUFFICIENT_DEMO_POOL = "insufficient_demo_pool"


def _fixture_tuple(fixtures: object, what: str) -> tuple[Fixture, ...]:
    """Validate a nonempty tuple of Fixture; raise ValueError otherwise."""
    if type(fixtures) is not tuple or not fixtures:
        raise ValueError(f"{what} must be a nonempty tuple")
    if any(type(case) is not Fixture for case in fixtures):
        raise ValueError(f"{what} must contain Fixture values")
    return fixtures


def validate_demo_pool(fixtures: object) -> tuple[Fixture, ...]:
    """Validate the search-split demo pool; returns it as a tuple."""
    return _fixture_tuple(fixtures, "demo pool")


def validate_eval_fixtures(fixtures: object) -> tuple[Fixture, ...]:
    """Validate the held-out measurement fixtures; returns them as a tuple."""
    return _fixture_tuple(fixtures, "eval fixtures")


def _demo_key(seed: int, fixture_id: str) -> tuple[bytes, str]:
    """Keyed sort key for seeded demo selection.

    SHA-256 over ``"<seed>:<fixture id>"``: deterministic in the seed,
    independent of the caller's input order, and with no PRNG state to
    reason about (the fixture id is the tiebreak; a sha256 collision
    between distinct ids is not a practical concern). Seeded sampling
    without ``random`` -- neither a cryptographic use nor a place where
    unpredictability matters, so no securerandom machinery is needed.
    """
    digest = hashlib.sha256(f"{seed}:{fixture_id}".encode()).digest()
    return (digest, fixture_id)


def sample_demos(
    pool: tuple[Fixture, ...],
    k: int = FEWSHOT_K,
    seed: int = FEWSHOT_SEED,
) -> tuple[Fixture, ...]:
    """Seeded demo-set sampler over the search-split pool.

    The k fixtures with the smallest keyed ``(sha256(seed, id), id)``
    sort keys are returned in key order: the selected set is
    independent of the caller's input order and reproducible for a
    fixed seed. Returns exactly ``k`` demos in sampled (key) order.
    Raises ``ValueError`` on a bad pool, a non-positive ``k``, a bad
    seed, or a pool smaller than ``k`` (the caller -- the stage --
    converts the too-small case into the fail-closed
    ``insufficient_demo_pool`` detail instead of raising).
    """
    clean = validate_demo_pool(pool)
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if len(clean) < k:
        raise ValueError(f"demo pool holds {len(clean)} fixtures, fewer than k={k}")
    keyed = sorted(clean, key=lambda case: _demo_key(seed, case.id))
    return tuple(keyed[:k])


def demo_permutations(demos: tuple[Fixture, ...]) -> tuple[tuple[Fixture, ...], ...]:
    """All orders of the demo set (the seeded permutation generator).

    For the v1 protocol (k=2) this is exactly the two orders. Raises
    ``ValueError`` on a bad demo tuple.
    """
    clean = validate_demo_pool(demos)
    return tuple(itertools.permutations(clean))


def demo_question_text(demo: Fixture) -> str:
    """The demo's question text: its user-turn contents joined by newline.

    Raises ``ValueError`` when the demo fixture has no user message (a
    demo without a question is meaningless).
    """
    if type(demo) is not Fixture:
        raise ValueError("demo must be a Fixture")
    parts = [message.content for message in demo.messages if message.role == "user"]
    if not parts:
        raise ValueError("demo fixture has no user message")
    return "\n".join(parts)


def build_demo_messages(demo: Fixture, index: int, n_demos: int) -> tuple[Message, Message]:
    """One demo as a (user, assistant) turn pair.

    The user turn carries the fixed header, the question and the answer
    (``A:`` at a line start, which scripted policies can locate); the
    assistant turn confirms the answer so the conversation keeps proper
    turn alternation for every template. Raises ``ValueError`` on bad
    input.
    """
    if type(demo) is not Fixture:
        raise ValueError("demo must be a Fixture")
    if type(index) is not int or type(n_demos) is not int:
        raise ValueError("index and n_demos must be integers")
    if not 1 <= index <= n_demos:
        raise ValueError("index must satisfy 1 <= index <= n_demos")
    question = demo_question_text(demo)
    user_text = f"Example {index} of {n_demos}:\nQ: {question}\nA: {demo.expected}"
    return (Message("user", user_text), Message("assistant", demo.expected))


def _messages_alternate_user_last(messages: tuple[Message, ...]) -> bool:
    """True iff messages alternate user/assistant starting AND ending with user."""
    if not messages or messages[0].role != "user" or messages[-1].role != "user":
        return False
    return all(
        message.role == ("user" if index % 2 == 0 else "assistant")
        for index, message in enumerate(messages)
    )


def build_demo_fixtures(
    eval_fixtures: tuple[Fixture, ...],
    demos: tuple[Fixture, ...],
    *,
    perm_index: int,
) -> tuple[Fixture, ...]:
    """Prepend one demo order to every eval fixture.

    The demos (in the given order) become real leading turns; the eval
    fixture's messages, id-namespace and ``expected`` follow unchanged
    apart from the namespaced id (``fewshot/perm{i}/<id>``) so
    demo-conditioned runs never collide with primary fixture ids.
    Alternation is checked up front: the combined message list must
    alternate user/assistant and end with a user turn (the renderer's
    contract), otherwise ``ValueError`` -- a fail-fast caller-contract
    check, not a per-cell degradation, since every cell would fail
    identically.
    """
    clean_eval = validate_eval_fixtures(eval_fixtures)
    clean_demos = validate_demo_pool(demos)
    if type(perm_index) is not int or perm_index < 0:
        raise ValueError("perm_index must be a nonnegative integer")
    demo_turns: tuple[Message, ...] = ()
    for position, demo in enumerate(clean_demos, start=1):
        demo_turns += build_demo_messages(demo, position, len(clean_demos))
    built: list[Fixture] = []
    for case in clean_eval:
        messages = demo_turns + case.messages
        if not _messages_alternate_user_last(messages):
            raise ValueError(
                f"eval fixture {case.id!r} messages do not alternate user/assistant "
                "ending with a user turn"
            )
        built.append(
            Fixture(
                id=f"fewshot/perm{perm_index}/{case.id}",
                messages=messages,
                expected=case.expected,
                group=case.group,
            )
        )
    return tuple(built)


def original_fixture_id(namespaced_id: str) -> str:
    """Strip the ``fewshot/perm{i}/`` namespace back to the eval fixture id."""
    if type(namespaced_id) is not str:
        raise ValueError("namespaced id must be a string")
    prefix = "fewshot/perm"
    if not namespaced_id.startswith(prefix):
        raise ValueError("id is not a fewshot-namespaced fixture id")
    _head, _sep, rest = namespaced_id[len(prefix) :].partition("/")
    if not rest:
        raise ValueError("id is not a fewshot-namespaced fixture id")
    return rest


def change_rate(
    baseline: dict[str, bool | None],
    arm: dict[str, bool | None],
) -> dict[str, int | float | None]:
    """Per-fixture verdict-change rate of one demo order vs the baseline.

    Only fixtures with a non-None verdict on BOTH sides are evaluable;
    unevaluable fixtures are excluded, never counted as unchanged.
    Returns ``{"n_evaluable", "n_changed", "rate"}`` with ``rate`` None
    when nothing is evaluable. Raises ``ValueError`` on non-dict or
    non-bool/None verdict maps.
    """
    for name, verdicts in (("baseline", baseline), ("arm", arm)):
        if type(verdicts) is not dict:
            raise ValueError(f"{name} verdicts must be a dict")
        for fid, verdict in verdicts.items():
            if type(fid) is not str or (verdict is not None and type(verdict) is not bool):
                raise ValueError(f"{name} verdicts must map str fixture ids to bool|None")
    fids = [fid for fid in baseline if fid in arm]
    evaluable = [fid for fid in fids if baseline[fid] is not None and arm[fid] is not None]
    changed = sum(1 for fid in evaluable if baseline[fid] != arm[fid])
    return {
        "n_evaluable": len(evaluable),
        "n_changed": changed,
        "rate": (changed / len(evaluable)) if evaluable else None,
    }


def order_sensitivity(rates: tuple[float | None, ...]) -> float | None:
    """Population std of the per-order change rates; None when empty.

    With the v1 two-orders design this is |r_a - r_b| / 2, in [0, 0.5].
    None rates are skipped (an order with no evaluable fixture contributes
    no information); None is returned when no order has a rate. Raises
    ``ValueError`` on non-tuple input or rates outside [0, 1].
    """
    if type(rates) is not tuple:
        raise ValueError("rates must be a tuple")
    clean: list[float] = []
    for rate in rates:
        if rate is None:
            continue
        if type(rate) not in (int, float) or isinstance(rate, bool):
            raise ValueError("rates must be floats in [0, 1] or None")
        if not 0.0 <= rate <= 1.0:
            raise ValueError("rates must be floats in [0, 1] or None")
        clean.append(float(rate))
    if not clean:
        return None
    mean = sum(clean) / len(clean)
    return math.sqrt(sum((rate - mean) ** 2 for rate in clean) / len(clean))
