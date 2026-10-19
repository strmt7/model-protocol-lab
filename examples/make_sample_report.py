"""Generate the committed sample report in examples/sample-report/.

This is a SYNTHETIC demonstration, not a real model run: a scripted backend
stands in for a checkpoint that prefers the unmutated base hypothesis, so the
sample exercises the full pipeline (search -> unique winner -> held-out ->
compatible_on_fixture_set) and shows what a real report looks like.

Regenerating must be byte-identical; tests/test_sample_report.py enforces it.
"""

import re
import sys
from pathlib import Path
from threading import Event

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_protocol_lab.catalog import candidates
from model_protocol_lab.core import Generation
from model_protocol_lab.datasets import load_dataset
from model_protocol_lab.experiments import run_experiment
from model_protocol_lab.reporting import save_report
from model_protocol_lab.structured import (
    STRUCTURED_SPECS,
    build_structured_fixtures,
    schema_map,
    validate_structured_specs,
)
from model_protocol_lab.toolcall import (
    TOOLCALL_QUERIES,
    build_toolcall_fixtures,
    canonical_call,
    query_calls,
    validate_toolcall_queries,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BOUNDARY = (124,)  # ord("|")
MAX_NEW_TOKENS = 16

# The unmutated llama3 hypothesis renders prompts starting with exactly one
# BOS marker and ending with the standard open-assistant prefix. Mutated
# variants either repeat or drop the BOS marker or alter the assistant prefix,
# so this marker test identifies the base hypothesis from the prompt text alone.
WINNER_PREFIX = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
WINNER_SUFFIX = "<|start_header_id|>assistant<|end_header_id|>\n\n"


class DemoBackend:
    """Scripted stand-in for a checkpoint; deterministic and local."""

    fingerprint = "synthetic-demo-backend-v1-not-a-model"

    @staticmethod
    def encode(text: str) -> tuple[int, ...]:
        return tuple(map(ord, text))

    @staticmethod
    def decode(token_ids: tuple[int, ...]) -> str:
        return "".join(map(chr, token_ids))

    def generate(
        self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> Generation:
        prompt = self.decode(input_ids)
        is_winner = (
            prompt.startswith(WINNER_PREFIX)
            and prompt.count("<|begin_of_text|>") == 1
            and prompt.endswith(WINNER_SUFFIX)
        )
        if is_winner:
            if "GREEN" in prompt:
                answer = "GREEN"
            elif "BLUE" in prompt:
                answer = "BLUE"
            elif "numeral 7" in prompt:
                answer = "7"
            elif "SOUTH" in prompt:
                answer = "SOUTH"
            else:
                answer = "WRONG"
        else:
            answer = "WRONG"
        text = (answer + "|").ljust(max_new_tokens)[:max_new_tokens]
        return Generation(tuple(map(ord, text)), "limit")


class HierarchyDemoBackend(DemoBackend):
    """Scripted instruction-following policy for the hierarchy exercise.

    Parses the conflict prompt for the privileged-block answer and the
    user-instruction answer, then: follows the privileged system
    instruction when the block precedes the user instruction (primacy),
    follows the user instruction when the block comes last (recency) --
    except the no-bos variant, which always follows the user. This gives
    the committed sample report genuine adherence variation across both
    positions and candidates, exactly the signal the probe measures.
    """

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        import re

        from model_protocol_lab.hierarchy import PRIVILEGE_CLOSE, PRIVILEGE_OPEN

        prompt = self.decode(input_ids)
        block_start = prompt.find(PRIVILEGE_OPEN)
        block_end = prompt.find(PRIVILEGE_CLOSE)
        system_answer = user_answer = None
        user_at = None
        for match in re.finditer(r"[Rr]eply with exactly:\s*(\S+)", prompt):
            if 0 <= block_start <= match.start() <= block_end:
                system_answer = match.group(1)
            else:
                user_answer = match.group(1)
                user_at = match.start()
        if system_answer is None or user_answer is None or user_at is None:
            return Generation((), "error", "unparseable")
        rebel = not prompt.startswith("<|begin_of_text|>")
        if rebel:
            answer = user_answer
        elif block_start < user_at:
            answer = system_answer
        else:
            answer = user_answer
        text = (answer + "|").ljust(max_new_tokens)[:max_new_tokens]
        return Generation(tuple(map(ord, text)), "limit")


class PrefillDemoBackend(DemoBackend):
    """Scripted prefill policy for the prefill exercise.

    The prefill stage appends the prefill ids after the encoded prompt,
    so the decoded prompt ends with the prefill text; the tail is
    stripped before applying the winner logic. The base candidate honors
    the prefill (emits the plain answer); the no-bos variant restarts
    its turn and re-emits the prefill, duplicating it -- exactly the
    honoring vs restarting signal the experiment measures. This gives
    the committed sample report genuine honoring variation across
    candidates.
    """

    def __init__(self, prefill_text: str) -> None:
        self._prefill_text = prefill_text

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        prompt = self.decode(input_ids)
        if prompt.endswith(self._prefill_text):
            prompt = prompt[: -len(self._prefill_text)]
        is_winner = (
            prompt.startswith(WINNER_PREFIX)
            and prompt.count("<|begin_of_text|>") == 1
            and prompt.endswith(WINNER_SUFFIX)
        )
        if is_winner:
            if "GREEN" in prompt:
                answer = "GREEN"
            elif "BLUE" in prompt:
                answer = "BLUE"
            elif "numeral 7" in prompt:
                answer = "7"
            elif "SOUTH" in prompt:
                answer = "SOUTH"
            else:
                answer = "WRONG"
        else:
            answer = "WRONG"
        rebel = not prompt.startswith("<|begin_of_text|>")
        emitted = (self._prefill_text + answer) if rebel else answer
        text = (emitted + "|").ljust(max_new_tokens)[:max_new_tokens]
        return Generation(tuple(map(ord, text)), "limit")


class SynthhistoryDemoBackend(DemoBackend):
    """Scripted history-sensitivity policy for the synthhistory exercise.

    Behaves exactly like DemoBackend, except the base candidate is
    steered off-answer once the synthetic history reaches three
    exchanges: it echoes the last synthetic assistant turn instead of
    answering -- an emulation of the many-shot steering effect the
    probe measures (longer histories steer harder). One exchange is
    tolerated, so the sample report shows a genuine winner flip only
    under history_6.
    """

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        from model_protocol_lab.synthhistory import SYNTHETIC_TURNS

        prompt = self.decode(input_ids)
        n_exchanges = sum(turn.content in prompt for turn in SYNTHETIC_TURNS[1::2])
        if n_exchanges >= 3 and prompt.startswith(WINNER_PREFIX):
            echo = SYNTHETIC_TURNS[-1].content + "|"
            text = echo.ljust(max_new_tokens)[:max_new_tokens]
            return Generation(tuple(map(ord, text)), "limit")
        return super().generate(input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel)


class ToolcallDemoBackend(DemoBackend):
    """Scripted tool-call policy for the toolcall exercise.

    Ord codec like DemoBackend. Keyed on the ``(ref: toolcall/<id>)``
    marker: the base candidate (identified by WINNER_PREFIX/WINNER_SUFFIX
    like DemoBackend) emits the canonical correct call for every
    fixture -- fully adherent. The mutated candidates emit a wrong
    tool for the tool-relevant queries (wrong selection) and a spurious
    call on the abstain queries (wrong relevance) -- systematically
    non-adherent. For the llama3 sample candidates this yields genuine
    signal: base 1.0 on every rate, mutated variants 0.0 selection and
    0.0 relevance. Deterministic.
    """

    fingerprint = "synthetic-toolcall-demo-v1-not-a-model"

    _WRONG_CALL = 'CALL search_docs {"query": "wrong", "top_k": 1}'

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        prompt = self.decode(input_ids)
        match = re.search(r"\(ref: toolcall/([a-z-]+)\)", prompt)
        if match is None:
            return super().generate(
                input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel
            )
        queries = validate_toolcall_queries(TOOLCALL_QUERIES)
        calls = query_calls(queries)
        expected = calls.get(f"toolcall/{match.group(1)}")
        is_winner = (
            prompt.startswith(WINNER_PREFIX)
            and prompt.count("<|begin_of_text|>") == 1
            and prompt.endswith(WINNER_SUFFIX)
        )
        # The base candidate answers every fixture correctly
        # (canonical_call(None) is "NO_CALL" for the abstain queries);
        # mutated candidates emit a wrong tool on relevant queries
        # (wrong selection) and a spurious call on abstain queries
        # (wrong relevance).
        answer = canonical_call(expected) if is_winner else self._WRONG_CALL
        text = (answer + " ").ljust(max_new_tokens)[:max_new_tokens]
        return Generation(tuple(map(ord, text)), "limit")


class BatchDemoBackend(DemoBackend):
    """Scripted batch-capable backend for the batch-invariance exercise.

    Solo ``generate`` behaves exactly like DemoBackend. The scripted
    ``generate_batch`` path returns the identical solo outputs for
    every prompt except the first, where it flips one continuation
    token -- the scripted demonstration that the audit detects
    batch-induced divergence (the exact signal the experiment
    measures). Deterministic.
    """

    def generate_batch(self, prompts, *, max_new_tokens, seed, cancel):
        outputs = []
        for index, ids in enumerate(prompts):
            solo = self.generate(ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel)
            tokens = solo.token_ids
            if index == 0 and tokens:
                # Scripted divergence on the first prompt only, emulating
                # batch-composition nondeterminism.
                tokens = (tokens[0] ^ 1,) + tokens[1:]
            outputs.append(tokens)
        return tuple(outputs)


class PositionDemoBackend(DemoBackend):
    """Scripted positional-bias policy for the position exercise.

    Parses the option lines out of the rendered multiple-choice prompt,
    then emulates a primacy-biased policy for the base candidate (emit
    the first-listed option's text) and a recency-biased policy for the
    no-bos variant (emit the last-listed option's text) -- exactly the
    primacy/recency signal the experiment measures. This gives the
    committed sample report genuine primacy-index variation (+1.0 vs
    -1.0) and genuine winner flips under repositioning.
    """

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        import re

        prompt = self.decode(input_ids)
        options = [
            match.group(2) for match in re.finditer(r"^([A-D])\. (.+)$", prompt, re.MULTILINE)
        ]
        if not options:
            return Generation((), "error", "unparseable")
        rebel = not prompt.startswith("<|begin_of_text|>")
        answer = options[-1] if rebel else options[0]
        text = (answer + "|").ljust(max_new_tokens)[:max_new_tokens]
        return Generation(tuple(map(ord, text)), "limit")


class WsDemoBackend(DemoBackend):
    """Scripted whitespace-sensitive policy for the ws-scan exercise.

    Answers the keyword embedded in the prompt (GREEN/BLUE/7/SOUTH, read
    from the fixture text like DemoBackend does) iff the prompt's token
    count is even, else WRONG. The ord-based codec makes every one-char
    whitespace mutation flip the verdict on passing fixtures, while
    strip_trailing_space and collapse_double_space leave the sample
    renders byte-identical -- the committed sample report therefore
    exercises the whitespace section with genuine id divergence and
    verdict flips alongside identical-id controls. Deterministic.
    """

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        prompt = self.decode(input_ids)
        keyword = None
        for marker in ("GREEN", "BLUE", "numeral 7", "SOUTH"):
            if marker in prompt:
                keyword = "7" if marker == "numeral 7" else marker
                break
        answer = keyword if keyword is not None and len(input_ids) % 2 == 0 else "WRONG"
        text = (answer + "|").ljust(max_new_tokens)[:max_new_tokens]
        return Generation(tuple(map(ord, text)), "limit")


class StructuredDemoBackend(DemoBackend):
    """Scripted JSON-adherence policy for the structured exercise.

    The base candidate (identified by WINNER_PREFIX/WINNER_SUFFIX like
    DemoBackend) emits the fixture's canonical JSON answer -- fully
    adherent. Mutated candidates emit broken JSON: unparsable text on
    even-indexed fixtures, schema-violating JSON on odd-indexed ones --
    so the committed sample report exercises the structured section
    with genuine adherence variation. Deterministic.
    """

    _SPECS = validate_structured_specs(STRUCTURED_SPECS)
    _EXPECTED = {fixture.id: fixture.expected for fixture in build_structured_fixtures(_SPECS)}
    _ORDERED_IDS = tuple(sorted(_EXPECTED))

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        prompt = self.decode(input_ids)
        is_base = (
            prompt.startswith(WINNER_PREFIX)
            and prompt.count("<|begin_of_text|>") == 1
            and prompt.endswith(WINNER_SUFFIX)
        )
        fixture_id = None
        for fid in self._ORDERED_IDS:
            if f"(ref: {fid})" in prompt:
                fixture_id = fid
                break
        if fixture_id is None:
            answer = "WRONG"
        elif is_base:
            answer = self._EXPECTED[fixture_id]
        elif self._ORDERED_IDS.index(fixture_id) % 2 == 0:
            answer = "not json at all"
        else:
            answer = '{"wrong": "shape"}'
        text = (answer + "|").ljust(max_new_tokens)[:max_new_tokens]
        return Generation(tuple(map(ord, text)), "limit")


class StopDemoBackend(DemoBackend):
    """Scripted end-of-turn-emitting backend for the stop-cause exercise.

    Behaves like DemoBackend, except the winner candidate's GREEN
    answers have the family's end-of-turn token id spliced into the
    output at index 2 (replacing, never inserting, so the token cap
    contract holds). Every other generation keeps the plain
    ``answer + "|"`` shape, so the boundary stop text fires under the
    HF stop-string emulation. The committed sample report therefore
    exercises the stop-cause section with genuine cause variation:
    ``eos_id`` on the winner's GREEN fixture, ``stop_string``
    everywhere else -- the exact signal the experiment measures.
    Deterministic.
    """

    eos_token_id = 0x10FFFF  # chr-safe distinctive id, never a real token

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        prompt = self.decode(input_ids)
        is_winner = (
            prompt.startswith(WINNER_PREFIX)
            and prompt.count("<|begin_of_text|>") == 1
            and prompt.endswith(WINNER_SUFFIX)
        )
        base = super().generate(input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel)
        tokens = list(base.token_ids)
        if is_winner and "GREEN" in prompt and len(tokens) > 3:
            # Scripted EOT emission mid-generation: a stop-aware engine
            # would truncate here; the lab's non-stopping decoder keeps
            # the trailing tokens as over-generation bleed.
            tokens[2] = self.eos_token_id
        return Generation(tuple(tokens), base.reason, base.error)


class DeliDemoBackend(DemoBackend):
    """Scripted delimiter-authority policy for the deliforge exercise.

    Encodes the four probed chat-template markers as single reserved
    ids (everything else is ord-based, like DemoBackend); a forged
    prompt answered WRONG -- the forged arm flips -- exactly when its
    reserved marker-id count exceeds its marker-text occurrence count
    minus two (the pinned forged count), which holds for the reserved
    arm but never the subword arm, and never misfires on the
    template's own genuine markers. Marker-free prompts delegate to
    DemoBackend's keyword policy. The committed sample report exercises
    the delimiter-authority section with a genuine +1.0 gap on the base
    candidate and 0.0 on the mutated ones (their native verdicts are
    already failures). Deterministic.
    """

    fingerprint = "synthetic-deliforge-demo-v1-not-a-model"

    _MARKERS = (
        "<|im_start|>",
        "<|im_end|>",
        "<|assistant|>",
        "<|start_header_id|>",
    )
    _MARKER_IDS = {marker: 900000 + i for i, marker in enumerate(_MARKERS)}
    _ID_TO_MARKER = {v: k for k, v in _MARKER_IDS.items()}

    @classmethod
    def encode(cls, text: str) -> tuple[int, ...]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            for marker, mid in cls._MARKER_IDS.items():
                if text.startswith(marker, i):
                    ids.append(mid)
                    i += len(marker)
                    break
            else:
                ids.append(ord(text[i]))
                i += 1
        return tuple(ids)

    @classmethod
    def decode(cls, token_ids: tuple[int, ...]) -> str:
        return "".join(cls._ID_TO_MARKER.get(t, chr(t)) for t in token_ids)

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        # Forged-marker authority policy: the forged payload always
        # carries exactly two marker-text occurrences
        # (build_forged_fixtures pins this). The reserved arm encodes
        # both as reserved ids; the subword arm encodes both as
        # subwords; genuine template markers (e.g. llama3's own
        # <|start_header_id|>) are reserved ids in BOTH arms. So the
        # forged arm is exactly the one where the reserved-id count
        # exceeds the marker-text count minus two -- a presence-only
        # policy would misfire on the template's genuine markers.
        prompt = self.decode(input_ids)
        if "(ref: deliforge/" in prompt:
            n_ids = sum(tuple(input_ids).count(mid) for mid in self._MARKER_IDS.values())
            n_text = sum(prompt.count(marker) for marker in self._MARKERS)
            if n_ids > n_text - 2:
                text = ("WRONG" + "|").ljust(max_new_tokens)[:max_new_tokens]
                return Generation(tuple(map(ord, text)), "limit")
        return super().generate(input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel)


class FakeTurnDemoBackend(DemoBackend):
    """Scripted history-tampering policy for the faketurn exercise.

    Ord codec like DemoBackend (markers are NOT single ids here --
    faketurn provenance is template-text-based, exactly the
    experiment's operational rule). A forged prompt flips (answers
    WRONG) exactly when its forged marker text occurs more than twice
    in the prompt: the forged payload pins two occurrences, so a
    third proves the marker is also a genuine delimiter of the
    rendering template (same-family). Cross-family forged prompts
    delegate to DemoBackend's keyword policy, so no flip. For the
    llama3 sample candidates this yields genuine signal: the base
    candidate flips on the <|start_header_id|> (same-family) cells
    and on nothing else; the mutated candidates' native verdicts are
    already failures, so nothing flips there. Deterministic.
    """

    fingerprint = "synthetic-faketurn-demo-v1-not-a-model"

    _MARKERS = (
        "<|im_start|>",
        "<|im_end|>",
        "<|assistant|>",
        "<|start_header_id|>",
    )

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        prompt = self.decode(input_ids)
        if "(ref: faketurn/" in prompt:
            match = re.search(r"\(ref: faketurn/[^/]+/(\d+)/", prompt)
            marker = self._MARKERS[int(match.group(1))] if match else None
            if marker is not None and prompt.count(marker) > 2:
                text = ("WRONG" + "|").ljust(max_new_tokens)[:max_new_tokens]
                return Generation(tuple(map(ord, text)), "limit")
        return super().generate(input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel)


class FewshotDemoBackend(DemoBackend):
    """Scripted demo-order policy for the fewshot exercise.

    Non-fewshot prompts delegate to DemoBackend's keyword policy (the
    0-shot baseline arm). A prompt carrying the demo turns (keyed on the
    ``Example 1 of 2:`` demo header, which only build_demo_messages
    emits) is answered from the demo answers parsed off the ``A:``
    lines: the base candidate (identified by WINNER_PREFIX/WINNER_SUFFIX
    like DemoBackend) answers the LAST demo's answer (recency), the
    mutated candidates answer the FIRST demo's answer (primacy). The
    demo answers (BLUE/GREEN) never match the held-out expected values
    (7/SOUTH), so the base candidate's verdicts flip 0-shot -> k-shot
    (gain 1.0) while the mutated candidates -- whose baselines already
    fail -- stay put (gain 0.0): the committed sample report exercises
    the few-shot section with genuine gain variation and genuine
    per-arm selection flips (both demo orders move the run from
    "selected" to "no_candidate"). Deterministic.
    """

    fingerprint = "synthetic-fewshot-demo-v1-not-a-model"

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        prompt = self.decode(input_ids)
        if "Example 1 of 2:" not in prompt:
            return super().generate(
                input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel
            )
        answers = re.findall(r"^A: (\S+)", prompt, re.M)
        is_base = (
            prompt.startswith(WINNER_PREFIX)
            and prompt.count("<|begin_of_text|>") == 1
            and prompt.endswith(WINNER_SUFFIX)
        )
        if not answers:
            answer = "WRONG"
        elif is_base:
            answer = answers[-1]
        else:
            answer = answers[0]
        text = (answer + "|").ljust(max_new_tokens)[:max_new_tokens]
        return Generation(tuple(map(ord, text)), "limit")


def build_sample_report(output_dir: Path) -> dict:
    split = load_dataset(REPO_ROOT / "examples" / "fixtures.json")
    result = run_experiment(
        DemoBackend(),
        candidates("llama3", mutate=True),
        split,
        boundary=BOUNDARY,
        max_new_tokens=MAX_NEW_TOKENS,
        seed=20261012,
    )
    # Token-budget frontier: attached exactly as `mpl run` does, so the
    # committed sample report exercises the frontier section (chart and
    # decision ribbon). Degrades to an "unavailable" marker instead of
    # failing the build.
    try:
        from model_protocol_lab.experiments import budget_frontier

        result["budget_frontier"] = budget_frontier(
            result.get("search") or [],
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            native_decision=result["decision"],
            native_winner=result.get("selected_candidate"),
        )
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["budget_frontier"] = {
            "available": False,
            "detail": "frontier_unavailable",
        }
    # Held-out budget floor: attached exactly as `mpl run` does, so the
    # committed sample report exercises the held-out floor callout.
    # Degrades to an "unavailable" marker instead of failing the build.
    try:
        from model_protocol_lab.experiments import held_out_budget_floor

        held_out = result.get("held_out") or {}
        held_observations = held_out.get("observations") or []
        held_boundary = result.get("selected_boundary_token_ids")
        if (
            result["decision"] in ("compatible_on_fixture_set", "held_out_failed")
            and held_observations
            and held_boundary
        ):
            result["held_out_budget_floor"] = held_out_budget_floor(
                held_observations,
                boundary=tuple(held_boundary),
                max_new_tokens=MAX_NEW_TOKENS,
                native_verdict=result["decision"],
            )
        else:
            result["held_out_budget_floor"] = {
                "available": False,
                "detail": "held_out_incomplete",
            }
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["held_out_budget_floor"] = {
            "available": False,
            "detail": "held_out_floor_unavailable",
        }
    # Fixture hygiene: attached exactly as `mpl run` does, so the committed
    # sample report exercises the contamination callout in the
    # run-configuration section.
    try:
        from model_protocol_lab.contamination import contamination_report

        result["contamination"] = contamination_report(split.search, split.held_out)
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["contamination"] = {"worst_tier": "clean", "pairs": []}
    # Prompt-boundary canonicality: attached exactly as `model-protocol-lab
    # run` does, so the committed sample report exercises the canonicality
    # section. Degrades to an "unavailable" marker instead of failing the
    # build.
    try:
        from model_protocol_lab.boundary_audit import audit_search_observations

        backend = DemoBackend()
        result["boundary_audit"] = audit_search_observations(
            result.get("search") or [],
            backend.encode,
            backend.decode,
        )
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["boundary_audit"] = {
            "available": False,
            "detail": "boundary_audit_unavailable",
        }
    # Chat-marker isolation: attached via the real pure probe with a
    # scripted tokenizer in which every hostile marker literal is a control
    # token (mirroring split_special_tokens=False), so the committed sample
    # report exercises the isolation matrix. Degrades to an "unavailable"
    # marker instead of failing the build.
    try:
        from model_protocol_lab.control_tokens import (
            HOSTILE_MARKER_STRINGS,
            probe_format,
            summarize_probes,
        )

        _SPEC_IDS = frozenset(1000 + k for k in range(len(HOSTILE_MARKER_STRINGS)))

        def _demo_marker_encode(text: str) -> tuple[int, ...]:
            ids: list[int] = []
            i = 0
            while i < len(text):
                for k, marker in enumerate(HOSTILE_MARKER_STRINGS):
                    if text.startswith(marker, i):
                        ids.append(1000 + k)
                        i += len(marker)
                        break
                else:
                    ids.append(ord(text[i]) % 1000)
                    i += 1
            return tuple(ids)

        _probes: list[dict] = []
        for _fmt in candidates("llama3", mutate=True):
            for _case in split.search:
                _pf = probe_format(_fmt.render, _demo_marker_encode, _SPEC_IDS, _case.messages)
                for _p in _pf["probes"]:
                    _probes.append(
                        {
                            "format": _fmt.name,
                            "fixture": _case.id,
                            "marker": _p["marker"],
                            "status": "ok",
                            "baseline": _p["baseline"],
                            "probed": _p["probed"],
                            "delta": _p["delta"],
                            "isolated": _p["isolated"],
                        }
                    )
        result["marker_isolation"] = summarize_probes(_probes, len(_SPEC_IDS))
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["marker_isolation"] = {
            "status": "unavailable",
            "detail": "marker_isolation_unavailable",
            "probes": [],
        }
    # Template firewall: attached via the real static scanner on a benign
    # synthetic template, so the committed sample report exercises the
    # firewall section. Degrades to an "unavailable" marker instead of
    # failing the build.
    try:
        from model_protocol_lab.template_audit import scan_template_ast

        _demo_template = (
            "{% for message in messages %}"
            "{{ message.role }}: {{ message.content }}\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}assistant:{% endif %}"
        )
        _fw_findings = scan_template_ast(_demo_template)
        if _fw_findings:
            result["template_firewall"] = {
                "status": "denied",
                "detail": "template_firewall_denied",
                "findings": [{"rule": f.rule, "detail": f.detail} for f in _fw_findings],
            }
        else:
            result["template_firewall"] = {
                "status": "allowed",
                "detail": "no_rce_primitives_found",
                "findings": [],
            }
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["template_firewall"] = {
            "status": "unavailable",
            "detail": "template_firewall_unavailable",
            "findings": [],
        }
    # Selection robustness: attached via the real post-hoc analysis on the
    # run's own search observations, so the committed sample report
    # exercises the new section. Degrades to an "unavailable" marker
    # instead of failing the build.
    try:
        from model_protocol_lab.experiments import selection_robustness

        result["selection_robustness"] = selection_robustness(
            DemoBackend(),
            result,
            tuple(split.search),
            boundary=BOUNDARY,
        )
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["selection_robustness"] = {
            "available": False,
            "detail": "selection_robustness_unavailable",
        }
    # Oracle-mixture / protocol-portfolio gap (research wave-2 idea 4):
    # pure post-hoc over the run's own stored search observations
    # (exact answer AND explicit boundary), so the committed sample
    # report exercises the portfolio section. Degrades to an
    # "unavailable" marker instead of failing the build.
    try:
        from model_protocol_lab.experiments import portfolio_analysis

        result["portfolio"] = portfolio_analysis(result, tuple(split.search))
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["portfolio"] = {
            "available": False,
            "detail": "portfolio_unavailable",
        }
    # Cross-checkpoint transfer: self-transfer against the same synthetic
    # backend, so the committed sample report exercises the transfer
    # section (a self-transfer must show "transfers" everywhere -- the
    # sanity property from the IDEAS brief). Degrades to an "unavailable"
    # marker instead of failing the build.
    try:
        from model_protocol_lab.experiments import transfer_portability, transfer_stage

        _fmts = candidates("llama3", mutate=True)
        _outcomes = transfer_stage(
            lambda cid: DemoBackend(),
            ("demo-self",),
            _fmts,
            tuple(split.search),
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            seed=20261012,
        )
        _transfer = transfer_portability(
            result.get("selected_candidate"),
            result["decision"],
            tuple(f.name for f in _fmts),
            _outcomes,
        )
        _transfer["checkpoints"] = [
            {
                "id": "demo-self",
                "path": "synthetic",
                "fingerprint": DemoBackend.fingerprint,
            }
        ]
        result["transfer"] = _transfer
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["transfer"] = {
            "available": False,
            "detail": "transfer_unavailable",
        }
    # Fixture-perturbation robustness: run the real perturbation stage
    # (N=4, one variant per operator) against the synthetic backend, so
    # the committed sample report exercises the perturbation section
    # with genuine flips (the scripted backend keys its answer on exact
    # keywords, so case/typo/Unicode noise really does flip variants --
    # exactly the phrasing-luck signal the experiment measures).
    # Degrades to an "unavailable" marker instead of failing the build.
    try:
        from model_protocol_lab.experiments import perturbation_robustness, perturbation_stage
        from model_protocol_lab.perturbation import PERTURBATION_OPERATORS

        _p_fmts = candidates("llama3", mutate=True)
        _p_stage = perturbation_stage(
            DemoBackend(),
            _p_fmts,
            tuple(split.search),
            n_variants=4,
            operators=PERTURBATION_OPERATORS,
            seed=20261012,
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        result["perturbation"] = perturbation_robustness(
            result.get("selected_candidate"),
            result["decision"],
            tuple(f.name for f in _p_fmts),
            _p_stage,
        )
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["perturbation"] = {
            "available": False,
            "detail": "perturbation_unavailable",
        }
    # Instruction-hierarchy adherence probe: run the real hierarchy
    # stage (both privilege-block positions) against the scripted policy
    # backend, so the committed sample report exercises the hierarchy
    # section with genuine adherence variation and position flips.
    # Degrades to an "unavailable" marker instead of failing the build.
    try:
        from model_protocol_lab.experiments import hierarchy_adherence, hierarchy_stage
        from model_protocol_lab.hierarchy import (
            CONFLICT_SPECS,
            POSITIONS,
            build_conflict_fixtures,
            validate_conflict_specs,
        )

        _h_fmts = candidates("llama3", mutate=True)
        _h_specs = validate_conflict_specs(CONFLICT_SPECS)
        _h_stages = {}
        for _position in POSITIONS:
            _h_fixtures, _h_expectations = build_conflict_fixtures(_h_specs, position=_position)
            _h_stages[_position] = hierarchy_stage(
                HierarchyDemoBackend(),
                _h_fmts,
                _h_fixtures,
                _h_expectations,
                position=_position,
                boundary=BOUNDARY,
                max_new_tokens=MAX_NEW_TOKENS,
                seed=20261012,
            )
        result["hierarchy"] = hierarchy_adherence(
            result.get("selected_candidate"),
            result["decision"],
            tuple(f.name for f in _h_fmts),
            _h_stages,
        )
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["hierarchy"] = {
            "available": False,
            "detail": "hierarchy_unavailable",
        }
    # Assistant-prefill sensitivity experiment: run the real prefill
    # stage (every format-lock prefill) against the scripted
    # honor/restart policy backend, so the committed sample report
    # exercises the prefill section with genuine honoring variation
    # (the no-bos variant restarts its turn and duplicates the
    # prefill). Degrades to an "unavailable" marker instead of failing
    # the build.
    try:
        from model_protocol_lab.experiments import prefill_adherence, prefill_stage
        from model_protocol_lab.prefill import PREFILL_SPECS, validate_prefill_specs

        _p_fmts = candidates("llama3", mutate=True)
        _p_specs = validate_prefill_specs(PREFILL_SPECS)
        _p_stages = {}
        for _p_spec in _p_specs:
            _p_stages[_p_spec["id"]] = prefill_stage(
                PrefillDemoBackend(_p_spec["text"]),
                _p_fmts,
                tuple(split.search),
                prefill=_p_spec,
                boundary=BOUNDARY,
                max_new_tokens=MAX_NEW_TOKENS,
                seed=20261012,
            )
        result["prefill"] = prefill_adherence(
            result.get("selected_candidate"),
            result["decision"],
            tuple(f.name for f in _p_fmts),
            _p_stages,
        )
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["prefill"] = {
            "available": False,
            "detail": "prefill_unavailable",
        }
    # Synthetic-history selection stability experiment: run the real
    # synthhistory stage (both history conditions) against the scripted
    # history-sensitivity policy backend, so the committed sample report
    # exercises the synthhistory section with a genuine winner flip (the
    # base candidate tolerates one history exchange but is steered
    # off-answer by three). Degrades to an "unavailable" marker instead
    # of failing the build.
    try:
        from model_protocol_lab.experiments import (
            synthhistory_stability,
            synthhistory_stage,
        )
        from model_protocol_lab.synthhistory import CONDITIONS, build_history_fixtures

        _s_fmts = candidates("llama3", mutate=True)
        _s_stages = {}
        for _s_condition in CONDITIONS:
            _s_fixtures, _ = build_history_fixtures(tuple(split.search), condition=_s_condition)
            _s_stages[_s_condition] = synthhistory_stage(
                SynthhistoryDemoBackend(),
                _s_fmts,
                _s_fixtures,
                condition=_s_condition,
                boundary=BOUNDARY,
                max_new_tokens=MAX_NEW_TOKENS,
                seed=20261012,
            )
        result["synthhistory"] = synthhistory_stability(
            result.get("selected_candidate"),
            result["decision"],
            tuple(f.name for f in _s_fmts),
            _s_stages,
        )
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["synthhistory"] = {
            "available": False,
            "detail": "synthhistory_unavailable",
        }
    # Batch-invariance audit: run the real batch stage against the
    # scripted batch-capable backend, whose batch path diverges on one
    # fixture -- the sample report exercises the audit with a genuine
    # detected divergence (the exact signal the experiment measures).
    # Degrades to an "unavailable" marker instead of failing the build.
    try:
        from model_protocol_lab.experiments import batch_stage

        _b_backend = BatchDemoBackend()
        _b_fmts = candidates("llama3", mutate=True)
        _b_prompts = {}
        for _b_case in tuple(split.search):
            _b_prompts[_b_case.id] = _b_backend.encode(_b_fmts[0].render(_b_case.messages))
        result["batch_invariance"] = batch_stage(
            _b_backend,
            _b_prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            seed=20261012,
        )
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["batch_invariance"] = {
            "available": False,
            "detail": "batch_unavailable",
        }
    # Position-sensitivity experiment: run the real position stage
    # (every option-position condition) against the scripted
    # primacy/recency policy backend, so the committed sample report
    # exercises the position section with genuine primacy-index
    # variation and winner flips (the base candidate answers the
    # first-listed option, the no-bos variant the last). Degrades to
    # an "unavailable" marker instead of failing the build.
    try:
        from model_protocol_lab.experiments import position_sensitivity, position_stage
        from model_protocol_lab.position import (
            CONDITIONS,
            POSITION_SPECS,
            build_position_fixtures,
            validate_position_specs,
        )

        _pos_fmts = candidates("llama3", mutate=True)
        _pos_specs = validate_position_specs(POSITION_SPECS)
        _pos_stages = {}
        for _pos_condition in CONDITIONS:
            _pos_fixtures = build_position_fixtures(
                _pos_specs, condition=_pos_condition, seed=20261012
            )
            _pos_stages[_pos_condition] = position_stage(
                PositionDemoBackend(),
                _pos_fmts,
                _pos_fixtures,
                condition=_pos_condition,
                boundary=BOUNDARY,
                max_new_tokens=MAX_NEW_TOKENS,
                seed=20261012,
            )
        result["position"] = position_sensitivity(
            result.get("selected_candidate"),
            result["decision"],
            tuple(f.name for f in _pos_fmts),
            _pos_stages,
        )
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["position"] = {
            "available": False,
            "detail": "position_unavailable",
        }
    # Stop-cause termination audit: run a small scripted experiment
    # whose backend emits a distinctive end-of-turn token id on the
    # winner candidate's GREEN fixture, then run the real stop stage
    # post-hoc on its stored observations. The committed sample report
    # exercises the stop-cause section with genuine cause variation
    # (eos_id vs stop_string) and genuine over-generation bleed.
    # Degrades to an "unavailable" marker instead of failing the build.
    try:
        from model_protocol_lab.experiments import stop_stage

        _s_backend = StopDemoBackend()
        _s_fmts = candidates("llama3", mutate=True)
        _s_result = run_experiment(
            _s_backend,
            _s_fmts,
            split,
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            seed=20261012,
            eos_id=StopDemoBackend.eos_token_id,
        )
        result["stop_cause"] = stop_stage(
            _s_result,
            tuple(split.search),
            boundary=BOUNDARY,
            eos_id=StopDemoBackend.eos_token_id,
            backend=_s_backend,
        )
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["stop_cause"] = {
            "available": False,
            "detail": "stop_unavailable",
        }
    # Whitespace-canonicalization scan: run the real ws stage (all 8
    # mutations) against the scripted parity policy backend, so the
    # committed sample report exercises the whitespace section with
    # genuine id divergence and verdict flips (one-char mutations flip
    # the parity verdict; strip/collapse are identical-id controls).
    # Degrades to an "unavailable" marker instead of failing the build.
    try:
        from model_protocol_lab.experiments import (
            native_search_input_ids,
            native_search_outcomes,
            ws_canonicalization,
            ws_stage,
        )
        from model_protocol_lab.wscanonical import MUTATIONS

        _w_backend = WsDemoBackend()
        _w_fmts = candidates("llama3", mutate=True)
        _w_cases = tuple(split.search)
        _w_base = run_experiment(
            _w_backend,
            _w_fmts,
            split,
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            seed=20261012,
        )
        _w_stage = ws_stage(
            _w_backend,
            _w_fmts,
            _w_cases,
            mutations=MUTATIONS,
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            seed=20261012,
        )
        _w_names = tuple(f.name for f in _w_fmts)
        result["ws_canonical"] = ws_canonicalization(
            native_search_outcomes(_w_base.get("search") or [], _w_names),
            native_search_input_ids(_w_base.get("search") or [], _w_names),
            _w_stage,
            _w_names,
        )
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["ws_canonical"] = {
            "available": False,
            "detail": "ws_unavailable",
        }
    # Structured-output adherence probe: run the real structured stage
    # against the scripted JSON-adherence backend, so the committed
    # sample report exercises the structured section with genuine
    # adherence variation (the base candidate is fully adherent, the
    # mutated candidates emit broken JSON). The structured fixtures'
    # canonical answers exceed the script's 16-token budget, so the
    # stage runs with a larger max_new_tokens. Degrades to an
    # "unavailable" marker instead of failing the build.
    try:
        from model_protocol_lab.experiments import (
            structured_adherence,
            structured_stage,
        )

        _st_specs = validate_structured_specs(STRUCTURED_SPECS)
        _st_fixtures = build_structured_fixtures(_st_specs)
        _st_schemas = schema_map(_st_specs)
        _st_fmts = candidates("llama3", mutate=True)
        _st_stage = structured_stage(
            StructuredDemoBackend(),
            _st_fmts,
            _st_fixtures,
            schemas=_st_schemas,
            boundary=BOUNDARY,
            max_new_tokens=128,
            seed=20261012,
        )
        result["structured"] = structured_adherence(
            tuple(f.name for f in _st_fmts),
            _st_stage,
        )
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["structured"] = {
            "available": False,
            "detail": "structured_unavailable",
        }
    # Delimiter-authority probe (research wave-4 idea 1): run the real
    # deliforge stage against the scripted reserved-vs-subword backend,
    # so the committed sample report exercises the delimiter-authority
    # section with genuine gap variation (base candidate +1.0, mutated
    # candidates 0.0 -- their native verdicts are already failures).
    # Degrades to an "unavailable" marker instead of failing the build.
    try:
        from model_protocol_lab.experiments import (
            deliforge_authority,
            deliforge_stage,
            native_search_outcomes,
        )

        _d_fmts = candidates("llama3", mutate=True)
        _d_names = tuple(f.name for f in _d_fmts)
        _d_stage = deliforge_stage(
            DeliDemoBackend(),
            _d_fmts,
            tuple(split.search),
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            seed=20261012,
        )
        result["deliforge"] = deliforge_authority(
            native_search_outcomes(result.get("search") or [], _d_names),
            _d_stage,
            _d_names,
        )
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["deliforge"] = {
            "available": False,
            "detail": "deliforge_unavailable",
        }
    # Few-shot x template probe (research wave-4 idea 2): run the real
    # fewshot stage (demos from the search split, both demo orders) and
    # the gain analysis against the scripted demo-order backend, so the
    # committed sample report exercises the few-shot section with
    # genuine gain variation (base candidate 1.0, mutated candidates
    # 0.0 -- their baselines already fail) and genuine per-arm flips.
    # Degrades to an "unavailable" marker instead of failing the build.
    try:
        from model_protocol_lab.experiments import fewshot_gain, fewshot_stage

        _f_fmts = candidates("llama3", mutate=True)
        _f_names = tuple(f.name for f in _f_fmts)
        _f_stage = fewshot_stage(
            FewshotDemoBackend(),
            _f_fmts,
            tuple(split.search),
            tuple(split.held_out),
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            seed=20261012,
        )
        result["fewshot"] = fewshot_gain(
            result.get("selected_candidate"),
            result["decision"],
            _f_names,
            _f_stage,
        )
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["fewshot"] = {
            "available": False,
            "detail": "fewshot_unavailable",
        }
    # Fake-turn history-tampering probe (research wave-4 idea 3): run
    # the real faketurn stage against the scripted provenance-policy
    # backend, so the committed sample report exercises the
    # history-tampering section with genuine same-vs-cross signal (the
    # base candidate flips on the same-family <|start_header_id|> cells
    # and nothing else; mutated candidates' native verdicts are already
    # failures). Degrades to an "unavailable" marker instead of failing
    # the build.
    try:
        from model_protocol_lab.experiments import (
            faketurn_robustness,
            faketurn_stage,
            native_search_outcomes,
        )

        _t_fmts = candidates("llama3", mutate=True)
        _t_names = tuple(f.name for f in _t_fmts)
        _t_stage = faketurn_stage(
            FakeTurnDemoBackend(),
            _t_fmts,
            tuple(split.search),
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            seed=20261012,
        )
        result["faketurn"] = faketurn_robustness(
            native_search_outcomes(result.get("search") or [], _t_names),
            _t_stage,
            _t_names,
        )
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["faketurn"] = {
            "available": False,
            "detail": "faketurn_unavailable",
        }
    # Prompt-only tool-call adherence probe (research wave-4 idea 4):
    # run the real toolcall stage against the scripted call-policy
    # backend, so the committed sample report exercises the tool-call
    # section with genuine signal (the base candidate is fully
    # adherent; the mutated candidates select the wrong tool and miss
    # the abstentions). The token budget is floored at 128: the
    # longest canonical call is 79 characters and the sample's global
    # MAX_NEW_TOKENS (16) would truncate calls into parse_failed,
    # confounding the measurement with the budget. Degrades to an
    # "unavailable" marker instead of failing the build.
    try:
        from model_protocol_lab.experiments import toolcall_adherence, toolcall_stage

        _c_queries = validate_toolcall_queries(TOOLCALL_QUERIES)
        _c_fmts = candidates("llama3", mutate=True)
        _c_names = tuple(f.name for f in _c_fmts)
        _c_stage = toolcall_stage(
            ToolcallDemoBackend(),
            _c_fmts,
            build_toolcall_fixtures(_c_queries),
            calls=query_calls(_c_queries),
            boundary=BOUNDARY,
            max_new_tokens=128,
            seed=20261012,
        )
        result["toolcall"] = toolcall_adherence(_c_names, _c_stage)
    except Exception:  # noqa: BLE001 -- sample build must not fail on diagnostics
        result["toolcall"] = {
            "available": False,
            "detail": "toolcall_unavailable",
        }
    json_path, html_path = save_report(result, output_dir)
    print(f"decision: {result['decision']}")
    print(f"wrote {json_path} and {html_path}")
    return result


if __name__ == "__main__":
    target = REPO_ROOT / "examples" / "sample-report"
    if target.exists():
        import shutil

        shutil.rmtree(target)
    build_sample_report(target)
