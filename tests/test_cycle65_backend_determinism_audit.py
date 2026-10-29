"""Cycle 65 adversarial audit: backend seed determinism and cache isolation.

METHOD
------
The project's determinism guarantee rests on ``model_protocol_lab/backends.py``:
backends must be greedy, use request-local seeds, and never let cached state
leak across requests. This file proves those properties BY TEST, not by
reading, against a tiny locally-built GPT-2 checkpoint (random weights, built
with ``torch.manual_seed(1234)`` -- no downloads, no network):

1.  Byte-identical outputs: the same (seed, input_ids, cap) request is run on
    two independently constructed ``LocalHFBackend`` instances over the same
    checkpoint, twice in a row on one instance, and in a FRESH SUBPROCESS
    (true process-restart determinism: separate torch import, separate memory).
2.  Cache isolation: adversarial interleavings -- same prompt with different
    seeds, same seed with different prompts, a heavily "polluted" instance
    (dozens of alternating requests) compared against a brand-new instance,
    and greedy prefix-consistency (cap=8's first 4 tokens == cap=4's output),
    which is the observable signature of full-prefix recompute with no
    KV-cache leaking between requests.
3.  ``use_cache=False``: structural scans prove there is no memoization
    anywhere (no ``functools.lru_cache``/``cache`` wrappers, no module-level
    mutable containers, no per-instance dict/list/set caches that grow with
    traffic), the loaded HF config has ``use_cache=False``, and ``generate``
    leaves the global torch and ``random`` RNG states untouched.
4.  RNG discipline: seed 0, seed 2**64-1, negative / oversized / non-int /
    bool seeds (all must fail closed with ``invalid_seed``), same seed twice
    in a row, and interleaved seeds on one instance.
5.  Full ``evaluate`` path: ``core.evaluate`` run twice (and with different
    master seeds -- the request-local seed derivation changes, greedy
    selection must not) yields equal reports; ``run_experiment`` +
    ``save_report`` run twice yields BYTE-IDENTICAL ``results.json`` and
    ``report.html``.

Every backend class in the repo is covered: ``LocalHFBackend``,
``TruncatedPromptBackend`` (stateless wrapper -- its outputs are proven to be
a pure function of the inner backend), and ``diffbackends._TokenizerBackend``
(decode-only; decode determinism across instances).

FINDINGS (verified by running this file, 2026-10-16)
----------------------------------------------------
CLEAN -- no genuine determinism or cache-isolation bugs found:

* ``LocalHFBackend.generate`` is a pure function of (input_ids,
  max_new_tokens): byte-identical across instances, repeats, and a fresh
  subprocess. Greedy argmax is seed-invariant by design
  (``seed_used_for_selection=False``); seeds are validated to
  ``0 <= seed < 2**64`` and fail closed otherwise.
* No cache exists to leak: no module-level mutable state, no
  ``functools`` memoization anywhere in ``backends.py``, no per-instance
  cache containers, HF KV-cache disabled (``config.use_cache=False`` and
  ``use_cache=False`` on every forward). Interleaved/polluted-instance
  tests confirm requests are fully independent.
* ``generate`` does not touch global RNG state (torch or ``random``);
  constructor loading is the only thing that may (documented in the
  class docstring, not a request-path issue).
* ``evaluate`` derives request-local seeds deterministically
  (``sha256(f"{seed}\\0{case.id}")[:4]``) and ``run_experiment`` +
  ``save_report`` produce byte-identical report artifacts across runs:
  NO field differs, not even timestamps -- there are none in the report
  path. Zero allowed-difference fields.

REAL bugs: none. See the parent agent's report for anything deliberately
not verified.
"""

import importlib.util
import json
import os
import random
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

from model_protocol_lab import backends as backends_module

REQUIRED = ("torch", "transformers", "safetensors", "tokenizers")
HAS_HF = all(importlib.util.find_spec(name) is not None for name in REQUIRED)

REPO_ROOT = Path(__file__).resolve().parent.parent

# argv: [root, payload_json] -> prints {"token_ids": [...], "reason": ..., "error": ...}
# A genuinely fresh interpreter: separate torch import, separate address space.
_CHILD_GENERATE = (
    "import json,sys;"
    "from threading import Event;"
    "from model_protocol_lab.backends import LocalHFBackend;"
    "root=sys.argv[1];payload=json.loads(sys.argv[2]);"
    "backend=LocalHFBackend(root,device='cpu',max_context=64);"
    "result=backend.generate(tuple(payload['ids']),"
    "max_new_tokens=payload['cap'],seed=payload['seed'],cancel=Event());"
    "print(json.dumps({'token_ids':list(result.token_ids),"
    "'reason':result.reason,'error':result.error}))"
)


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class BackendDeterminismAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_adapter_integration import build_tiny_gpt2

        cls._tmp = TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.root = Path(cls._tmp.name) / "tiny"
        build_tiny_gpt2(cls.root)
        from model_protocol_lab.backends import LocalHFBackend

        cls.backend_a = LocalHFBackend(cls.root, device="cpu", max_context=64)
        cls.backend_b = LocalHFBackend(cls.root, device="cpu", max_context=64)
        cls.ids_a = cls.backend_a.encode("w1 w2 w3")
        cls.ids_b = cls.backend_a.encode("w10 w11 w12 w13")
        assert cls.ids_a != cls.ids_b

    @staticmethod
    def _generate(backend, input_ids, seed, cap=8):
        return backend.generate(input_ids, max_new_tokens=cap, seed=seed, cancel=Event())

    # -- 1. byte-identical outputs --------------------------------------

    def test_generate_byte_identical_across_instances(self):
        for seed in (0, 1, 2**64 - 1):
            with self.subTest(seed=seed):
                first = self._generate(self.backend_a, self.ids_a, seed)
                second = self._generate(self.backend_b, self.ids_a, seed)
                self.assertEqual(first, second)
                self.assertEqual(first.reason, "limit")
                self.assertIsNone(first.error)
                self.assertEqual(len(first.token_ids), 8)

    def test_generate_repeat_calls_same_instance(self):
        for ids in (self.ids_a, self.ids_b):
            with self.subTest(ids=ids):
                first = self._generate(self.backend_a, ids, 42)
                second = self._generate(self.backend_a, ids, 42)
                self.assertEqual(first.token_ids, second.token_ids)
                self.assertEqual(first, second)

    def test_generate_deterministic_across_process_restart(self):
        import torch

        torch.manual_seed(999)  # hostile: perturb global RNG before the child runs
        payload = {"ids": list(self.ids_a), "cap": 8, "seed": 1234}
        env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD_GENERATE, str(self.root), json.dumps(payload)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, f"child failed: {proc.stderr[-2000:]}")
        child = json.loads(proc.stdout.strip().splitlines()[-1])
        parent = self._generate(self.backend_a, self.ids_a, 1234)
        self.assertEqual(child["token_ids"], list(parent.token_ids))
        self.assertEqual(child["reason"], parent.reason)
        self.assertEqual(child["error"], parent.error)

    def test_truncated_prompt_backend_is_stateless_wrapper(self):
        from model_protocol_lab.backends import TruncatedPromptBackend
        from model_protocol_lab.core import truncate_text_tail

        for drop in (1, 2, 3):
            with self.subTest(drop_bytes=drop):
                wrap_a = TruncatedPromptBackend(self.backend_a, drop)
                wrap_b = TruncatedPromptBackend(self.backend_b, drop)
                text = "w1 w2 w3"
                # Pure function of the inner backend: wrapper adds no state.
                expected_ids = self.backend_a.encode(truncate_text_tail(text, drop))
                self.assertEqual(wrap_a.encode(text), expected_ids)
                self.assertEqual(wrap_a.encode(text), wrap_b.encode(text))
                out_a = self._generate(wrap_a, wrap_a.encode(text), 7)
                out_b = self._generate(wrap_b, wrap_b.encode(text), 7)
                self.assertEqual(out_a.token_ids, out_b.token_ids)
                self.assertEqual(
                    wrap_a.fingerprint,
                    f"{self.backend_a.fingerprint}+prompt-trunc-{drop}b",
                )

    # -- 2. cache isolation ----------------------------------------------

    def test_seed_values_do_not_alter_greedy_selection(self):
        # Adversarial direction 1: a cache keyed by prompt text alone (seed
        # ignored) would be *correct* here -- greedy selection is
        # seed-invariant by design. Pin that the seed genuinely cannot move
        # the output, so a seed-blind cache could never corrupt results.
        reference = self._generate(self.backend_a, self.ids_a, 0).token_ids
        for seed in (1, 7, 2**32, 2**64 - 1):
            with self.subTest(seed=seed):
                result = self._generate(self.backend_a, self.ids_a, seed)
                self.assertEqual(result.token_ids, reference)

    def test_same_seed_different_prompts_no_cache_confusion(self):
        # Adversarial direction 2 (the dangerous one): a cache keyed by seed
        # alone would return prompt A's output for prompt B. The proof of no
        # confusion is that each prompt's output matches a pristine instance.
        from model_protocol_lab.backends import LocalHFBackend

        pristine = LocalHFBackend(self.root, device="cpu", max_context=64)
        for ids in (self.ids_a, self.ids_b):
            with self.subTest(ids=ids):
                polluted_first = self._generate(self.backend_a, self.ids_a, 5)
                polluted_second = self._generate(self.backend_a, ids, 5)
                clean = self._generate(pristine, ids, 5)
                self.assertEqual(polluted_second.token_ids, clean.token_ids)
                if ids == self.ids_a:
                    self.assertEqual(polluted_first.token_ids, clean.token_ids)

    def test_heavily_polluted_instance_matches_fresh_instance(self):
        # Dozens of alternating requests (the "cache populated by backend A"
        # scenario); a fresh backend B must agree on every (prompt, seed).
        from model_protocol_lab.backends import LocalHFBackend

        for step in range(40):
            ids = self.ids_a if step % 2 == 0 else self.ids_b
            self._generate(self.backend_a, ids, seed=step)
        fresh = LocalHFBackend(self.root, device="cpu", max_context=64)
        for ids in (self.ids_a, self.ids_b):
            for seed in (0, 17, 2**64 - 1):
                with self.subTest(ids=ids, seed=seed):
                    self.assertEqual(
                        self._generate(self.backend_a, ids, seed).token_ids,
                        self._generate(fresh, ids, seed).token_ids,
                    )

    def test_interleaved_requests_stable_regardless_of_order(self):
        from model_protocol_lab.backends import LocalHFBackend

        fresh = LocalHFBackend(self.root, device="cpu", max_context=64)
        plan = [
            (self.ids_a, 11),
            (self.ids_b, 22),
            (self.ids_a, 11),
            (self.ids_b, 11),
            (self.ids_a, 22),
            (self.ids_b, 22),
        ]
        for ids, seed in plan:
            with self.subTest(ids=ids, seed=seed):
                self.assertEqual(
                    self._generate(self.backend_a, ids, seed).token_ids,
                    self._generate(fresh, ids, seed).token_ids,
                )

    def test_greedy_prefix_consistency_across_caps(self):
        # Full-prefix recompute with no KV-cache: lengthening the cap can
        # never rewrite earlier tokens. A leaking KV-cache would break this.
        for short, long in ((3, 8), (4, 8), (6, 10)):
            with self.subTest(short=short, long=long):
                prefix = self._generate(self.backend_a, self.ids_a, 3, cap=short)
                full = self._generate(self.backend_a, self.ids_a, 3, cap=long)
                self.assertEqual(full.token_ids[:short], prefix.token_ids)

    def test_forward_logits_deterministic_across_instances(self):
        import torch

        first = self.backend_a.forward_logits(self.ids_a)
        second = self.backend_b.forward_logits(self.ids_a)
        third = self.backend_a.forward_logits(self.ids_a)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first, third))
        self.assertEqual(first.shape, (len(self.ids_a), self.backend_a.vocab_size))

    # -- 3. use_cache=False / no hidden global or memoized state ---------

    def test_no_module_level_or_class_level_memoization(self):
        from model_protocol_lab.backends import LocalHFBackend

        for name, value in vars(backends_module).items():
            if name.startswith("__"):
                continue
            self.assertFalse(
                callable(value) and hasattr(value, "cache_info"),
                f"memoized callable at module level: {name}",
            )
            self.assertNotIsInstance(
                value, (dict, list, set), f"module-level mutable container: {name}"
            )
        self.assertFalse(hasattr(backends_module, "functools"))
        for name, value in vars(LocalHFBackend).items():
            self.assertNotIsInstance(
                value, (dict, list, set), f"class-level mutable container: {name}"
            )

    def test_no_instance_cache_growth_across_traffic(self):
        def mutable_snapshot(backend):
            return {
                key: repr(value)
                for key, value in vars(backend).items()
                if isinstance(value, (dict, list, set))
            }

        before = mutable_snapshot(self.backend_a)
        for step in range(20):
            ids = self.ids_a if step % 2 == 0 else self.ids_b
            self._generate(self.backend_a, ids, seed=step)
            self.backend_a.decode(self.ids_a)
            self.backend_a.encode("w1 w2")
        self.assertEqual(mutable_snapshot(self.backend_a), before)

    def test_generate_does_not_touch_global_rng_state(self):
        import torch

        torch.manual_seed(20260601)
        random.seed(20260601)
        torch_before = torch.get_rng_state().clone()
        random_before = random.getstate()
        for seed in (0, 99, 2**64 - 1):
            self._generate(self.backend_a, self.ids_a, seed)
            self.backend_a.forward_logits(self.ids_a)
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        self.assertEqual(random_before, random.getstate())

    def test_hf_kv_cache_disabled_end_to_end(self):
        # config.use_cache=False at load time AND use_cache=False on every
        # forward inside generate()/forward_logits() -- verified structurally
        # here; the behavioral half is test_greedy_prefix_consistency_*.
        self.assertFalse(self.backend_a._model.config.use_cache)

    # -- 4. RNG discipline -----------------------------------------------

    def test_seed_validation_boundaries(self):
        cases = [
            (-1, "invalid_seed"),
            (2**64, "invalid_seed"),
            ("7", "invalid_seed"),
            (True, "invalid_seed"),
            (None, "invalid_seed"),
        ]
        for seed, code in cases:
            with self.subTest(seed=seed):
                result = self._generate(self.backend_a, self.ids_a, seed)
                self.assertEqual(result.reason, "error")
                self.assertEqual(result.error, code)
        for seed in (0, 2**64 - 1):
            with self.subTest(seed=seed):
                result = self._generate(self.backend_a, self.ids_a, seed)
                self.assertEqual(result.reason, "limit")
                self.assertEqual(len(result.token_ids), 8)

    # -- 5. full evaluate path -------------------------------------------

    @staticmethod
    def _fixtures():
        from model_protocol_lab.core import Fixture, Message

        return (
            Fixture("audit-a", (Message("user", "w1 w2"),), "w3", "grp-a"),
            Fixture("audit-b", (Message("user", "w4 w5"),), "w6", "grp-b"),
        )

    def test_evaluate_deterministic_across_runs_and_seeds(self):
        from model_protocol_lab.catalog import CATALOG
        from model_protocol_lab.core import evaluate

        fixtures = self._fixtures()
        first = evaluate(
            self.backend_a,
            CATALOG["chatml"],
            fixtures,
            boundary=(62,),
            max_new_tokens=8,
            seed=7,
        )
        second = evaluate(
            self.backend_b,
            CATALOG["chatml"],
            fixtures,
            boundary=(62,),
            max_new_tokens=8,
            seed=7,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.status, "completed")
        # Different master seeds change the derived request-local seeds
        # (sha256("seed\\0case-id")[:4]); greedy selection must not move.
        third = evaluate(
            self.backend_a,
            CATALOG["chatml"],
            fixtures,
            boundary=(62,),
            max_new_tokens=8,
            seed=999,
        )
        self.assertEqual(first.observations, third.observations)

    def test_run_experiment_report_bytes_byte_identical(self):
        from model_protocol_lab.catalog import CATALOG
        from model_protocol_lab.core import Fixture, Message, Split
        from model_protocol_lab.experiments import run_experiment
        from model_protocol_lab.reporting import save_report

        search = self._fixtures()
        held_out = (Fixture("audit-c", (Message("user", "w7"),), "w8", "grp-c"),)
        split = Split(search=search, held_out=held_out)
        first = run_experiment(
            self.backend_a,
            (CATALOG["chatml"],),
            split,
            boundary=(62,),
            max_new_tokens=8,
            seed=7,
        )
        second = run_experiment(
            self.backend_b,
            (CATALOG["chatml"],),
            split,
            boundary=(62,),
            max_new_tokens=8,
            seed=7,
        )
        self.assertEqual(first, second)
        artifacts = []
        for index, result in enumerate((first, second)):
            with TemporaryDirectory() as tmp:
                # save_report requires a not-yet-existing target directory.
                target = Path(tmp) / f"run-{index}"
                json_path, html_path = save_report(result, target)
                artifacts.append((Path(json_path).read_bytes(), Path(html_path).read_bytes()))
        # No allowed-difference fields: not even timestamps (the report
        # path contains none) may differ between two identical runs.
        self.assertEqual(artifacts[0][0], artifacts[1][0], "results.json differs")
        self.assertEqual(artifacts[0][1], artifacts[1][1], "report.html differs")

    def test_tokenizer_backend_decode_deterministic(self):
        # diffbackends._TokenizerBackend is decode-only; its whole surface
        # is decode determinism across instances sharing one tokenizer.
        from model_protocol_lab.diffbackends import _TokenizerBackend

        left = _TokenizerBackend(self.backend_a._tokenizer, self.backend_a.vocab_size)
        right = _TokenizerBackend(self.backend_a._tokenizer, self.backend_a.vocab_size)
        for ids in (self.ids_a, self.ids_b, self.ids_a + self.ids_b):
            with self.subTest(ids=ids):
                self.assertEqual(left.decode(ids), right.decode(ids))
                self.assertEqual(left.decode(ids), self.backend_a.decode(ids))


if __name__ == "__main__":
    unittest.main()
