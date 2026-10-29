"""Cycle 60: test-harness concurrency robustness.

Background: the full suite flaked intermittently (1-3 failures per run in a
3-run verification sequence). Reproduced deterministically by running two
full suites concurrently:

* ``test_cycle37_diffbackends_verify.py::SensitivityTests::test_mutants_are_caught``
  -> ``FileExistsError`` on the FIXED scratch dir
  ``/tmp/cycle37_diffbackends_mutants`` (both processes staged into it).
* ``test_cycle33_reporting_verify.py::TestMutantSensitivity::test_mutants_are_caught``
  -> ``ModuleNotFoundError: 'm12_overhead_stat'`` (one process's
  ``setUpClass`` rmtree deleted the other's in-use ``/tmp/cycle33_mutants``
  packages mid-import).
* ``test_cycle35_backends_verify.py`` 128 MiB fixture writes ->
  ``OSError: ENOSPC`` on the 512 MiB /tmp tmpfs under load.

Fixes pinned by this file:

1. All mutant scratch roots are PID-scoped (cycle19, cycle20, cycle32,
   cycle33, cycle37, cycle42), so concurrent pytest processes stage into
   disjoint directories.
2. The oversized-asset fixture is a sparse file: identical byte count on
   read (holes read as zeros), ~zero blocks on disk.
3. The exactly-at-cap test skips honestly when <256 MiB scratch space is
   free instead of dying with ENOSPC (which says nothing about the code).

Product code is untouched; every failure observed was in test scaffolding.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestMutantRootsPidScoped(unittest.TestCase):
    """Every mutant staging root must contain the current PID."""

    def test_cycle32(self):
        """Test cycle32."""
        from tests.test_cycle32_core_verify import MUTANT_DIR

        self.assertIn(str(os.getpid()), str(MUTANT_DIR))

    def test_cycle42(self):
        """Test cycle42."""
        from tests.test_cycle42_core_helpers_verify import MUTANT_DIR

        self.assertIn(str(os.getpid()), str(MUTANT_DIR))

    def test_cycle33(self):
        """Test cycle33."""
        from tests.test_cycle33_reporting_verify import _MUTANT_ROOT

        self.assertIn(str(os.getpid()), str(_MUTANT_ROOT))

    def test_cycle37(self):
        """Test cycle37."""
        from tests.test_cycle37_diffbackends_verify import _SCRATCH_ROOT

        self.assertIn(str(os.getpid()), str(_SCRATCH_ROOT))

    def test_cycle19(self):
        """Test cycle19."""
        from tests.test_cycle19_failure_taxonomy_verify import _MUTANT_ROOT

        self.assertIn(str(os.getpid()), str(_MUTANT_ROOT))

    def test_cycle20_staged_file_is_pid_scoped(self):
        """Test cycle20 staged file is pid scoped."""
        from tests.test_cycle20_evaluate_verify import load_mutant

        case = unittest.TestCase()
        module = load_mutant(
            case,
            "control",
            "boundary_found=cut is not None,",
            "boundary_found=cut is not None,",
        )
        try:
            self.assertTrue(hasattr(module, "evaluate"))
            staged = list(Path(tempfile.gettempdir()).glob(f"cycle20_mut_{os.getpid()}_control.py"))
            self.assertEqual(len(staged), 1)
        finally:
            case.doCleanups()
        self.assertFalse(
            Path(tempfile.gettempdir(), f"cycle20_mut_{os.getpid()}_control.py").exists(),
            "load_mutant cleanup must remove the staged file",
        )


class TestConcurrentMutantStaging(unittest.TestCase):
    """Two processes running the mutant-sensitivity battery concurrently must
    both succeed. Before PID-scoping this raced on the fixed scratch dir
    (FileExistsError / half-written imports).
    """

    def test_two_parallel_sensitivity_runs_both_pass(self):
        """Test two parallel sensitivity runs both pass."""
        script = (
            "import sys, unittest; "
            f"sys.path.insert(0, {str(REPO_ROOT)!r}); "
            "from tests.test_cycle32_core_verify import TestMutantSensitivity; "
            "suite = unittest.TestLoader().loadTestsFromName("
            "'test_all_mutants_caught', TestMutantSensitivity); "
            "result = unittest.TextTestRunner(verbosity=0).run(suite); "
            "sys.exit(0 if result.wasSuccessful() else 1)"
        )
        procs = [
            subprocess.Popen(  # noqa: S603 -- fixed argv, no shell, local python
                [sys.executable, "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(2)
        ]
        codes = [p.wait(timeout=600) for p in procs]
        for i, (proc, code) in enumerate(zip(procs, codes, strict=True)):
            _, err = proc.communicate()
            self.assertEqual(code, 0, f"parallel worker {i} failed:\n{err[-3000:]}")


class TestOversizedFixtureRobustness(unittest.TestCase):
    """The 128 MiB cap tests must not require 128 MiB of free disk."""

    def test_sparse_oversized_file_rejected(self):
        """Test sparse oversized file rejected."""
        from model_protocol_lab.backends import _read_json

        with tempfile.TemporaryDirectory(prefix="cycle60-") as tmp:
            target = Path(tmp) / "big.json"
            with target.open("wb") as handle:
                handle.seek(128 * 1024 * 1024)
                handle.write(b"\x00")
            self.assertEqual(target.stat().st_size, 128 * 1024 * 1024 + 1)
            # Sparse: the byte count is real, the disk cost is ~nothing.
            self.assertLess(target.stat().st_blocks * 512, 1024 * 1024)
            with self.assertRaises(ValueError) as ctx:
                _read_json(target)
            self.assertEqual(str(ctx.exception), "oversized_asset_json")

    def test_exactly_at_cap_skips_when_disk_full(self):
        """Test exactly at cap skips when disk full."""
        from tests.test_cycle35_backends_verify import TestReadJson

        case = TestReadJson("test_exactly_at_cap_parses")
        case.setUp()
        try:
            fake_usage = SimpleNamespace(total=10**9, used=10**9, free=0)
            with (
                mock.patch.object(shutil, "disk_usage", return_value=fake_usage),
                self.assertRaises(unittest.SkipTest),
            ):
                TestReadJson.test_exactly_at_cap_parses(case)
        finally:
            case.tearDown()


if __name__ == "__main__":
    unittest.main()
