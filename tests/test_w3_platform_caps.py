"""W3-2: platform_caps containment primitives (RR-08 typing half).

The migration contract: doctor's render cap/watchdog and the MCP
sweep deadline keep their observable Linux behaviour on top of the
single typed implementation in model_protocol_lab.platform_caps, and
the win32 degradation (unarmed, unbounded, no signal state touched)
is a designed outcome, not an accident. Expected values come from
the pre-migration semantics documented in doctor.py / mcp_server.py
and pinned by cycles 116/118/132/165/227; these tests pin the new
module's own surface (armed flags, restore behaviour, fault
suppression, the restore-retry call count) directly.
"""

from __future__ import annotations

import signal
import stat
import sys
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from model_protocol_lab import platform_caps


class MakeFifoTests(unittest.TestCase):
    """platform_caps.make_fifo / FIFO_SUPPORTED on this host."""

    def test_fifo_supported_and_created_on_linux(self):
        """Test fifo supported and created on linux."""
        if sys.platform == "win32":
            self.skipTest("POSIX host expected for the positive arm")
        self.assertTrue(platform_caps.FIFO_SUPPORTED)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.fifo"
            self.assertTrue(platform_caps.make_fifo(path))
            self.assertTrue(stat.S_ISFIFO(path.stat().st_mode))

    def test_make_fifo_errors_propagate(self):
        """Test make fifo errors propagate."""
        if sys.platform == "win32":
            self.skipTest("POSIX host expected for the positive arm")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.fifo"
            self.assertTrue(platform_caps.make_fifo(path))
            with self.assertRaises(FileExistsError):
                platform_caps.make_fifo(path)


class AlarmDeadlineTests(unittest.TestCase):
    """platform_caps.alarm_deadline arming, expiry and teardown."""

    def test_unarmed_cases_touch_no_signal_state(self):
        """Test unarmed cases touch no signal state."""
        if not platform_caps.alarm_available():
            self.skipTest("SIGALRM host expected")
        before_handler = signal.getsignal(signal.SIGALRM)
        before_timer = signal.getitimer(signal.ITIMER_REAL)
        for seconds in (0.0, -1.0, float("inf"), float("nan")):
            with platform_caps.alarm_deadline(seconds, lambda: None) as armed:
                self.assertFalse(armed, f"seconds={seconds}")
        self.assertEqual(signal.getsignal(signal.SIGALRM), before_handler)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), before_timer)

    def test_unarmed_from_worker_thread(self):
        """Test unarmed from worker thread."""
        outcome: dict[str, bool] = {}

        def run() -> None:
            """Run."""
            with platform_caps.alarm_deadline(60.0, lambda: None) as armed:
                outcome["armed"] = armed

        thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(outcome, {"armed": False})

    def test_armed_then_host_timer_rearmed_with_full_delay(self):
        """Test armed then host timer rearmed with full delay."""
        if not platform_caps.alarm_available():
            self.skipTest("SIGALRM host expected")
        fired = threading.Event()
        old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, lambda signum, frame: fired.set())
        signal.setitimer(signal.ITIMER_REAL, 45.0)
        try:
            with platform_caps.alarm_deadline(60.0, lambda: None) as armed:
                self.assertTrue(armed)
            remaining, _interval = signal.getitimer(signal.ITIMER_REAL)
            # Full original delay re-armed, not the remaining time:
            # teardown restores 45s, of which only moments have passed.
            self.assertGreater(remaining, 30.0)
            self.assertFalse(fired.is_set())
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

    def test_expiry_invokes_on_expire(self):
        """Test expiry invokes on expire."""
        if not platform_caps.alarm_available():
            self.skipTest("SIGALRM host expected")
        fired = threading.Event()
        with platform_caps.alarm_deadline(0.05, fired.set) as armed:
            self.assertTrue(armed)
            self.assertTrue(fired.wait(timeout=5.0))
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))

    def test_teardown_faults_do_not_escape(self):
        """Test teardown faults do not escape."""
        if not platform_caps.alarm_available():
            self.skipTest("SIGALRM host expected")
        real_signal = platform_caps.signal

        class FaultyMask:
            """Signal stand-in whose sigmask calls fail."""

            def __getattr__(self, name):
                if name == "pthread_sigmask":
                    raise AttributeError("no pthread_sigmask")
                return getattr(real_signal, name)

        with (
            mock.patch.object(platform_caps, "signal", FaultyMask()),
            platform_caps.alarm_deadline(60.0, lambda: None) as armed,
        ):
            self.assertTrue(armed)
        # Real state is consistent afterwards (the host handler/timer
        # were restored by the non-mask steps).
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))


class AddressSpaceCapTests(unittest.TestCase):
    """platform_caps.address_space_cap install/restore paths."""

    def test_installs_and_restores(self):
        """Test installs and restores."""
        if sys.platform == "win32":
            self.skipTest("POSIX host expected")
        import resource

        old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
        if old_soft != resource.RLIM_INFINITY:
            self.skipTest("ambient soft limit is finite; install path not exercisable")
        with platform_caps.address_space_cap(256 * 1024 * 1024) as installed:
            self.assertTrue(installed)
            capped_soft, capped_hard = resource.getrlimit(resource.RLIMIT_AS)
            self.assertNotEqual(capped_soft, resource.RLIM_INFINITY)
            self.assertEqual(capped_hard, old_hard)
        self.assertEqual(resource.getrlimit(resource.RLIMIT_AS), (old_soft, old_hard))

    def test_finite_ambient_limit_is_respected(self):
        """Test finite ambient limit is respected."""
        if sys.platform == "win32":
            self.skipTest("POSIX host expected")
        import resource

        old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
        if old_soft != resource.RLIM_INFINITY:
            self.skipTest("ambient soft limit already finite")
        # A finite ambient limit must be respected, never loosened --
        # exercised in a child so this process keeps its own limits.
        import subprocess

        script = (
            "import resource\n"
            "from model_protocol_lab import platform_caps\n"
            "soft, hard = resource.getrlimit(resource.RLIMIT_AS)\n"
            "resource.setrlimit(resource.RLIMIT_AS, (soft if soft != -1 else 2**31, hard))\n"
            "before = resource.getrlimit(resource.RLIMIT_AS)\n"
            "with platform_caps.address_space_cap(1024) as installed:\n"
            "    print('installed', installed)\n"
            "    print('inside', resource.getrlimit(resource.RLIMIT_AS) == before)\n"
            "print('after', resource.getrlimit(resource.RLIMIT_AS) == before)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-1000:])
        self.assertIn("installed False", proc.stdout)
        self.assertIn("inside True", proc.stdout)
        self.assertIn("after True", proc.stdout)

    def test_getrlimit_failure_yields_uninstalled(self):
        """Test getrlimit failure yields uninstalled."""
        if sys.platform == "win32":
            self.skipTest("POSIX host expected")
        import resource

        with (
            mock.patch.object(resource, "getrlimit", side_effect=OSError("nope")),
            platform_caps.address_space_cap(1024) as installed,
        ):
            self.assertFalse(installed)

    def test_install_failure_yields_uninstalled(self):
        """Test install failure yields uninstalled."""
        if sys.platform == "win32":
            self.skipTest("POSIX host expected")
        import resource

        old_soft, _ = resource.getrlimit(resource.RLIMIT_AS)
        if old_soft != resource.RLIM_INFINITY:
            self.skipTest("ambient soft limit is finite; install path not exercisable")
        with (
            mock.patch.object(resource, "setrlimit", side_effect=ValueError("nope")),
            platform_caps.address_space_cap(1024) as installed,
        ):
            self.assertFalse(installed)

    def test_restore_reraise_retries_once_then_raises(self):
        """Test restore reraise retries once then raises."""
        if sys.platform == "win32":
            self.skipTest("POSIX host expected")
        import resource

        old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
        if old_soft != resource.RLIM_INFINITY:
            self.skipTest("ambient soft limit is finite; install path not exercisable")

        class BudgetBlow(ValueError):
            """Budget blow."""

        real_setrlimit = resource.setrlimit
        calls: list[tuple[int, tuple[int, int]]] = []

        def flaky(which, limits):
            """Flaky."""
            calls.append((which, limits))
            if len(calls) == 2:
                raise BudgetBlow("alarm landed mid-restore")
            return real_setrlimit(which, limits)

        with mock.patch.object(resource, "setrlimit", side_effect=flaky):
            try:
                with platform_caps.address_space_cap(
                    256 * 1024 * 1024,
                    reraise_on_restore=(BudgetBlow,),
                ):
                    self.fail("the mid-restore budget exception must re-raise")
            except BudgetBlow:
                pass
        # install, failed restore, successful retry -- the cycle-132
        # three-call shape, at the capability level.
        self.assertEqual(len(calls), 3)
        self.assertEqual(resource.getrlimit(resource.RLIMIT_AS), (old_soft, old_hard))

    def test_fifo_unsupported_short_circuit_is_honest(self):
        """Test fifo unsupported short circuit is honest."""
        # On this POSIX host FIFO_SUPPORTED is True, so the honest
        # platform_unsupported result is exercised by patching the
        # constant -- the same shape a Windows run produces.
        from model_protocol_lab import fixturefuzz

        cases = [c for c in fixturefuzz.generate_cases(0) if c.file_kind == "fifo"]
        self.assertTrue(cases, "corpus must contain fifo cases")
        with (
            TemporaryDirectory() as tmp,
            mock.patch.object(platform_caps, "FIFO_SUPPORTED", False),
        ):
            for case in cases:
                result = fixturefuzz.evaluate_case(case, Path(tmp))
                self.assertEqual(result.kind, "platform_unsupported")
                self.assertEqual(result.observed, "reject")
                self.assertTrue(result.conform)


class AddressSpaceCapDomainTests(unittest.TestCase):
    """Wave-11 BH11-3/BH11-4: address_space_cap's domain totality.

    The docstring promises every not-installed path yields False
    with no exception and no state touched. Wave 11 executed two
    escapes: an unvalidated headroom_bytes (2**63 left as a raw
    OverflowError from the setrlimit C conversion; a hugely
    negative value installed an undefined wrapped limit and yielded
    True), and a /proc VmSize parse that caught only OSError (a
    present-but-malformed line escaped as IndexError/ValueError).
    """

    def test_headroom_out_of_domain_yields_uninstalled(self):
        """Test headroom out of domain yields uninstalled."""
        if sys.platform == "win32":
            self.skipTest("POSIX host expected")
        import resource

        old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
        if old_soft != resource.RLIM_INFINITY:
            self.skipTest("ambient soft limit is finite; install path not exercisable")
        for headroom in (-1, -(2**63), 2**63, 2**64, 2**100):
            with self.subTest(headroom=headroom):
                with platform_caps.address_space_cap(headroom) as installed:
                    self.assertFalse(installed)
                self.assertEqual(resource.getrlimit(resource.RLIMIT_AS), (old_soft, old_hard))

    def test_headroom_boundary_at_c_long_max(self):
        """Test headroom boundary at c long max."""
        if sys.platform == "win32":
            self.skipTest("POSIX host expected")
        import io
        import resource

        # A finite RLIMIT_AS value must convert to a C long for
        # setrlimit: sys.maxsize is the largest installable limit
        # and sys.maxsize + 1 is out of domain (probed on this host
        # class: 2**63 - 1 accepted, 2**63 OverflowError). Serve a
        # fixed VmSize (1024 kB) so the boundary is exact, and
        # record setrlimit calls instead of making them.
        vmsize = 1024 * 1024
        real_open = open

        def fake_open(path, *args, **kwargs):
            """Fake open."""
            if str(path) == "/proc/self/status":
                return io.StringIO("VmSize:\t1024 kB\n")
            return real_open(path, *args, **kwargs)

        calls: list[tuple[int, tuple[int, int]]] = []

        def recording_setrlimit(which, limits):
            """Recording setrlimit."""
            calls.append((which, limits))

        with (
            mock.patch("builtins.open", side_effect=fake_open),
            mock.patch.object(
                resource,
                "getrlimit",
                return_value=(resource.RLIM_INFINITY, resource.RLIM_INFINITY),
            ),
            mock.patch.object(resource, "setrlimit", side_effect=recording_setrlimit),
        ):
            with platform_caps.address_space_cap(sys.maxsize - vmsize) as installed:
                self.assertTrue(installed)
            self.assertEqual(
                calls[0],
                (resource.RLIMIT_AS, (sys.maxsize, resource.RLIM_INFINITY)),
            )
            calls.clear()
            with platform_caps.address_space_cap(sys.maxsize - vmsize + 1) as installed:
                self.assertFalse(installed)
            self.assertEqual(calls, [], "out-of-domain headroom must not reach setrlimit")

    def test_proc_status_malformed_yields_uninstalled(self):
        """Test proc status malformed yields uninstalled."""
        if sys.platform == "win32":
            self.skipTest("POSIX host expected")
        import io

        real_open = open

        # The shipped unreadable test covers only the OSError arm;
        # a present-but-unparsable VmSize line is the same
        # documented not-installed path (no readable VmSize).
        for content in (
            "VmSize:\n",
            "VmSize: notanumber kB\n",
            "VmSize: 12x kB\n",
            "Name:\tno vmsize line at all\n",
        ):
            with self.subTest(content=content):

                def fake_open(path, *args, _content=content, **kwargs):
                    """Fake open."""
                    if str(path) == "/proc/self/status":
                        return io.StringIO(_content)
                    return real_open(path, *args, **kwargs)

                with (
                    mock.patch("builtins.open", side_effect=fake_open),
                    platform_caps.address_space_cap(1024) as installed,
                ):
                    self.assertFalse(installed)


class PlatformSimulationTests(unittest.TestCase):
    """Simulated-platform and fault-injection paths of platform_caps.

    The win32 branches cannot execute on this Linux host unpatched;
    patching sys.platform is the standard simulation (the functions
    read it at call time), and every assertion mirrors what a real
    Windows run must produce: unarmed, uninstalled, no file created,
    no signal state touched.
    """

    def test_win32_platform_yields_unsupported_everywhere(self):
        """Test win32 platform yields unsupported everywhere."""
        with TemporaryDirectory() as tmp:
            fifo_path = Path(tmp) / "probe.fifo"
            with mock.patch.object(sys, "platform", "win32"):
                self.assertFalse(platform_caps.alarm_available())
                self.assertFalse(platform_caps.make_fifo(fifo_path))
                self.assertFalse(fifo_path.exists())
                with platform_caps.alarm_deadline(60.0, lambda: None) as armed:
                    self.assertFalse(armed)
                with platform_caps.address_space_cap(1024) as installed:
                    self.assertFalse(installed)

    def test_make_fifo_without_mkfifo_attribute(self):
        """Test make fifo without mkfifo attribute."""
        import types

        with (
            TemporaryDirectory() as tmp,
            mock.patch.object(platform_caps, "os", types.SimpleNamespace()),
        ):
            self.assertFalse(platform_caps.make_fifo(Path(tmp) / "probe.fifo"))

    def test_handler_install_failure_yields_unarmed(self):
        """Test handler install failure yields unarmed."""
        if not platform_caps.alarm_available():
            self.skipTest("SIGALRM host expected")
        before = signal.getsignal(signal.SIGALRM)
        with (
            mock.patch.object(
                platform_caps.signal, "signal", side_effect=OSError("cannot install")
            ),
            platform_caps.alarm_deadline(60.0, lambda: None) as armed,
        ):
            self.assertFalse(armed)
        self.assertEqual(signal.getsignal(signal.SIGALRM), before)

    def test_arm_failure_yields_unarmed_and_restores_handler(self):
        """Test arm failure yields unarmed and restores handler."""
        if not platform_caps.alarm_available():
            self.skipTest("SIGALRM host expected")
        import signal as real_signal

        class ArmFailSignal:
            """Signal stand-in whose setitimer always fails."""

            def setitimer(self, *args):
                raise OSError("cannot arm")

            def __getattr__(self, name):
                return getattr(real_signal, name)

        before = signal.getsignal(signal.SIGALRM)
        with (
            mock.patch.object(platform_caps, "signal", ArmFailSignal()),
            platform_caps.alarm_deadline(60.0, lambda: None) as armed,
        ):
            self.assertFalse(armed)
        self.assertEqual(signal.getsignal(signal.SIGALRM), before)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))

    def test_resource_import_failure_yields_uninstalled(self):
        """Test resource import failure yields uninstalled."""
        if sys.platform == "win32":
            self.skipTest("POSIX host expected")
        with (
            mock.patch.dict(sys.modules, {"resource": None}),
            platform_caps.address_space_cap(1024) as installed,
        ):
            self.assertFalse(installed)

    def test_proc_status_unreadable_yields_uninstalled(self):
        """Test proc status unreadable yields uninstalled."""
        if sys.platform == "win32":
            self.skipTest("POSIX host expected")
        real_open = open

        def fake_open(path, *args, **kwargs):
            """Fake open."""
            if str(path) == "/proc/self/status":
                raise OSError("no proc")
            return real_open(path, *args, **kwargs)

        with (
            mock.patch("builtins.open", side_effect=fake_open),
            platform_caps.address_space_cap(1024) as installed,
        ):
            self.assertFalse(installed)

    def test_finite_ambient_limit_mock_yields_uninstalled(self):
        """Test finite ambient limit mock yields uninstalled."""
        if sys.platform == "win32":
            self.skipTest("POSIX host expected")
        import resource

        setrlimit = mock.Mock(side_effect=AssertionError("must not be called"))
        with (
            mock.patch.object(resource, "getrlimit", return_value=(1024, resource.RLIM_INFINITY)),
            mock.patch.object(resource, "setrlimit", setrlimit),
            platform_caps.address_space_cap(1024) as installed,
        ):
            self.assertFalse(installed)
        setrlimit.assert_not_called()

    def test_restore_oserror_is_swallowed(self):
        """Test restore oserror is swallowed."""
        if sys.platform == "win32":
            self.skipTest("POSIX host expected")
        import resource

        old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
        if old_soft != resource.RLIM_INFINITY:
            self.skipTest("ambient soft limit is finite; install path not exercisable")
        real_setrlimit = resource.setrlimit
        calls: list[tuple[int, tuple[int, int]]] = []

        def flaky(which, limits):
            """Flaky."""
            calls.append((which, limits))
            if len(calls) == 2:
                raise OSError("restore failed")
            return real_setrlimit(which, limits)

        try:
            with (
                mock.patch.object(resource, "setrlimit", side_effect=flaky),
                platform_caps.address_space_cap(256 * 1024 * 1024) as installed,
            ):
                self.assertTrue(installed)
        finally:
            # The swallowed restore failure leaves the cap installed;
            # put the process limit back before anything else runs.
            real_setrlimit(resource.RLIMIT_AS, (old_soft, old_hard))
        self.assertEqual(len(calls), 2)
        self.assertEqual(resource.getrlimit(resource.RLIMIT_AS), (old_soft, old_hard))

    def test_materialize_fifo_unsupported_raises_valueerror(self):
        """Test materialize fifo unsupported raises valueerror."""
        from model_protocol_lab import fixturefuzz

        cases = [c for c in fixturefuzz.generate_cases(0) if c.file_kind == "fifo"]
        self.assertTrue(cases)
        with (
            TemporaryDirectory() as tmp,
            mock.patch.object(platform_caps, "make_fifo", return_value=False),
            self.assertRaises(ValueError),
        ):
            fixturefuzz._materialize(cases[0], Path(tmp))


if __name__ == "__main__":
    unittest.main()
