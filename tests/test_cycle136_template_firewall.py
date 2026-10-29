"""Cycle 136: template conformance firewall (research R3).

New pure model_protocol_lab/template_audit.py (scan_template_ast: static
AST walk denying dunder getattr/getitem, the |attr filter, dangerous Jinja
globals, super(), and unbounded range(); fail-closed on parse failure),
the checkpoint-backed doctor.template_firewall driver, fail-closed wiring
in doctor.template_conformance (a denied template is never rendered),
`mpl run` attaching `template_firewall`, the HTML "Template firewall"
section, the jinja2>=3.1.6 floor pin, and the synthetic firewall payload in
the committed sample report.
"""

from __future__ import annotations

import html
import json
import tempfile
import unittest
from pathlib import Path

from model_protocol_lab import doctor as DOCTOR
from model_protocol_lab import reporting
from model_protocol_lab import template_audit as TA
from model_protocol_lab.template_audit import TemplateFinding, scan_template_ast

CHATML_JINJA = (
    "{% for message in messages %}"
    "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

LLAMA3_JINJA = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message.role == 'user' %}"
    "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message.content }}"
    "{% elif message.role == 'assistant' %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message.content }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)


class BenignTemplatesAllowedTests(unittest.TestCase):
    def test_chatml_style(self):
        self.assertEqual(scan_template_ast(CHATML_JINJA), ())

    def test_llama3_style(self):
        self.assertEqual(scan_template_ast(LLAMA3_JINJA), ())

    def test_plain_getattr_allowed(self):
        # message.role / message.content are the bread and butter of chat
        # templates; only dunder attribute access is denied.
        self.assertEqual(scan_template_ast("{{ message.role }}:{{ message.content }}"), ())

    def test_literal_range_allowed(self):
        self.assertEqual(scan_template_ast("{% for i in range(3) %}{{ i }}{% endfor %}"), ())

    def test_namespace_allowed(self):
        self.assertEqual(scan_template_ast("{% set ns = namespace(x=0) %}{{ ns.x }}"), ())

    def test_plain_text_allowed(self):
        self.assertEqual(scan_template_ast("Hello, world!"), ())


class SstiCorpusDeniedTests(unittest.TestCase):
    @staticmethod
    def _rules(source):
        return [f.rule for f in scan_template_ast(source)]

    def test_lipsum_globals(self):
        rules = self._rules("{{ lipsum.__globals__.os.popen('id').read() }}")
        self.assertIn("dangerous_global", rules)
        self.assertIn("dunder_getattr", rules)

    def test_dunder_class_chain(self):
        self.assertIn("dunder_getattr", self._rules("{{ ''.__class__.__mro__ }}"))

    def test_attr_filter(self):
        self.assertIn("attr_filter", self._rules("{{ x|attr('__class__') }}"))

    def test_dunder_getitem(self):
        self.assertIn("dunder_getitem", self._rules("{{ x['__globals__'] }}"))

    def test_self_reference(self):
        rules = self._rules("{{ self._TemplateReference__context }}")
        self.assertIn("dangerous_global", rules)

    def test_cycler_joiner(self):
        self.assertIn("dangerous_global", self._rules("{{ cycler('a', 'b') }}"))
        self.assertIn("dangerous_global", self._rules("{{ joiner(',') }}"))

    def test_unbounded_range_literal(self):
        self.assertIn("unbounded_range", self._rules("{% for i in range(10**12) %}x{% endfor %}"))

    def test_unbounded_range_variable(self):
        self.assertIn("unbounded_range", self._rules("{% for i in range(n) %}x{% endfor %}"))

    def test_super_call(self):
        self.assertIn("super_call", self._rules("{{ super() }}"))

    def test_findings_are_fixed_vocabulary(self):
        for finding in scan_template_ast("{{ lipsum.__globals__ }}{{ x|attr('y') }}"):
            self.assertIsInstance(finding, TemplateFinding)
            self.assertIsInstance(finding.rule, str)
            self.assertIsInstance(finding.detail, str)


class FailClosedTests(unittest.TestCase):
    def test_non_str_source(self):
        findings = scan_template_ast(None)  # type: ignore[arg-type]
        self.assertEqual([f.rule for f in findings], ["parse_error"])

    def test_empty_source(self):
        findings = scan_template_ast("   ")
        self.assertEqual([f.rule for f in findings], ["parse_error"])

    def test_unparseable_source(self):
        findings = scan_template_ast("{% if %}")
        self.assertEqual([f.rule for f in findings], ["parse_error"])


def _write_config(root: Path, chat_template) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": chat_template}), encoding="utf-8"
    )
    return root


class DoctorFirewallTests(unittest.TestCase):
    def test_allowed(self):
        root = _write_config(Path(tempfile.mkdtemp(prefix="mpl_fw_")), CHATML_JINJA)
        out = DOCTOR.template_firewall(root)
        self.assertEqual(out["status"], "allowed")
        self.assertEqual(out["findings"], [])

    def test_denied(self):
        root = _write_config(
            Path(tempfile.mkdtemp(prefix="mpl_fw_")),
            "{{ lipsum.__globals__.os.popen('id').read() }}",
        )
        out = DOCTOR.template_firewall(root)
        self.assertEqual(out["status"], "denied")
        rules = [f["rule"] for f in out["findings"]]
        self.assertIn("dangerous_global", rules)

    def test_no_template(self):
        root = _write_config(Path(tempfile.mkdtemp(prefix="mpl_fw_")), None)
        out = DOCTOR.template_firewall(root)
        self.assertEqual(out["status"], "no_template")

    def test_missing_dir_unavailable(self):
        missing = Path(tempfile.mkdtemp(prefix="mpl_fw_")) / "no_such_dir"
        out = DOCTOR.template_firewall(missing)
        self.assertEqual(out["status"], "unavailable")

    def test_never_renders_denied_template(self):
        # template_conformance must fail closed: a denied template yields
        # status "denied" without reaching the Jinja compiler. A template
        # that would also explode at render time proves no render happened
        # (no exception escapes).
        from model_protocol_lab.core import Fixture, Format, Message

        root = _write_config(
            Path(tempfile.mkdtemp(prefix="mpl_fw_")),
            "{{ lipsum.__globals__ }}{% for i in range(10**18) %}x{% endfor %}",
        )
        fmt = Format(
            name="chatml",
            bos="",
            user_prefix="u:",
            assistant_prefix="a:",
            turn_end="\n",
            repeat_bos=False,
            user_turn_end="\n",
            assistant_turn_end="\n",
            open_suffix="a:",
        )
        case = Fixture(
            id="f1",
            messages=(Message(role="user", content="hi"),),
            expected="",
            group="g",
        )
        out = DOCTOR.template_conformance(root, (fmt,), (case,))
        self.assertEqual(out["status"], "denied")
        self.assertEqual(out["diffs"], [])
        self.assertTrue(out["findings"])


class FirewallHtmlTests(unittest.TestCase):
    @staticmethod
    def _result(firewall):
        return {
            "decision": "compatible_on_fixture_set",
            "selected_candidate": "chatml",
            "selection_rule": "x",
            "measurement": {},
            "statistical_confidence": {},
            "compute_comparison": {},
            "backend_fingerprint": {},
            "search": [],
            "template_firewall": firewall,
        }

    def test_allowed_renders(self):
        rendered = reporting.render_html(self._result({"status": "allowed", "findings": []}))
        self.assertIn('id="template-firewall"', rendered)
        self.assertIn("allowed", rendered)

    def test_denied_lists_findings(self):
        rendered = reporting.render_html(
            self._result(
                {
                    "status": "denied",
                    "findings": [{"rule": "dunder_getattr", "detail": "x"}],
                }
            )
        )
        self.assertIn("denied", rendered)
        self.assertIn("dunder_getattr", rendered)

    def test_absent_when_not_run(self):
        result = self._result({"status": "unavailable"})
        rendered = reporting.render_html(result)
        self.assertNotIn('id="template-firewall"', rendered)

    def test_hostile_findings_escaped(self):
        hostile = '"><img src=x onerror=alert(1)>'
        rendered = reporting.render_html(
            self._result(
                {
                    "status": "denied",
                    "findings": [{"rule": hostile, "detail": hostile}],
                }
            )
        )
        self.assertNotIn(hostile, rendered)
        self.assertIn(html.escape(hostile), rendered)


class FirewallMutationGateTests(unittest.TestCase):
    """4 committed mutants on template_audit.py; the suite must kill each."""

    MUTANTS = {
        # M1: dunder getattr no longer denied.
        "m1": (
            'if type(node.attr) is str and node.attr.startswith("_"):',
            'if type(node.attr) is str and node.attr.startswith("__") and False:',
        ),
        # M2: |attr filter no longer denied.
        "m2": (
            'if node.name == "attr":',
            'if node.name == "attr_" or node.name == "attrx":',
        ),
        # M3: lipsum no longer dangerous.
        "m3": (
            '_DANGEROUS_GLOBALS = frozenset({"lipsum", "cycler", "joiner", "self"})',
            '_DANGEROUS_GLOBALS = frozenset({"cycler", "joiner", "self"})',
        ),
        # M4: unbounded range no longer denied.
        "m4": (
            "abs(a.value) <= _RANGE_LITERAL_BOUND",
            "abs(a.value) <= _RANGE_LITERAL_BOUND or True",
        ),
    }

    def test_mutants_killed(self):
        import subprocess
        import sys

        src = Path(TA.__file__)
        original = src.read_text(encoding="utf-8")
        base = "tests/test_cycle136_template_firewall.py"
        targets = {
            "m1": f"{base}::SstiCorpusDeniedTests::test_dunder_class_chain",
            "m2": f"{base}::SstiCorpusDeniedTests::test_attr_filter",
            "m3": f"{base}::SstiCorpusDeniedTests::test_lipsum_globals",
            "m4": f"{base}::SstiCorpusDeniedTests::test_unbounded_range_literal",
        }
        killed = []
        survived = []
        for name, (old, new) in self.MUTANTS.items():
            self.assertIn(old, original, f"mutant {name} anchor missing")
            src.write_text(original.replace(old, new, 1), encoding="utf-8")
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        "-p",
                        "no:cacheprovider",
                        "-x",
                        "--tb=no",
                        targets[name],
                    ],
                    cwd=str(Path(__file__).resolve().parent.parent),
                    capture_output=True,
                    timeout=300,
                    check=False,
                )
            finally:
                src.write_text(original, encoding="utf-8")
                # Invalidate the bytecode cache: a same-size mutant can
                # leave a .pyc whose (mtime, size) still validates against
                # the reverted source, poisoning later imports with mutant
                # bytecode (see the cycle-135 gate fix).
                for pyc in src.parent.glob("__pycache__/template_audit.*.pyc"):
                    pyc.unlink(missing_ok=True)
            if proc.returncode != 0:
                killed.append(name)
            else:
                survived.append(name)
        self.assertEqual(survived, [], f"mutants survived: {survived}")
        self.assertEqual(sorted(killed), sorted(self.MUTANTS))


if __name__ == "__main__":
    unittest.main()
