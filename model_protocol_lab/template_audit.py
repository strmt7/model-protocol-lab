"""Template conformance firewall (research backlog R3).

A checkpoint's ``chat_template`` is attacker-controlled Jinja: CVE-2026-5760
(CVSS 9.8) is unsandboxed render of a model-shipped chat template leading to
RCE (SGLang hostile ``tokenizer.chat_template``), and CVE-2025-27516 is a
Jinja2 sandbox ``|attr`` escape (fixed in 3.1.6). The lab's existing Jinja
mitigation is a 10s CPU watchdog — DoS only. Nothing stops *code execution*
from a hostile template during conformance renders, the exact class the lab
is exposed to by design (it renders the checkpoint's template locally).

This module is the static layer: :func:`scan_template_ast` walks the parsed
Jinja AST *before* any render and flags the statically recognizable RCE
primitives (dangerous globals, direct dunder/subscript access, ``|attr``
with a constant argument) with fixed-vocabulary findings. It is a
pattern-matching pre-filter, not a sandbox: obfuscated spellings such as
concatenated dunders (``{{ config['__cl' ~ 'ass__'] }}``) are invisible to
the static scan by construction. Those payloads are neutralized one layer
down, by the sandboxed render. It is pure (no render, no model) and fully
testable on a synthetic SSTI corpus. The checkpoint-backed driver is
:func:`doctor.template_firewall`; :func:`doctor.template_conformance`
refuses to render a denied template (fail closed).

The render itself inherits transformers' ``ImmutableSandboxedEnvironment``
(and the lab pins ``jinja2>=3.1.6`` for the sandbox-escape fix), so the
stack is: static firewall -> sandboxed render -> CPU watchdog + memory cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TemplateFinding:
    """One firewall rule hit: fixed-vocabulary rule id plus detail."""

    rule: str
    detail: str


# Jinja globals that expose Python internals; never legitimate in a chat
# template. (`namespace` is deliberately NOT denied: real tool-calling
# templates use it for loop-local state.)
_DANGEROUS_GLOBALS = frozenset({"lipsum", "cycler", "joiner", "self"})

# Upper bound for a literal range() call a chat template may plausibly
# need (message-list iteration). Larger or non-literal bounds are a DoS
# primitive (the CPU watchdog is the backstop; this fails closed first).
_RANGE_LITERAL_BOUND = 1_000_000


def _findings_for(source: str) -> tuple[TemplateFinding, ...]:
    """Walk the parsed AST; implementation of :func:`scan_template_ast`."""
    try:
        from jinja2 import Environment, nodes
    except ImportError:
        raise
    # Parse-only: the AST is walked for the firewall, nothing is rendered
    # here, so autoescape is inapplicable (ruff S701 / bandit B701).
    tree = Environment().parse(source)  # noqa: S701 # nosec B701
    findings: list[TemplateFinding] = []

    def deny(rule: str, detail: str) -> None:
        findings.append(TemplateFinding(rule=rule, detail=detail))

    def visit(node: Any) -> None:
        # Attribute access: only dunder/private names are dangerous;
        # message.role / message.content must keep working.
        if isinstance(node, nodes.Getattr):
            if type(node.attr) is str and node.attr.startswith("_"):
                deny("dunder_getattr", f"dunder attribute access: {node.attr!r}")
        elif isinstance(node, nodes.Getitem):
            arg = node.arg
            if type(arg) is nodes.Const and type(arg.value) is str and arg.value.startswith("_"):
                deny("dunder_getitem", f"dunder subscript: {arg.value!r}")
        elif isinstance(node, nodes.Filter):
            if node.name == "attr":
                deny("attr_filter", "the |attr filter enables sandbox escape")
        elif isinstance(node, nodes.Name):
            if node.name in _DANGEROUS_GLOBALS:
                deny("dangerous_global", f"dangerous Jinja global: {node.name!r}")
        elif isinstance(node, nodes.Call):
            func = node.node
            if type(func) is nodes.Name and func.name == "super":
                deny("super_call", "super() has no legitimate use in a chat template")
            elif type(func) is nodes.Name and func.name == "range":
                args = node.args
                if not args or not all(
                    type(a) is nodes.Const
                    and type(a.value) is int
                    and not isinstance(a.value, bool)
                    and abs(a.value) <= _RANGE_LITERAL_BOUND
                    for a in args
                ):
                    deny(
                        "unbounded_range",
                        "range() with non-literal or huge bounds is a DoS primitive",
                    )
        for child in node.iter_child_nodes():
            visit(child)

    visit(tree)
    return tuple(findings)


def scan_template_ast(source: str) -> tuple[TemplateFinding, ...]:
    """Scan a Jinja chat-template source for RCE primitives.

    Returns the findings (empty tuple = allowed). Never raises: a
    non-string source or a parse failure yields a single ``parse_error``
    finding (fail closed — an unscannable template is not rendered).
    """
    if type(source) is not str:
        return (TemplateFinding(rule="parse_error", detail="template source is not str"),)
    if not source.strip():
        return (TemplateFinding(rule="parse_error", detail="template source is empty"),)
    try:
        return _findings_for(source)
    except ImportError:
        return (TemplateFinding(rule="scan_unavailable", detail="jinja2 not installed"),)
    except Exception:  # noqa: BLE001 -- fail closed on any scan failure
        return (TemplateFinding(rule="parse_error", detail="template AST parse failed"),)
