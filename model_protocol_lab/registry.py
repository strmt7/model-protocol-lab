"""The section registry: one typed row per results section key.

Every key a ``results.json`` payload can carry has exactly one
:class:`Instrument` row here (65 keys at the W7 baseline: 57 diffed
by :mod:`model_protocol_lab.diffreports`, 8 intentionally
undispatched). The diff dispatch tables in ``diffreports``
(``_SECTION_PAIRS`` / ``SECTION_DIFF_KEYS`` /
``UNDISPATCHED_SECTIONS``) are derived views of this table, so a
new producer section cannot silently go undispatched: it has no
row, and the registry pins (tests/test_w7_registry.py) fail.

Field semantics:

* ``producer`` — dotted path of the callable whose return value
  becomes the section payload. Where a CLI probe assembles the
  payload inline from a stage result, the producer is the probe;
  where ``run``/``run_experiment`` assembles it inline, the
  producer is that assembler.
* ``stage`` — dotted path of the stage that generates the
  section's measurements, or ``None`` for post-hoc sections
  computed from already-recorded observations and for CLI/doctor
  diagnostics.
* ``diff_key`` / ``differ`` / ``validator`` — the diff-side key,
  the differ and the mirror validator (dotted paths into
  ``diffreports``); all three are ``None`` exactly when
  ``undispatched_reason`` gives the reason the section is not
  diffed. The budget section's validator is ``_budget_side``, a
  recompute validator returning the validated side or ``None``,
  not a boolean ``_valid_*`` predicate.
* ``projection`` — the rule ``reporting._public_view`` applies to
  the section in the default (evidence-excluded) export, as
  shipped; ``projection_fields`` names the affected fields and
  ``projection_reason`` records what stays and why.
  ``config_allowlist`` is set only for ``serving_backends`` (the
  ``identity.config`` allowlist).
* ``docs_anchor`` — the USAGE heading the section is documented
  under (the heading text, without the leading ``#`` marks).
* ``cli_flag`` — the ``run`` flag that requests the section, or
  ``None`` when the section is attached unconditionally or under
  a data condition rather than a flag.

Metadata only: nothing in this module is imported by the
measurement path, and no behaviour is derived from the table
except the diff dispatch views named above.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProjectionKind(Enum):
    """How the default public view treats a section."""

    VERBATIM = "verbatim"
    STRIP_FIELDS = "strip_fields"


@dataclass(frozen=True)
class Instrument:
    """One results section key and everything that consumes it."""

    key: str
    producer: str
    stage: str | None
    diff_key: str | None
    differ: str | None
    validator: str | None
    undispatched_reason: str | None
    projection: ProjectionKind
    projection_fields: frozenset[str]
    projection_reason: str
    config_allowlist: frozenset[str] | None
    docs_anchor: str
    cli_flag: str | None


INSTRUMENTS: tuple[Instrument, ...] = (
    Instrument(
        key="budget_frontier",
        producer="model_protocol_lab.experiments.budget_frontier",
        stage=None,
        diff_key="frontier",
        differ="model_protocol_lab.diffreports.diff_frontiers",
        validator="model_protocol_lab.diffreports._valid_frontier",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.6 Token-budget robustness frontier",
        cli_flag=None,
    ),
    Instrument(
        key="held_out_budget_floor",
        producer="model_protocol_lab.experiments.held_out_budget_floor",
        stage=None,
        diff_key="held_out_floor",
        differ="model_protocol_lab.diffreports.diff_held_out_floor",
        validator="model_protocol_lab.diffreports._valid_held_out_floor",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.6 Token-budget robustness frontier",
        cli_flag=None,
    ),
    Instrument(
        key="budget",
        producer="model_protocol_lab.experiments.run_experiment",
        stage=None,
        diff_key="budget",
        differ="model_protocol_lab.diffreports.diff_budgets",
        validator="model_protocol_lab.diffreports._budget_side",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="5. Run a bounded experiment",
        cli_flag=None,
    ),
    Instrument(
        key="boundary_audit",
        producer="model_protocol_lab.boundary_audit.audit_search_observations",
        stage=None,
        diff_key="boundary_audit",
        differ="model_protocol_lab.diffreports.diff_boundary_audit",
        validator="model_protocol_lab.diffreports._valid_boundary_audit",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.10 Prompt-boundary canonicality (`boundary_audit`)",
        cli_flag=None,
    ),
    Instrument(
        key="selection_robustness",
        producer="model_protocol_lab.experiments.selection_robustness",
        stage=None,
        diff_key="selection_robustness",
        differ="model_protocol_lab.diffreports.diff_selection_robustness",
        validator="model_protocol_lab.diffreports._valid_selection_robustness",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.13 Selection robustness under stop semantics (`selection_robustness`)",
        cli_flag=None,
    ),
    Instrument(
        key="transfer",
        producer="model_protocol_lab.experiments.transfer_portability",
        stage="model_protocol_lab.experiments.transfer_stage",
        diff_key="transfer",
        differ="model_protocol_lab.diffreports.diff_transfer",
        validator="model_protocol_lab.diffreports._valid_transfer",
        undispatched_reason=None,
        projection=ProjectionKind.STRIP_FIELDS,
        projection_fields=frozenset({"path"}),
        projection_reason=(
            "checkpoints[].path is a host filesystem path; stripped in the default view. "
            "Checkpoint ids and fingerprints stay."
        ),
        config_allowlist=None,
        docs_anchor="8.14 Cross-checkpoint transfer (`transfer`)",
        cli_flag="--transfer-model",
    ),
    Instrument(
        key="contamination",
        producer="model_protocol_lab.contamination.contamination_report",
        stage=None,
        diff_key="contamination",
        differ="model_protocol_lab.diffreports.diff_contamination",
        validator="model_protocol_lab.diffreports._valid_contamination",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="1. Inspect the fixture schema",
        cli_flag=None,
    ),
    Instrument(
        key="perturbation",
        producer="model_protocol_lab.experiments.perturbation_robustness",
        stage=None,
        diff_key="perturbation",
        differ="model_protocol_lab.diffreports.diff_perturbation",
        validator="model_protocol_lab.diffreports._valid_perturbation",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.15 Fixture-perturbation robustness (`perturbation`)",
        cli_flag="--perturb",
    ),
    Instrument(
        key="token_healing",
        producer="model_protocol_lab.experiments.healing_delta",
        stage=None,
        diff_key="token_healing",
        differ="model_protocol_lab.diffreports.diff_token_healing",
        validator="model_protocol_lab.diffreports._valid_token_healing",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.16 Token-healing selection sensitivity (`token_healing`)",
        cli_flag="--token-healing",
    ),
    Instrument(
        key="hierarchy",
        producer="model_protocol_lab.experiments.hierarchy_adherence",
        stage="model_protocol_lab.experiments.hierarchy_stage",
        diff_key="hierarchy",
        differ="model_protocol_lab.diffreports.diff_hierarchy",
        validator="model_protocol_lab.diffreports._valid_hierarchy",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.17 Instruction-hierarchy adherence probe (`hierarchy`)",
        cli_flag="--hierarchy",
    ),
    Instrument(
        key="prefill",
        producer="model_protocol_lab.experiments.prefill_adherence",
        stage="model_protocol_lab.experiments.prefill_stage",
        diff_key="prefill",
        differ="model_protocol_lab.diffreports.diff_prefill",
        validator="model_protocol_lab.diffreports._valid_prefill",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.18 Assistant-prefill sensitivity (`prefill`)",
        cli_flag="--prefill",
    ),
    Instrument(
        key="portfolio",
        producer="model_protocol_lab.experiments.portfolio_analysis",
        stage=None,
        diff_key="portfolio",
        differ="model_protocol_lab.diffreports.diff_portfolio",
        validator="model_protocol_lab.diffreports._valid_portfolio",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.19 Protocol portfolio (`portfolio`)",
        cli_flag=None,
    ),
    Instrument(
        key="synthhistory",
        producer="model_protocol_lab.experiments.synthhistory_stability",
        stage="model_protocol_lab.experiments.synthhistory_stage",
        diff_key="synthhistory",
        differ="model_protocol_lab.diffreports.diff_synthhistory",
        validator="model_protocol_lab.diffreports._valid_synthhistory",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.20 Synthetic-history selection stability (`synthhistory`)",
        cli_flag="--synth-history",
    ),
    Instrument(
        key="batch_invariance",
        producer="model_protocol_lab.experiments.batch_stage",
        stage="model_protocol_lab.experiments.batch_stage",
        diff_key="batch_invariance",
        differ="model_protocol_lab.diffreports.diff_batchinv",
        validator="model_protocol_lab.diffreports._valid_batchinv",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.21 Batch-invariance audit (`batch_invariance`)",
        cli_flag="--batch-invariance",
    ),
    Instrument(
        key="position",
        producer="model_protocol_lab.experiments.position_sensitivity",
        stage="model_protocol_lab.experiments.position_stage",
        diff_key="position",
        differ="model_protocol_lab.diffreports.diff_position",
        validator="model_protocol_lab.diffreports._valid_position",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.22 Position sensitivity (`position`)",
        cli_flag="--position",
    ),
    Instrument(
        key="stop_cause",
        producer="model_protocol_lab.experiments.stop_stage",
        stage="model_protocol_lab.experiments.stop_stage",
        diff_key="stop_cause",
        differ="model_protocol_lab.diffreports.diff_stop_cause",
        validator="model_protocol_lab.diffreports._valid_stop_cause",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.23 Stop-cause termination audit (`stop_cause`)",
        cli_flag="--stop-cause",
    ),
    Instrument(
        key="ws_canonical",
        producer="model_protocol_lab.experiments.ws_canonicalization",
        stage="model_protocol_lab.experiments.ws_stage",
        diff_key="ws_canonical",
        differ="model_protocol_lab.diffreports.diff_ws_canonical",
        validator="model_protocol_lab.diffreports._valid_ws_canonical",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.24 Whitespace canonicalization (`ws_canonical`)",
        cli_flag="--ws-scan",
    ),
    Instrument(
        key="structured",
        producer="model_protocol_lab.experiments.structured_adherence",
        stage="model_protocol_lab.experiments.structured_stage",
        diff_key="structured",
        differ="model_protocol_lab.diffreports.diff_structured",
        validator="model_protocol_lab.diffreports._valid_structured",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.25 Structured-output adherence (`structured`)",
        cli_flag="--structured",
    ),
    Instrument(
        key="deliforge",
        producer="model_protocol_lab.experiments.deliforge_authority",
        stage="model_protocol_lab.experiments.deliforge_stage",
        diff_key="deliforge",
        differ="model_protocol_lab.diffreports.diff_deliforge",
        validator="model_protocol_lab.diffreports._valid_deliforge",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.26 Delimiter authority (`deliforge`)",
        cli_flag="--deliforge",
    ),
    Instrument(
        key="fewshot",
        producer="model_protocol_lab.experiments.fewshot_gain",
        stage="model_protocol_lab.experiments.fewshot_stage",
        diff_key="fewshot",
        differ="model_protocol_lab.diffreports.diff_fewshot",
        validator="model_protocol_lab.diffreports._valid_fewshot",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.27 Few-shot x template (`fewshot`)",
        cli_flag="--few-shot",
    ),
    Instrument(
        key="faketurn",
        producer="model_protocol_lab.experiments.faketurn_robustness",
        stage="model_protocol_lab.experiments.faketurn_stage",
        diff_key="faketurn",
        differ="model_protocol_lab.diffreports.diff_faketurn",
        validator="model_protocol_lab.diffreports._valid_faketurn",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.28 Fake-turn history tampering (`faketurn`)",
        cli_flag="--fake-turn",
    ),
    Instrument(
        key="toolcall",
        producer="model_protocol_lab.experiments.toolcall_adherence",
        stage="model_protocol_lab.experiments.toolcall_stage",
        diff_key="toolcall",
        differ="model_protocol_lab.diffreports.diff_toolcall",
        validator="model_protocol_lab.diffreports._valid_toolcall",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.29 Prompt-only tool-call adherence (`toolcall`)",
        cli_flag="--tool-call",
    ),
    Instrument(
        key="verbosity",
        producer="model_protocol_lab.experiments.verbosity_profile",
        stage="model_protocol_lab.experiments.verbosity_stage",
        diff_key="verbosity",
        differ="model_protocol_lab.diffreports.diff_verbosity",
        validator="model_protocol_lab.diffreports._valid_verbosity",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.30 Response verbosity profile (`verbosity`)",
        cli_flag="--verbosity",
    ),
    Instrument(
        key="refusals",
        producer="model_protocol_lab.experiments.refusal_profile",
        stage=None,
        diff_key="refusals",
        differ="model_protocol_lab.diffreports.diff_refusals",
        validator="model_protocol_lab.diffreports._valid_refusals",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.36 Suspected-refusal profile (`refusals`)",
        cli_flag="--refusals",
    ),
    Instrument(
        key="scorerlens",
        producer="model_protocol_lab.experiments.scorerlens_profile",
        stage=None,
        diff_key="scorerlens",
        differ="model_protocol_lab.diffreports.diff_scorerlens",
        validator="model_protocol_lab.diffreports._valid_scorerlens",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.37 Scorer-lens regrade (`scorerlens`)",
        cli_flag="--scorer-lens",
    ),
    Instrument(
        key="aliasing",
        producer="model_protocol_lab.experiments.aliasing_profile",
        stage=None,
        diff_key="aliasing",
        differ="model_protocol_lab.diffreports.diff_aliasing",
        validator="model_protocol_lab.diffreports._valid_aliasing",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.38 Candidate aliasing audit (`aliasing`)",
        cli_flag="--aliasing",
    ),
    Instrument(
        key="repetition",
        producer="model_protocol_lab.experiments.repetition_analysis",
        stage="model_protocol_lab.experiments.repetition_stage",
        diff_key="repetition",
        differ="model_protocol_lab.diffreports.diff_repetition",
        validator="model_protocol_lab.diffreports._valid_repetition",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.39 Prompt repetition (`repetition`)",
        cli_flag="--repeat",
    ),
    Instrument(
        key="distraction",
        producer="model_protocol_lab.experiments.distraction_analysis",
        stage="model_protocol_lab.experiments.distraction_stage",
        diff_key="distraction",
        differ="model_protocol_lab.diffreports.diff_distraction",
        validator="model_protocol_lab.diffreports._valid_distraction",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.40 Distractor dilution (`distraction`)",
        cli_flag="--distract",
    ),
    Instrument(
        key="leading",
        producer="model_protocol_lab.experiments.leading_analysis",
        stage="model_protocol_lab.experiments.leading_stage",
        diff_key="leading",
        differ="model_protocol_lab.diffreports.diff_leading",
        validator="model_protocol_lab.diffreports._valid_leading",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.41 Leading-claim caving (`leading`)",
        cli_flag="--leading",
    ),
    Instrument(
        key="numform",
        producer="model_protocol_lab.experiments.numform_analysis",
        stage="model_protocol_lab.experiments.numform_stage",
        diff_key="numform",
        differ="model_protocol_lab.diffreports.diff_numform",
        validator="model_protocol_lab.diffreports._valid_numform",
        undispatched_reason=None,
        projection=ProjectionKind.STRIP_FIELDS,
        projection_fields=frozenset({"answer_text"}),
        projection_reason=(
            "Per-record answer_text is the decoded model answer verbatim; it is stripped "
            "from records, their cells and candidate records in the default view. Digests "
            "and aggregates stay."
        ),
        config_allowlist=None,
        docs_anchor="8.42 Numeric surface form (`numform`)",
        cli_flag="--numform",
    ),
    Instrument(
        key="dateline",
        producer="model_protocol_lab.experiments.dateline_analysis",
        stage="model_protocol_lab.experiments.dateline_stage",
        diff_key="dateline",
        differ="model_protocol_lab.diffreports.diff_dateline",
        validator="model_protocol_lab.diffreports._valid_dateline",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.43 Date-line sensitivity (`dateline`)",
        cli_flag="--dateline",
    ),
    Instrument(
        key="multiturn",
        producer="model_protocol_lab.experiments.multiturn_analysis",
        stage="model_protocol_lab.experiments.multiturn_stage",
        diff_key="multiturn",
        differ="model_protocol_lab.diffreports.diff_multiturn",
        validator="model_protocol_lab.diffreports._valid_multiturn",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.44 Multi-turn drift (`multiturn`)",
        cli_flag="--multiturn",
    ),
    Instrument(
        key="needle",
        producer="model_protocol_lab.experiments.needle_analysis",
        stage="model_protocol_lab.experiments.needle_stage",
        diff_key="needle",
        differ="model_protocol_lab.diffreports.diff_needle",
        validator="model_protocol_lab.diffreports._valid_needle",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.45 Needle depth (`needle`)",
        cli_flag="--needle",
    ),
    Instrument(
        key="placement",
        producer="model_protocol_lab.experiments.placement_analysis",
        stage="model_protocol_lab.experiments.placement_stage",
        diff_key="placement",
        differ="model_protocol_lab.diffreports.diff_placement",
        validator="model_protocol_lab.diffreports._valid_placement",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.46 Question/payload placement (`placement`)",
        cli_flag="--placement",
    ),
    Instrument(
        key="canonical_form",
        producer="model_protocol_lab.experiments.canonicalform_analysis",
        stage="model_protocol_lab.experiments.canonicalform_stage",
        diff_key="canonical_form",
        differ="model_protocol_lab.diffreports.diff_canonicalform",
        validator="model_protocol_lab.diffreports._valid_canonicalform",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.54 Canonical-form sensitivity (`run --canonical-form`)",
        cli_flag="--canonical-form",
    ),
    Instrument(
        key="instruction_distance",
        producer="model_protocol_lab.experiments.instruction_distance_analysis",
        stage="model_protocol_lab.experiments.instruction_distance_stage",
        diff_key="instruction_distance",
        differ="model_protocol_lab.diffreports.diff_instruction_distance",
        validator="model_protocol_lab.diffreports._valid_instruction_distance",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.56 Instruction distance (`run --instruction-distance`)",
        cli_flag="--instruction-distance",
    ),
    Instrument(
        key="history_frame",
        producer="model_protocol_lab.experiments.history_frame_analysis",
        stage="model_protocol_lab.experiments.history_frame_stage",
        diff_key="history_frame",
        differ="model_protocol_lab.diffreports.diff_history_frame",
        validator="model_protocol_lab.diffreports._valid_history_frame",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.57 History frame (`run --history-frame`)",
        cli_flag="--history-frame",
    ),
    Instrument(
        key="prune_sweep",
        producer="model_protocol_lab.experiments.prune_sweep_analysis",
        stage="model_protocol_lab.experiments.prune_sweep_stage",
        diff_key="prune_sweep",
        differ="model_protocol_lab.diffreports.diff_prune_sweep",
        validator="model_protocol_lab.diffreports._valid_prune_sweep",
        undispatched_reason=None,
        projection=ProjectionKind.STRIP_FIELDS,
        projection_fields=frozenset({"answer_text"}),
        projection_reason=(
            "Per-record answer_text is the decoded model answer verbatim; it is stripped "
            "from records, their cells and candidate records in the default view. Digests "
            "and aggregates stay."
        ),
        config_allowlist=None,
        docs_anchor="8.58 Prune sweep (`run --prune-sweep`)",
        cli_flag="--prune-sweep",
    ),
    Instrument(
        key="record_notation",
        producer="model_protocol_lab.experiments.record_notation_analysis",
        stage="model_protocol_lab.experiments.record_notation_stage",
        diff_key="record_notation",
        differ="model_protocol_lab.diffreports.diff_record_notation",
        validator="model_protocol_lab.diffreports._valid_record_notation",
        undispatched_reason=None,
        projection=ProjectionKind.STRIP_FIELDS,
        projection_fields=frozenset({"answer_text"}),
        projection_reason=(
            "Per-record answer_text is the decoded model answer verbatim; it is stripped "
            "from records, their cells and candidate records in the default view. Digests "
            "and aggregates stay."
        ),
        config_allowlist=None,
        docs_anchor="8.59 Record notation (`run --record-notation`)",
        cli_flag="--record-notation",
    ),
    Instrument(
        key="invisible_chars",
        producer="model_protocol_lab.experiments.invisible_chars_analysis",
        stage="model_protocol_lab.experiments.invisible_chars_stage",
        diff_key="invisible_chars",
        differ="model_protocol_lab.diffreports.diff_invisible_chars",
        validator="model_protocol_lab.diffreports._valid_invisible_chars",
        undispatched_reason=None,
        projection=ProjectionKind.STRIP_FIELDS,
        projection_fields=frozenset({"answer_text"}),
        projection_reason=(
            "Per-record answer_text is the decoded model answer verbatim; it is stripped "
            "from records, their cells and candidate records in the default view. Digests "
            "and aggregates stay."
        ),
        config_allowlist=None,
        docs_anchor="8.60 Invisible characters (`run --invisible-chars`)",
        cli_flag="--invisible-chars",
    ),
    Instrument(
        key="confusable_skeleton",
        producer="model_protocol_lab.experiments.confusable_skeleton_analysis",
        stage="model_protocol_lab.experiments.confusable_skeleton_stage",
        diff_key="confusable_skeleton",
        differ="model_protocol_lab.diffreports.diff_confusable_skeleton",
        validator="model_protocol_lab.diffreports._valid_confusable_skeleton",
        undispatched_reason=None,
        projection=ProjectionKind.STRIP_FIELDS,
        projection_fields=frozenset({"answer_text"}),
        projection_reason=(
            "Per-record answer_text is the decoded model answer verbatim; it is stripped "
            "from records, their cells and candidate records in the default view. Digests "
            "and aggregates stay."
        ),
        config_allowlist=None,
        docs_anchor="8.61 Confusable skeleton (`run --confusable-skeleton`)",
        cli_flag="--confusable-skeleton",
    ),
    Instrument(
        key="compatibility_form",
        producer="model_protocol_lab.experiments.compatibility_form_analysis",
        stage="model_protocol_lab.experiments.compatibility_form_stage",
        diff_key="compatibility_form",
        differ="model_protocol_lab.diffreports.diff_compatibility_form",
        validator="model_protocol_lab.diffreports._valid_compatibility_form",
        undispatched_reason=None,
        projection=ProjectionKind.STRIP_FIELDS,
        projection_fields=frozenset({"answer_text"}),
        projection_reason=(
            "Per-record answer_text is the decoded model answer verbatim; it is stripped "
            "from records, their cells and candidate records in the default view. Digests "
            "and aggregates stay."
        ),
        config_allowlist=None,
        docs_anchor="8.62 Compatibility form (`run --compatibility-form`)",
        cli_flag="--compatibility-form",
    ),
    Instrument(
        key="enumeration_convention",
        producer="model_protocol_lab.experiments.enumeration_convention_analysis",
        stage="model_protocol_lab.experiments.enumeration_convention_stage",
        diff_key="enumeration_convention",
        differ="model_protocol_lab.diffreports.diff_enumeration_convention",
        validator="model_protocol_lab.diffreports._valid_enumeration_convention",
        undispatched_reason=None,
        projection=ProjectionKind.STRIP_FIELDS,
        projection_fields=frozenset({"answer_text"}),
        projection_reason=(
            "Per-record answer_text is the decoded model answer verbatim; it is stripped "
            "from records, their cells and candidate records in the default view. Digests "
            "and aggregates stay."
        ),
        config_allowlist=None,
        docs_anchor="8.63 Enumeration convention (`run --enumeration-convention`)",
        cli_flag="--enumeration-convention",
    ),
    Instrument(
        key="typographic_emphasis",
        producer="model_protocol_lab.experiments.typographic_emphasis_analysis",
        stage="model_protocol_lab.experiments.typographic_emphasis_stage",
        diff_key="typographic_emphasis",
        differ="model_protocol_lab.diffreports.diff_typographic_emphasis",
        validator="model_protocol_lab.diffreports._valid_typographic_emphasis",
        undispatched_reason=None,
        projection=ProjectionKind.STRIP_FIELDS,
        projection_fields=frozenset({"answer_text"}),
        projection_reason=(
            "Per-record answer_text is the decoded model answer verbatim; it is stripped "
            "from records, their cells and candidate records in the default view. Digests "
            "and aggregates stay."
        ),
        config_allowlist=None,
        docs_anchor="8.64 Typographic emphasis (`run --typographic-emphasis`)",
        cli_flag="--typographic-emphasis",
    ),
    Instrument(
        key="choice_audit",
        producer="model_protocol_lab.experiments.choiceaudit_analysis",
        stage="model_protocol_lab.experiments.choiceaudit_stage",
        diff_key="choice_audit",
        differ="model_protocol_lab.diffreports.diff_choiceaudit",
        validator="model_protocol_lab.diffreports._valid_choiceaudit",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.47 Choice-scoring audit (`choice_audit`)",
        cli_flag="--choice-audit",
    ),
    Instrument(
        key="precision",
        producer="model_protocol_lab.experiments.precision_stage",
        stage="model_protocol_lab.experiments.precision_stage",
        diff_key="precision",
        differ="model_protocol_lab.diffreports.diff_precision",
        validator="model_protocol_lab.diffreports._valid_precision",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.48 Load precision (`precision`)",
        cli_flag="--precision",
    ),
    Instrument(
        key="confidence",
        producer="model_protocol_lab.experiments.confidence_stage",
        stage="model_protocol_lab.experiments.confidence_stage",
        diff_key="confidence",
        differ="model_protocol_lab.diffreports.diff_confidence",
        validator="model_protocol_lab.diffreports._valid_confidence",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.49 Verbalized confidence (`confidence`)",
        cli_flag="--confidence",
    ),
    Instrument(
        key="rotation",
        producer="model_protocol_lab.experiments.rotation_analysis",
        stage="model_protocol_lab.experiments.rotation_stage",
        diff_key="rotation",
        differ="model_protocol_lab.diffreports.diff_rotation",
        validator="model_protocol_lab.diffreports._valid_rotation",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.50 Context rotation (`rotation`)",
        cli_flag="--rotate",
    ),
    Instrument(
        key="density",
        producer="model_protocol_lab.experiments.density_analysis",
        stage="model_protocol_lab.experiments.density_stage",
        diff_key="density",
        differ="model_protocol_lab.diffreports.diff_density",
        validator="model_protocol_lab.diffreports._valid_density",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.51 Instruction density (`density`)",
        cli_flag="--density",
    ),
    Instrument(
        key="field_order",
        producer="model_protocol_lab.experiments.fieldorder_analysis",
        stage="model_protocol_lab.experiments.fieldorder_stage",
        diff_key="field_order",
        differ="model_protocol_lab.diffreports.diff_field_order",
        validator="model_protocol_lab.diffreports._valid_field_order",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.52 Schema field order (`field_order`)",
        cli_flag="--field-order",
    ),
    Instrument(
        key="cache_profile",
        producer="model_protocol_lab.experiments.cache_profile_profile",
        stage="model_protocol_lab.experiments.cache_profile_stage",
        diff_key="cache_profile",
        differ="model_protocol_lab.diffreports.diff_cache_profile",
        validator="model_protocol_lab.diffreports._valid_cache_profile",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.33 Prefix-cache sharability profile (`cache_profile`)",
        cli_flag="--cache-profile",
    ),
    Instrument(
        key="template_robustness",
        producer="model_protocol_lab.cli._run_template_robustness_probe",
        stage="model_protocol_lab.experiments.template_robustness_stage",
        diff_key="template_robustness",
        differ="model_protocol_lab.diffreports.diff_template_robustness",
        validator="model_protocol_lab.diffreports._valid_template_robustness",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.31 Template surface-noise robustness (`template_robustness`)",
        cli_flag="--template-robustness",
    ),
    Instrument(
        key="ablation",
        producer="model_protocol_lab.cli._run_ablation_probe",
        stage="model_protocol_lab.experiments.ablation_stage",
        diff_key="ablation",
        differ="model_protocol_lab.diffreports.diff_ablation",
        validator="model_protocol_lab.diffreports._valid_ablation",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.32 Template field ablation (`ablation`)",
        cli_flag="--ablate-fields",
    ),
    Instrument(
        key="confidentiality",
        producer="model_protocol_lab.experiments.confidentiality_profile",
        stage="model_protocol_lab.experiments.confidentiality_stage",
        diff_key="confidentiality",
        differ="model_protocol_lab.diffreports.diff_confidentiality",
        validator="model_protocol_lab.diffreports._valid_confidentiality",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.34 Labeled-block confidentiality (`confidentiality`)",
        cli_flag="--confidentiality",
    ),
    Instrument(
        key="answer_collision",
        producer="model_protocol_lab.experiments.answer_collision_profile",
        stage="model_protocol_lab.experiments.answer_collision_stage",
        diff_key="answer_collision",
        differ="model_protocol_lab.diffreports.diff_answer_collision",
        validator="model_protocol_lab.diffreports._valid_answer_collision",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.35 Answer-collision audit (`answer_collision`)",
        cli_flag="--answer-collision",
    ),
    Instrument(
        key="serving_backends",
        producer="model_protocol_lab.experiments.serving_backends_analysis",
        stage="model_protocol_lab.servingbackends.run_stage",
        diff_key="serving_backends",
        differ="model_protocol_lab.diffreports.diff_serving_backends",
        validator="model_protocol_lab.diffreports._valid_serving_backends",
        undispatched_reason=None,
        projection=ProjectionKind.STRIP_FIELDS,
        projection_fields=frozenset({"prompt_ids", "returned_ids"}),
        projection_reason=(
            "records[] carry prompt/returned token IDs (stripped in the default view) and "
            "identity.config is allowlist-filtered to the recorded keys, so unlisted "
            "server configuration cannot leak. Digests and verdicts stay."
        ),
        config_allowlist=frozenset(
            {
                "abort_probe",
                "build_info",
                "cache_probe",
                "checkpoint_sha256",
                "declared_eos_ids",
                "default_sampling_params_digest",
                "detokenizer_variant",
                "engine",
                "env",
                "generation_settings_digest",
                "model_path_digest",
                "platform",
                "posture",
                "protocol",
                "stack",
                "tier_a",
                "tier_a_vllm",
                "torch_version",
                "total_slots",
                "transformers_version",
                "vllm_module_digest",
                "vllm_version",
            }
        ),
        docs_anchor="8.65 Serving-stack measurement (`exec-backends`)",
        cli_flag=None,
    ),
    Instrument(
        key="negation",
        producer="model_protocol_lab.experiments.negation_analysis",
        stage="model_protocol_lab.experiments.negation_stage",
        diff_key="negation",
        differ="model_protocol_lab.diffreports.diff_negation",
        validator="model_protocol_lab.diffreports._valid_negation",
        undispatched_reason=None,
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.55 Antonym-negated definite-description discordance (`run --negation`)",
        cli_flag="--negation",
    ),
    Instrument(
        key="sensitivity_sweep",
        producer="model_protocol_lab.experiments.prompt_sensitivity_sweep",
        stage=None,
        diff_key=None,
        differ=None,
        validator=None,
        undispatched_reason=(
            "opt-in single-run CLI diagnostic attached in cli.py's fail-softened "
            "diagnostics block (the code's own contract: diagnostics must not fail the "
            "run); it describes the winning candidate of one run, is not a per-candidate "
            "measurement stage, and has no cross-run recompute standard to diff against"
        ),
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.5 Prompt-boundary sensitivity sweep (`run --sensitivity-sweep`)",
        cli_flag="--sensitivity-sweep",
    ),
    Instrument(
        key="sensitivity_sweep_tokens",
        producer="model_protocol_lab.experiments.prompt_token_depth_sweep",
        stage=None,
        diff_key=None,
        differ=None,
        validator=None,
        undispatched_reason=(
            "opt-in single-run CLI diagnostic attached in cli.py's fail-softened "
            "diagnostics block, token-axis twin of sensitivity_sweep; same disposition"
        ),
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.53 Token-depth sensitivity sweep (`run --token-depth-sweep`)",
        cli_flag="--token-depth-sweep",
    ),
    Instrument(
        key="stop_matrix",
        producer="model_protocol_lab.cli.main",
        stage=None,
        diff_key=None,
        differ=None,
        validator=None,
        undispatched_reason=(
            "opt-in single-run CLI diagnostic attached in cli.py's fail-softened "
            "diagnostics block; a per-run stop-behaviour matrix with no cross-run "
            "recompute standard"
        ),
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.2 Stop-behavior isolation matrix (`stop_matrix`)",
        cli_flag=None,
    ),
    Instrument(
        key="logprob_diagnostic",
        producer="model_protocol_lab.experiments.logprob_diagnostic",
        stage=None,
        diff_key=None,
        differ=None,
        validator=None,
        undispatched_reason=(
            "opt-in single-run CLI diagnostic attached in cli.py's fail-softened "
            "diagnostics block; free-prose diagnostic output, not a structured "
            "measurement stage"
        ),
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.9 Teacher-forced log-likelihood diagnostic (`run --logprob-diagnostic`)",
        cli_flag="--logprob-diagnostic",
    ),
    Instrument(
        key="template_conformance",
        producer="model_protocol_lab.doctor.template_conformance",
        stage=None,
        diff_key=None,
        differ=None,
        validator=None,
        undispatched_reason=(
            "doctor / diff-backends diagnostic surface: it is compared, where applicable, "
            "by the diff-backends tool's own conformance logic; it is not a per-run "
            "measurement section of results.json"
        ),
        projection=ProjectionKind.STRIP_FIELDS,
        projection_fields=frozenset({"hand_ids", "template_ids"}),
        projection_reason=(
            "hand_ids/template_ids enumerate the operator's template variants inside each "
            "conformance diff segment; stripped in the default view. Verdicts and digests "
            "stay."
        ),
        config_allowlist=None,
        docs_anchor="8.1 Template conformance",
        cli_flag=None,
    ),
    Instrument(
        key="template_firewall",
        producer="model_protocol_lab.doctor.template_firewall",
        stage=None,
        diff_key=None,
        differ=None,
        validator=None,
        undispatched_reason=(
            "doctor diagnostic surface (template firewall audit); no stored per-run "
            "measurement payload is produced for diff-reports"
        ),
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.12 Template firewall (`template_firewall`)",
        cli_flag=None,
    ),
    Instrument(
        key="marker_isolation",
        producer="model_protocol_lab.doctor.marker_isolation",
        stage=None,
        diff_key=None,
        differ=None,
        validator=None,
        undispatched_reason=(
            "doctor diagnostic surface (chat-marker isolation audit); no stored per-run "
            "measurement payload is produced for diff-reports"
        ),
        projection=ProjectionKind.VERBATIM,
        projection_fields=frozenset(),
        projection_reason=(
            "Carried unchanged in the default view: the section's stored fields are "
            "aggregates, verdicts and digests by construction, with no raw model text, "
            "token IDs or host paths (projection census; W7 re-census of "
            "reporting._public_view)."
        ),
        config_allowlist=None,
        docs_anchor="8.11 Chat-marker isolation (`marker_isolation`)",
        cli_flag=None,
    ),
    Instrument(
        key="backend_divergences",
        producer="model_protocol_lab.diffbackends.divergence_section_data",
        stage=None,
        diff_key=None,
        differ=None,
        validator=None,
        undispatched_reason=(
            "diff-backends diagnostic surface: the divergence listing is the output of "
            "the backend-comparison tool itself, not an input section two runs can be "
            "diffed on"
        ),
        projection=ProjectionKind.STRIP_FIELDS,
        projection_fields=frozenset({"stop_texts", "stop_text", "detail"}),
        projection_reason=(
            "stop_texts/stop_text can carry boundary text and detail is free diagnostic "
            "text: stop texts are stripped and detail is replaced by the withheld marker "
            "in the default view. Divergence classes (fixed vocabulary) stay."
        ),
        config_allowlist=None,
        docs_anchor="11. Differential backend stop testing (`diff-backends`)",
        cli_flag=None,
    ),
)

INSTRUMENT_BY_KEY: dict[str, Instrument] = {i.key: i for i in INSTRUMENTS}
