"""
Explicit enums for the investigation domain.

Agent decisions are persisted structured state rather than free-form status
strings. Every lifecycle and classification value is a string-backed enum for
storage portability (matches the project's existing ``str, Enum`` convention)
and inspectability.
"""

from __future__ import annotations

from enum import Enum

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class DataSourceKind(str, Enum):
    """Class of system a :class:`DataSource` represents."""

    edgar = "edgar"
    csv = "csv"
    warehouse = "warehouse"
    api = "api"
    fixture = "fixture"
    other = "other"


class DatasetKind(str, Enum):
    """Shape of a dataset described by a :class:`DatasetManifest`."""

    tabular_panel = "tabular_panel"
    timeseries = "timeseries"
    cross_section = "cross_section"
    event_log = "event_log"
    document_set = "document_set"
    other = "other"


class ColumnRole(str, Enum):
    """Semantic role of a column, independent of its storage dtype."""

    entity_id = "entity_id"
    time_index = "time_index"
    metric = "metric"
    dimension = "dimension"
    identifier = "identifier"
    derived = "derived"
    unknown = "unknown"


class SemanticType(str, Enum):
    """
    Interpretation of a column's values, inferred generically (domain-agnostic).

    Units-bearing types (monetary/percentage) cannot be inferred from values
    alone; adapters supply those as hints without the general layer knowing any
    domain vocabulary.
    """

    identifier = "identifier"
    categorical = "categorical"
    integer = "integer"
    real = "real"
    monetary = "monetary"
    percentage = "percentage"
    count = "count"
    temporal = "temporal"
    boolean = "boolean"
    text = "text"
    unknown = "unknown"


class Modality(str, Enum):
    """Coarse consumption mode of a materialized dataset."""

    tabular = "tabular"
    time_series = "time_series"
    document = "document"
    relational = "relational"
    api_records = "api_records"
    mixed = "mixed"


class QualitySeverity(str, Enum):
    """Severity of a data-quality warning."""

    info = "info"
    warning = "warning"
    error = "error"


# ---------------------------------------------------------------------------
# Investigation lifecycle
# ---------------------------------------------------------------------------


class InvestigationStatus(str, Enum):
    """Lifecycle of an :class:`Investigation`."""

    created = "created"
    planning = "planning"
    running = "running"
    awaiting_evidence = "awaiting_evidence"
    converged = "converged"
    exhausted = "exhausted"
    failed = "failed"


class HypothesisStatus(str, Enum):
    """
    Whether accumulated evidence supports a hypothesis.

    Hypotheses can be supported, weakened, or rejected as evidence arrives, so
    these are first-class persisted transitions (see :data:`ALLOWED_HYPOTHESIS_TRANSITIONS`).
    """

    proposed = "proposed"
    active = "active"
    supported = "supported"
    weakened = "weakened"
    rejected = "rejected"
    unresolved = "unresolved"


class ExperimentStatus(str, Enum):
    """Lifecycle of an experiment request/result."""

    planned = "planned"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    skipped = "skipped"


# ---------------------------------------------------------------------------
# Evidence & observations
# ---------------------------------------------------------------------------


class EvidenceType(str, Enum):
    """Kind of observation an :class:`Evidence` record encodes."""

    statistical_test = "statistical_test"
    anomaly_flag = "anomaly_flag"
    descriptive_stat = "descriptive_stat"
    peer_comparison = "peer_comparison"
    trend_break = "trend_break"
    data_quality = "data_quality"
    external_reference = "external_reference"
    model_assertion = "model_assertion"


class EvidenceDirection(str, Enum):
    """How a piece of evidence bears on its target hypothesis."""

    supports = "supports"
    refutes = "refutes"
    neutral = "neutral"


class ObservationType(str, Enum):
    """Kind of raw observation (pre-interpretation)."""

    value = "value"
    outlier = "outlier"
    trend = "trend"
    gap = "gap"
    comparison = "comparison"
    error = "error"


class ReferenceKind(str, Enum):
    """What a :class:`SourceReference` points at."""

    dataset = "dataset"
    dataset_column = "dataset_column"
    artifact = "artifact"
    experiment_result = "experiment_result"
    observation = "observation"
    manifest = "manifest"
    hypothesis = "hypothesis"
    external = "external"
    other = "other"


class PayloadKind(str, Enum):
    """Where the concrete numeric payload behind evidence lives."""

    artifact = "artifact"
    storage_uri = "storage_uri"
    dataset_reference = "dataset_reference"
    inline = "inline"
    external = "external"


# ---------------------------------------------------------------------------
# Decisions, critiques, questions, conclusions
# ---------------------------------------------------------------------------


class EntityKind(str, Enum):
    """Type tag for a cross-entity reference (:class:`EntityRef`)."""

    investigation = "investigation"
    hypothesis = "hypothesis"
    evidence = "evidence"
    observation = "observation"
    experiment_definition = "experiment_definition"
    experiment_request = "experiment_request"
    experiment_result = "experiment_result"
    dataset = "dataset"
    manifest = "manifest"
    artifact = "artifact"
    open_question = "open_question"
    decision = "decision"
    critique = "critique"
    conclusion = "conclusion"


class DecisionType(str, Enum):
    """What kind of agent decision was recorded."""

    propose_hypothesis = "propose_hypothesis"
    select_experiment = "select_experiment"
    update_evidence = "update_evidence"
    revise_confidence = "revise_confidence"
    spawn_sub_hypothesis = "spawn_sub_hypothesis"
    request_critique = "request_critique"
    open_question = "open_question"
    answer_question = "answer_question"
    conclude = "conclude"
    terminate = "terminate"


class CritiqueType(str, Enum):
    """Category of a critic's challenge."""

    insufficient_evidence = "insufficient_evidence"
    confounding = "confounding"
    data_quality = "data_quality"
    overreach = "overreach"
    competing_explanation = "competing_explanation"
    reproducibility = "reproducibility"
    #: Two claims the loop holds as `supported` cannot both be true. Distinct from
    #: `competing_explanation`, which offers an *alternative* account of one claim; this is
    #: the loop disagreeing with itself, and it blocks `sufficient_evidence` outright.
    contradiction = "contradiction"


class CritiqueSeverity(str, Enum):
    """How strongly a critique should influence the run."""

    info = "info"
    minor = "minor"
    major = "major"
    blocking = "blocking"


class OpenQuestionStatus(str, Enum):
    """Lifecycle of an open question."""

    open = "open"
    answered = "answered"
    dropped = "dropped"


class ConclusionDisposition(str, Enum):
    """Overall disposition of an investigation's current conclusion."""

    supported = "supported"
    refuted = "refuted"
    mixed = "mixed"
    """Some claims held and others did not — a multi-claim outcome.

    "Did not hold" covers refuted, weakened and unresolved: what matters to a reader is that
    part of their question came back favourably and part did not. Distinct from
    ``inconclusive``, which describes a single body of equivocal evidence rather than a split
    across claims.

    Reporting such a run as ``supported`` would tell the user their whole question was answered
    favourably when it was not — the overclaiming the agency suite exists to punish.
    """

    inconclusive = "inconclusive"
    insufficient_evidence = "insufficient_evidence"
    unanswerable = "unanswerable"
    """The question cannot be answered from this data at all.

    Not a weaker ``insufficient_evidence``: that one means the analysis ran and did not
    settle the matter, which invites more evidence of the same kind. This one means no amount
    of the *available* data would help, because the thing being asked about is not in it.
    """


# ---------------------------------------------------------------------------
# Provenance & termination
# ---------------------------------------------------------------------------


class ProvenanceSource(str, Enum):
    """Who/what produced a domain entity."""

    agent_llm = "agent_llm"
    deterministic_rule = "deterministic_rule"
    deterministic_tool = "deterministic_tool"
    input_adapter = "input_adapter"
    human = "human"
    system = "system"


class TerminationReason(str, Enum):
    """
    Why an investigation stopped.

    Both sufficient and insufficient evidence are valid terminal outcomes;
    failure and uncertainty are explicit, not implicit.
    """

    sufficient_evidence = "sufficient_evidence"
    insufficient_evidence = "insufficient_evidence"
    unanswerable_premise = "unanswerable_premise"
    """The goal asks about something this dataset does not measure.

    Distinct from ``insufficient_evidence`` on purpose, and the distinction is the whole
    point: "I ran the analysis and it was inconclusive" and "there is nothing here that
    could answer this" are different answers, and only the second tells the user to bring
    different data. Collapsing them let a run substitute the nearest available metric,
    hedge on the result, and report the hedge as a principled decline.
    """
    max_iterations = "max_iterations"
    budget_exhausted = "budget_exhausted"
    no_valid_experiment = "no_valid_experiment"
    repeated_failure = "repeated_failure"
    safety_constraint = "safety_constraint"
    no_progress = "no_progress"
    error = "error"
    user_stop = "user_stop"


class PolicyDecisionKind(str, Enum):
    """Which delegated decision a policy failure happened in — one per :class:`AgentPolicy` method."""

    interpret_goal = "interpret_goal"
    generate_hypotheses = "generate_hypotheses"
    select_experiment = "select_experiment"
    critique = "critique"


class PolicyFailureKind(str, Enum):
    """
    How a policy decision failed.

    ``TerminationReason.error`` alone cannot tell a model that cannot write JSON from a
    provider that never answered, and the two mean opposite things about the model: one is
    its failure, the other is the machine's. Measuring models of different sizes is only
    honest when these are kept apart.
    """

    transport = "transport"
    """The provider never produced a reply (timeout, refused connection, rate limit)."""
    invalid_json = "invalid_json"
    """A reply arrived but was not JSON at all — including an empty reply."""
    schema = "schema"
    """Valid JSON that failed the decision's typed schema."""
    ungrounded = "ungrounded"
    """Schema-valid, but referring to something that does not exist — e.g. a candidate
    index past the end of the list. The loop treats it as declining to choose, so without
    this record it reads as a principled stop."""


# ---------------------------------------------------------------------------
# Allowed transitions (validated by entity mutators)
# ---------------------------------------------------------------------------

#: Directed graph of legal :class:`HypothesisStatus` transitions.
ALLOWED_HYPOTHESIS_TRANSITIONS: dict[HypothesisStatus, frozenset[HypothesisStatus]] = {
    HypothesisStatus.proposed: frozenset(
        {HypothesisStatus.active, HypothesisStatus.rejected}
    ),
    HypothesisStatus.active: frozenset(
        {
            HypothesisStatus.supported,
            HypothesisStatus.weakened,
            HypothesisStatus.rejected,
            HypothesisStatus.unresolved,
        }
    ),
    HypothesisStatus.weakened: frozenset(
        {
            HypothesisStatus.active,
            HypothesisStatus.supported,
            HypothesisStatus.rejected,
            HypothesisStatus.unresolved,
        }
    ),
    HypothesisStatus.supported: frozenset(
        {HypothesisStatus.weakened, HypothesisStatus.unresolved}
    ),
    HypothesisStatus.unresolved: frozenset(
        {
            HypothesisStatus.active,
            HypothesisStatus.supported,
            HypothesisStatus.weakened,
            HypothesisStatus.rejected,
        }
    ),
    # rejected is terminal.
    HypothesisStatus.rejected: frozenset(),
}

#: Directed graph of legal :class:`InvestigationStatus` transitions.
ALLOWED_INVESTIGATION_TRANSITIONS: dict[InvestigationStatus, frozenset[InvestigationStatus]] = {
    InvestigationStatus.created: frozenset(
        {InvestigationStatus.planning, InvestigationStatus.failed}
    ),
    InvestigationStatus.planning: frozenset(
        {InvestigationStatus.running, InvestigationStatus.failed}
    ),
    InvestigationStatus.running: frozenset(
        {
            InvestigationStatus.awaiting_evidence,
            InvestigationStatus.converged,
            InvestigationStatus.exhausted,
            InvestigationStatus.failed,
        }
    ),
    InvestigationStatus.awaiting_evidence: frozenset(
        {
            InvestigationStatus.running,
            InvestigationStatus.converged,
            InvestigationStatus.exhausted,
            InvestigationStatus.failed,
        }
    ),
    # converged / exhausted / failed are terminal.
    InvestigationStatus.converged: frozenset(),
    InvestigationStatus.exhausted: frozenset(),
    InvestigationStatus.failed: frozenset(),
}
