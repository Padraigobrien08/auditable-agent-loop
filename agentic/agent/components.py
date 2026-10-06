"""
The ten explicit investigation-loop components.

Each is a small deterministic class that consumes typed policy decisions
(:mod:`agentic.agent.policy`) and the deterministic experiment registry
(:mod:`agentic.experiments`). The policy interprets/plans; these components and
the registry compute. Every step records an :class:`AgentDecision` into state.
"""

from __future__ import annotations

import math
import threading
from typing import Callable, TypeVar

import pandas as pd

from agentic.domain import (
    AgentDecision,
    Conclusion,
    ConclusionDisposition,
    Critique,
    DecisionType,
    EntityKind,
    EntityRef,
    Evidence,
    EvidenceDirection,
    ExperimentRequest,
    ExperimentResult,
    Hypothesis,
    HypothesisStatus,
    InvestigationState,
    OpenQuestion,
    PolicyDecisionKind,
    PolicyFailure,
    PolicyFailureKind,
    TerminationDecision,
    TerminationReason,
)
from agentic.domain.enums import CritiqueSeverity, CritiqueType, ProvenanceSource
from agentic.domain.manifest import DatasetManifest
from agentic.domain.provenance import Provenance
from agentic.experiments import ArtifactSink, ExperimentContext, ExperimentRegistry, InMemoryArtifactSink
from agentic.experiments.record import ExperimentExecutionRecord

from .alternatives import poses_alternatives
from .budget import BudgetTracker
from .direction import direction_sign
from .ids import DeterministicIds
from .narrative import AllowedFigures, verify_narrative
from .policy import (
    AgentPolicy,
    AnalysisIntent,
    ChallengeReason,
    CritiqueProposal,
    GoalInterpretation,
    drain_policy_cost,
    narrate_answer,
)

# Intent -> ordered candidate tools (general layer). Order encodes priority.
INTENT_TOOLS: dict[AnalysisIntent, list[str]] = {
    AnalysisIntent.trend: ["analyze_time_series_trend", "detect_change_points", "summarize_distribution"],
    AnalysisIntent.comparison: ["compare_groups", "rank_entities"],
    AnalysisIntent.correlation: ["analyze_correlation", "fit_simple_regression"],
    AnalysisIntent.anomaly: ["detect_outliers", "summarize_distribution"],
    AnalysisIntent.ranking: ["rank_entities", "compare_groups"],
    # Second tool for the same reason as `distribution` above: with only `test_association`
    # available, a goal resolving here could state a rival explanation and never run anything
    # against it. `compare_groups` answers the usual one — that the association is really a
    # difference between the groups themselves.
    AnalysisIntent.association: ["test_association", "compare_groups"],
    # `detect_outliers` belongs here, not only under `anomaly`. "Is a small tail dragging the
    # mean up?" is a distribution question whose rival explanation is literally outliers, and
    # without the tool the planner had nothing to raise against that claim — a real run left
    # it at `proposed` after exhausting both other tools, and the loop correctly refused to
    # conclude while an untested alternative stood.
    AnalysisIntent.distribution: ["summarize_distribution", "profile_dataset", "detect_outliers"],
    AnalysisIntent.profile: ["profile_dataset", "summarize_distribution"],
    AnalysisIntent.general: ["profile_dataset", "summarize_distribution"],
}

# EDGAR domain tools added when the dataset is an EDGAR panel.
EDGAR_INTENT_TOOLS: dict[AnalysisIntent, list[str]] = {
    AnalysisIntent.anomaly: ["edgar_revenue_growth_analysis", "edgar_margin_quality_analysis"],
    AnalysisIntent.trend: ["edgar_trend_break_analysis"],
    AnalysisIntent.comparison: ["edgar_peer_comparison"],
}

_T = TypeVar("_T")



def _invoke_policy(tracker: BudgetTracker, policy: AgentPolicy, call: Callable[[], _T]) -> _T:
    """
    Run one policy decision against the run's budget: count it before the call (so a
    policy that raises is still counted) and attribute its cost after, whether or not
    it raised. Policies that don't track cost contribute zero.
    """
    tracker.record_model_call()
    try:
        return call()
    finally:
        tracker.record_model_cost(drain_policy_cost(policy))


def _prov(agent_id: str) -> Provenance:
    return Provenance(source=ProvenanceSource.agent_llm, agent_id=agent_id)


def _quote(statement: str, *, limit: int = 120) -> str:
    """A claim quoted for prose, shortened at a word boundary so it reads as a sentence."""
    text = " ".join(statement.split())
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "…"
    return f"“{text}”"


def _sign(x: float | None) -> int:
    if x is None or not math.isfinite(x) or abs(x) < 1e-9:
        return 0
    return 1 if x > 0 else -1


def is_edgar_manifest(manifest: DatasetManifest) -> bool:
    if manifest.source_identity is not None and manifest.source_identity.adapter_id == "edgar":
        return True
    names = {c.name for c in manifest.columns}
    return {"cik", "period"}.issubset(names)


def expectation_direction(h: Hypothesis) -> int | None:
    """+1 (up), -1 (down), or None (non-directional) parsed from the statement."""
    return direction_sign(h.statement)


# ---------------------------------------------------------------------------
# 1. GoalInterpreter
# ---------------------------------------------------------------------------


class GoalInterpreter:
    def __init__(self, policy: AgentPolicy) -> None:
        self._policy = policy

    def interpret(self, goal_text: str, manifest: DatasetManifest, tracker: BudgetTracker) -> GoalInterpretation:
        entity_column = manifest.entity_id_column()
        summary = {
            "metrics": manifest.metric_names(),
            "dimensions": [c.name for c in manifest.columns if c.role.value == "dimension"],
            # Entities were missing entirely, and their absence reads as *evidence of absence*
            # to anything deciding whether a goal is answerable. An EDGAR panel has no
            # `dimension` columns at all — the units of analysis are tickers in an `entity_id`
            # column — so a goal comparing NVDA to AAPL and MSFT looked like a question about
            # groups the dataset did not have. A model asked to judge answerability declined
            # it, correctly, on the summary it was given.
            "entity_column": entity_column.name if entity_column is not None else None,
            "entities": list(manifest.entities),
        }
        return _invoke_policy(
            tracker, self._policy,
            lambda: self._policy.interpret_goal(goal_text, capability_summary=summary))


# ---------------------------------------------------------------------------
# 2. HypothesisGenerator
# ---------------------------------------------------------------------------


class HypothesisGenerator:
    def __init__(self, policy: AgentPolicy) -> None:
        self._policy = policy

    def generate(
        self, interpretation: GoalInterpretation, state: InvestigationState,
        manifest: DatasetManifest, idgen: DeterministicIds, tracker: BudgetTracker,
    ) -> None:
        dims = [c.name for c in manifest.columns if c.role.value == "dimension"]
        proposals = _invoke_policy(
            tracker, self._policy,
            lambda: self._policy.generate_hypotheses(
                interpretation, metric_names=manifest.metric_names(), dimension_names=dims,
                goal_text=state.objective.objective))
        # A claim may only reference a column the dataset actually has. The policy is asked
        # for a metric name and can return one that merely sounds right ("loyalty_score"),
        # and an unchecked reference does not fail loudly — it falls through to the planner's
        # default metric, so the claim gets tested against whatever was nearest to hand. That
        # is the substitution failure arriving one step later than `answerable=false` catches
        # it. Dropping the reference keeps the claim and its untestability visible.
        known_metrics = set(manifest.metric_names())
        for i, p in enumerate(proposals.hypotheses):
            metric_refs = [p.metric] if p.metric and p.metric in known_metrics else []
            h = Hypothesis(
                id=idgen.make("hyp", i), statement=p.statement, rationale=p.rationale,
                metric_refs=metric_refs, provenance=_prov("hypothesis_generator"),
            )
            state.add_hypothesis(h)
            state.record_decision(AgentDecision(
                id=idgen.make("dec-hyp", i), decision_type=DecisionType.propose_hypothesis,
                rationale=p.rationale or "proposed from goal",
                targets=[EntityRef(kind=EntityKind.hypothesis, id=h.id)], provenance=_prov("hypothesis_generator")))
        # A goal phrased as "is it X, or is it Y?" is asking which, and the question's shape
        # says so without a model reading it. Recording the rivalry now means the loop knows
        # these two cannot both stand before either has been scored — rather than depending on
        # the critic to notice afterwards, which is best-effort and has missed.
        #
        # Exactly two claims, deliberately. Three or more is not the construction this
        # recognises, and guessing which pair is the rival would invent a conflict.
        if poses_alternatives(state.objective.objective) and len(state.hypotheses) == 2:
            first, second = state.hypotheses
            first.mutually_exclusive_with = [second.id]
            second.mutually_exclusive_with = [first.id]

        for j, q in enumerate(proposals.questions):
            state.add_open_question(OpenQuestion(
                id=idgen.make("q", j), question=q,
                related_hypothesis_ids=[h.id for h in state.hypotheses], provenance=_prov("hypothesis_generator")))


# ---------------------------------------------------------------------------
# 3. InvestigationPlanner  (builds validated candidate experiments)
# ---------------------------------------------------------------------------


class InvestigationPlanner:
    def __init__(self, registry: ExperimentRegistry) -> None:
        self._registry = registry

    @staticmethod
    def _metric_for(
        hypothesis: Hypothesis | None, interpretation: GoalInterpretation, manifest: DatasetManifest
    ) -> str | None:
        """
        Which metric an experiment should measure, most specific source first.

        The hypothesis wins: an experiment exists to test *a claim*, and a claim already knows
        its own metric. Reading only ``interpretation.metric_hint`` — one value for the whole
        investigation — is why a second hypothesis over a second metric could never be
        investigated by any policy, however well it reasoned.
        """
        if hypothesis is not None and hypothesis.metric_refs:
            return hypothesis.metric_refs[0]
        metrics = manifest.metric_names()
        return interpretation.metric_hint or (metrics[0] if metrics else None)

    def _params_for(
        self,
        tool: str,
        interpretation: GoalInterpretation,
        manifest: DatasetManifest,
        hypothesis: Hypothesis | None = None,
    ) -> dict:
        metrics = manifest.metric_names()
        metric = self._metric_for(hypothesis, interpretation, manifest)
        dims = [c.name for c in manifest.columns if c.role.value == "dimension"]
        group = interpretation.group_hint or (dims[0] if dims else None)
        entity = manifest.entity_id_column().name if manifest.entity_id_column() else None
        if tool in ("summarize_distribution", "detect_outliers"):
            return {"column": metric} if metric else {}
        if tool == "analyze_time_series_trend":
            return {"value_column": metric} if metric else {}
        if tool == "detect_change_points":
            return {"value_column": metric} if metric else {}
        if tool == "compare_groups":
            return {"value_column": metric, "group_column": group} if metric and group else {}
        if tool == "rank_entities":
            p: dict = {"metric_column": metric} if metric else {}
            if entity:
                p["entity_column"] = entity
            # Which *end* of the ranking answers the goal. ``_RankParams.ascending`` defaults to
            # descending, so a goal asking for the weakest entity was answered with the
            # strongest one until this was plumbed — not a weaker answer, the opposite one.
            if interpretation.direction == "down":
                p["ascending"] = True
            return p
        if tool == "fit_simple_regression":
            return {"x_column": metrics[0], "y_column": metrics[1]} if len(metrics) >= 2 else {}
        if tool == "test_association":
            return {"column_a": dims[0], "column_b": dims[1]} if len(dims) >= 2 else {}
        return {}  # profile_dataset, analyze_correlation, edgar_* take no/auto params

    def _executed_pairs(
        self, state: InvestigationState, interpretation: GoalInterpretation, manifest: DatasetManifest
    ) -> set[tuple[str, str | None]]:
        """``(tool, metric)`` combinations already run, resolved the same way they were planned.

        Reads :attr:`InvestigationState.executed_requests` — the results themselves record only
        a tool name, so they cannot say which claim, or which column, an experiment addressed.
        """
        pairs: set[tuple[str, str | None]] = set()
        for request in state.executed_requests:
            hyp = None
            if request.target_hypothesis_ids:
                hyp = state.find_hypothesis(request.target_hypothesis_ids[0])
            pairs.add((request.tool_name, self._metric_for(hyp, interpretation, manifest)))
        return pairs

    def _target_hypothesis(self, state: InvestigationState, metric: str | None) -> str | None:
        for h in state.open_hypotheses():
            if not h.metric_refs or (metric and metric in h.metric_refs):
                return h.id
        return state.hypotheses[0].id if state.hypotheses else None

    def candidates(
        self, state: InvestigationState, interpretation: GoalInterpretation,
        manifest: DatasetManifest, executed_tools: set[str], tracker: BudgetTracker, idgen: DeterministicIds,
    ) -> list[ExperimentRequest]:
        metric = interpretation.metric_hint or (manifest.metric_names()[0] if manifest.metric_names() else None)
        target = self._target_hypothesis(state, metric)
        tools = list(INTENT_TOOLS.get(interpretation.intent, INTENT_TOOLS[AnalysisIntent.general]))
        if is_edgar_manifest(manifest):
            tools = EDGAR_INTENT_TOOLS.get(interpretation.intent, []) + tools

        out: list[ExperimentRequest] = []
        # Keyed by (tool, metric) rather than tool alone: running one tool against two different
        # metrics answers two different questions, while running it twice against the same
        # metric is the redundancy `executed_tools` was there to prevent.
        seen: set[tuple[str, str | None]] = set()
        done = self._executed_pairs(state, interpretation, manifest)
        n = len(state.pending_experiments) + len(state.completed_experiments) + len(state.failed_experiments)

        # falsification candidates from open critiques (unused tool, targets a hypothesis)
        for c in state.critiques:
            ftool = c.suggested_action
            if not ftool or tracker.tool_at_limit(ftool):
                continue
            challenged = state.find_hypothesis(c.target.id)
            key = (ftool, self._metric_for(challenged, interpretation, manifest))
            if key in done or key in seen:
                continue
            if not self._registry.has(ftool):
                continue
            params = self._params_for(ftool, interpretation, manifest, challenged)
            if not self._registry.get(ftool).validate(params=params, manifest=manifest).ok:
                continue
            out.append(ExperimentRequest(
                id=idgen.make("exp", n + len(out)), tool_name=ftool, parameters=params,
                purpose=f"falsify hypothesis {c.target.id}", definition_id="falsification",
                target_hypothesis_ids=[c.target.id], expected_information_gain=0.95,
                provenance=_prov("investigation_planner")))
            seen.add(key)

        # Every open claim gets candidates, in hypothesis order then tool priority — a pure
        # function of state, never set or dict iteration order, so ids, batching, replay and
        # diff stay deterministic. With a single open hypothesis this yields exactly the list
        # it produced before, in the same order.
        # Claims are grouped by the metric their experiments would measure, because the
        # dedupe key is `(tool, metric)` and two claims about the same metric would otherwise
        # collide: the first took the key, and every later claim silently got nothing and sat
        # at `proposed` forever. That is the normal case, not an edge one — a goal phrased as
        # two competing explanations is two claims about one metric.
        #
        # Grouping rather than duplicating is deliberate. The same tool over the same metric
        # returns the same numbers, so running it twice would buy nothing and double-count
        # the evidence; it runs once and names every claim it bears on, and the evidence
        # updater scores it separately against each.
        #
        # Dict insertion order follows `open_hypotheses()`, so ordering stays a pure function
        # of state — ids, batching, replay and diff remain deterministic.
        grouped: dict[str | None, list[Hypothesis | None]] = {}
        for hypothesis in state.open_hypotheses() or [None]:
            grouped.setdefault(
                self._metric_for(hypothesis, interpretation, manifest), []
            ).append(hypothesis)

        for group_metric, group in grouped.items():
            hyp_targets = [h.id for h in group if h is not None] or ([target] if target else [])
            representative = group[0]
            for prio, tool in enumerate(tools):
                if tracker.tool_at_limit(tool):
                    continue
                key = (tool, group_metric)
                if key in done or key in seen:
                    continue
                if not self._registry.has(tool):
                    continue
                params = self._params_for(tool, interpretation, manifest, representative)
                if not self._registry.get(tool).validate(params=params, manifest=manifest).ok:
                    continue
                gain = round(max(0.3, 0.85 - 0.1 * prio), 4)
                out.append(ExperimentRequest(
                    id=idgen.make("exp", n + len(out)), tool_name=tool, parameters=params,
                    purpose=f"{interpretation.intent.value} analysis via {tool}",
                    target_hypothesis_ids=list(hyp_targets),
                    expected_information_gain=gain,
                    provenance=_prov("investigation_planner")))
                seen.add(key)

        # A request's reproducibility manifest defaults to a random id, which made every
        # candidate differ between two runs of the same seed. Derived from the request id
        # rather than a fresh counter: that id is already deterministic, and tying them
        # together means they cannot drift apart.
        for request in out:
            request.reproducibility.id = f"{request.id}-repro"
        return out


# ---------------------------------------------------------------------------
# 4. ExperimentSelector
# ---------------------------------------------------------------------------


class ExperimentSelector:
    def __init__(self, policy: AgentPolicy) -> None:
        self._policy = policy
        #: Set when the last selection named a candidate that does not exist. The selector
        #: still returns ``None`` for it — the loop's behavior is unchanged — but the loop
        #: copies this onto the termination so the stop is not mistaken for a principled one.
        self.last_fault: PolicyFailure | None = None

    def select(
        self, state: InvestigationState, candidates: list[ExperimentRequest],
        interpretation: GoalInterpretation, tracker: BudgetTracker, idgen: DeterministicIds,
    ) -> ExperimentRequest | None:
        self.last_fault = None
        if not candidates:
            return None
        summaries = [
            {"index": i, "tool_name": c.tool_name, "purpose": c.purpose,
             "expected_information_gain": c.expected_information_gain,
             "falsification": c.definition_id == "falsification"}
            for i, c in enumerate(candidates)
        ]
        choice = _invoke_policy(
            tracker, self._policy,
            lambda: self._policy.select_experiment(
                goal_summary={"intent": interpretation.intent.value}, candidates=summaries))
        if choice.request_index is None:
            return None
        if not (0 <= choice.request_index < len(candidates)):
            self.last_fault = PolicyFailure(
                decision=PolicyDecisionKind.select_experiment, kind=PolicyFailureKind.ungrounded,
                detail=f"request_index {choice.request_index} outside 0..{len(candidates) - 1}")
            return None
        chosen = candidates[choice.request_index]
        self._accept(state, chosen, idgen, choice.rationale or "selected next experiment")
        return chosen

    def select_batch(
        self, state: InvestigationState, candidates: list[ExperimentRequest],
        interpretation: GoalInterpretation, tracker: BudgetTracker, idgen: DeterministicIds,
        *, limit: int,
    ) -> list[ExperimentRequest]:
        """
        Select up to ``limit`` experiments to run together, in execution order.

        The **lead** experiment is chosen by the policy exactly as in :meth:`select` — one
        model call per iteration, unchanged. Any remaining slots are filled deterministically
        from the planner's ranked candidates, which already encode priority order.

        Filling deterministically rather than asking the policy per slot is deliberate: it
        keeps ``AgentPolicy`` a four-method contract, keeps model calls at one per iteration
        (so a wider batch *lowers* cost per experiment), and keeps the batch reproducible.
        The trade-off is that followers are picked without seeing the lead's result — which
        is exactly why the batch is bounded.
        """
        if limit <= 1:
            lead = self.select(state, candidates, interpretation, tracker, idgen)
            return [lead] if lead is not None else []

        lead = self.select(state, candidates, interpretation, tracker, idgen)
        if lead is None:
            return []
        batch = [lead]
        for candidate in candidates:
            if len(batch) >= limit:
                break
            if candidate.id == lead.id or candidate.tool_name in {b.tool_name for b in batch}:
                continue
            self._accept(state, candidate, idgen, "batched alongside the selected experiment")
            batch.append(candidate)
        return batch

    def _accept(
        self, state: InvestigationState, chosen: ExperimentRequest, idgen: DeterministicIds,
        rationale: str,
    ) -> None:
        state.add_experiment_request(chosen)
        state.record_decision(AgentDecision(
            id=idgen.make("dec-sel", len(state.decisions)), decision_type=DecisionType.select_experiment,
            rationale=rationale,
            targets=[EntityRef(kind=EntityKind.experiment_request, id=chosen.id)],
            chosen_option=chosen.tool_name, provenance=_prov("experiment_selector")))


# ---------------------------------------------------------------------------
# 5. ExperimentExecutor
# ---------------------------------------------------------------------------


class LockedArtifactSink(ArtifactSink):
    """
    Serializes emission into a shared sink so a parallel batch can share one safely.

    Only emission is serialized — the analysis itself still runs concurrently. The order
    of records within the shared sink is therefore completion order and is deliberately
    not part of the contract: artifacts are looked up by id, never by position.
    """

    def __init__(self, inner: ArtifactSink) -> None:
        self._inner = inner
        self._lock = threading.Lock()

    def emit_table(self, name: str, frame: pd.DataFrame):
        with self._lock:
            return self._inner.emit_table(name, frame)

    def emit_json(self, name: str, obj):
        with self._lock:
            return self._inner.emit_json(name, obj)

    def emit_chart(self, name: str, spec: dict):
        with self._lock:
            return self._inner.emit_chart(name, spec)


class ExperimentExecutor:
    def __init__(self, registry: ExperimentRegistry, *, artifact_sink: ArtifactSink | None = None) -> None:
        self._registry = registry
        # When a shared sink is supplied (e.g. by the backend), every experiment emits into
        # it so the emitted bytes survive the loop and can be ingested + linked afterwards.
        # When absent, each experiment gets a throwaway sink (unchanged in-process behavior).
        self._artifact_sink = artifact_sink

    def execute(
        self, request: ExperimentRequest, manifest: DatasetManifest, frame: pd.DataFrame | None,
        idgen: DeterministicIds, state: InvestigationState,
    ) -> tuple[ExperimentExecutionRecord, ExperimentResult]:
        """Run one experiment and fold it into state (the sequential path)."""
        record = self.run(request, manifest, frame)
        return record, self.record(record, request, idgen, state)

    def run(
        self, request: ExperimentRequest, manifest: DatasetManifest, frame: pd.DataFrame | None,
        *, sink: ArtifactSink | None = None,
    ) -> ExperimentExecutionRecord:
        """
        Run the deterministic tool. **Pure with respect to investigation state**, so a batch
        of experiments can run concurrently; ``sink`` lets each one collect artifacts into its
        own buffer rather than racing on the shared one.

        Tools are read-only over ``frame``; they must not mutate it, or concurrent execution
        would not be safe.
        """
        target = sink if sink is not None else self._artifact_sink
        ctx = ExperimentContext(
            manifest=manifest, frame=frame, raw_params=dict(request.parameters),
            artifact_sink=target if target is not None else InMemoryArtifactSink(),
            request_id=request.id)
        return self._registry.get(request.tool_name).run(ctx)

    def record(
        self, record: ExperimentExecutionRecord, request: ExperimentRequest,
        idgen: DeterministicIds, state: InvestigationState,
    ) -> ExperimentResult:
        """Fold one finished experiment into state. Always called in selection order, so
        result ids stay a pure function of state regardless of completion order.

        Returns the folded result because the evidence updater runs next and needs this
        result's id: evidence is what links a claim back to the computation behind it, and
        that link can only be written once the result has its deterministic id.
        """
        result = record.to_domain_result()
        # deterministic ids for the persisted result/observations
        ordinal = len(state.completed_experiments) + len(state.failed_experiments)
        result.id = idgen.make("res", ordinal)
        # Everything the tool minted with a random id, re-stamped as part of *this* run. The
        # tool is standalone and cannot know the investigation it is serving, so it defaults
        # to uuid4 — correct there, and the reason two runs of the same seed over the same
        # bytes used to differ in every observation, artifact and reproducibility id.
        for i, observation in enumerate(result.observations):
            observation.id = idgen.make(f"obs-{ordinal}", i)
            observation.experiment_result_id = result.id
        result.reproducibility.id = idgen.make("repro", ordinal)
        # Artifact ids are deliberately *not* re-stamped here. They are content-addressed at
        # emission (see `ArtifactRecord`), which makes them reproducible already, and the
        # sink keys its bytes by the same value — renaming them here would orphan every
        # artifact from the content it names.
        # The tool minted its own evidence ids before the loop ever saw them; the evidence
        # that reaches state is re-minted under `idgen` by the evidence updater. Carrying the
        # tool's ids here would publish a list that resolves against nothing, so the link is
        # left empty and filled in by the updater once the state-side ids exist.
        result.produced_evidence_ids = []
        request.status = record.status
        state.record_experiment_result(result)
        return result

    @property
    def shared_sink(self) -> ArtifactSink | None:
        return self._artifact_sink


# ---------------------------------------------------------------------------
# 6. EvidenceUpdater
# ---------------------------------------------------------------------------


class EvidenceUpdater:
    def update(
        self, state: InvestigationState, record: ExperimentExecutionRecord, request: ExperimentRequest,
        idgen: DeterministicIds, result: ExperimentResult | None = None,
    ) -> list[Evidence]:
        """
        File one experiment's evidence into state, linked to the result that produced it.

        ``result`` is the folded :class:`ExperimentResult` returned by
        :meth:`ExperimentExecutor.record`. It is what makes a claim traceable: without it an
        evidence row states a number with no way back to the computation behind it, which is
        the one thing this system is not allowed to do. It is optional only so a caller
        holding a bare record can still score evidence in isolation; the loop always passes it.
        """
        # Every claim the experiment was raised to test, not just the first.
        #
        # One measurement can bear on two rival explanations at once, and it does not bear on
        # them the same way: `expectation_direction` is a property of the hypothesis, so the
        # slope that supports "this is a sustained change" is the slope that refutes "this is
        # a few extreme quarters". Reading only `target_hypothesis_ids[0]` scored the first
        # claim and left the second with no evidence at all — which is how a rival stayed at
        # `proposed` while every tool in its intent had already run.
        targets: list[str | None] = [
            t for t in request.target_hypothesis_ids if state.find_hypothesis(t) is not None
        ] or [None]
        record_signal = self._signal_sign(record)

        produced: list[Evidence] = []
        for e, target in [(e, t) for e in record.evidence for t in targets]:
            hyp = state.find_hypothesis(target) if target else None
            expected = expectation_direction(hyp) if hyp is not None else None
            if expected is not None and e.direction is EvidenceDirection.neutral:
                # The tool's own verdict that it found nothing comes first. A shift or slope
                # always has *a* sign, and reading support off that sign turned a change point
                # the tool had judged insignificant, or a trend fit with R² = 0.01, into
                # evidence for whichever direction the claim happened to name.
                direction = EvidenceDirection.neutral
            elif expected is not None:
                # Prefer the evidence item's own directional statistic (e.g. per-entity
                # slope), so opposing signals in one experiment yield contradictory evidence.
                signal = self._evidence_sign(e)
                if signal == 0:
                    signal = record_signal
                if signal == 0:
                    direction = EvidenceDirection.neutral
                elif signal * expected > 0:
                    direction = EvidenceDirection.supports
                else:
                    direction = EvidenceDirection.refutes
            else:
                direction = EvidenceDirection.supports if e.direction == EvidenceDirection.supports else EvidenceDirection.neutral
            ev = Evidence(
                id=idgen.make("evd", len(state.evidence)), evidence_type=e.evidence_type,
                source_reference=e.source_reference,
                experiment_result_id=result.id if result is not None else None,
                hypothesis_ids=[target] if target else [], claim=e.claim, direction=direction,
                strength=e.strength, reliability=e.reliability, coverage=e.coverage,
                statistics=e.statistics, provenance=_prov("evidence_updater"))
            state.add_evidence(ev)
            produced.append(ev)
        # The link written from both ends, over the same id space. `experiment_result_id`
        # answers "what computed this?" from a claim; `produced_evidence_ids` answers "what
        # did this establish?" from a result. A reader arriving from either direction lands
        # on a row that exists — which is the whole of the traceability guarantee.
        if result is not None:
            result.produced_evidence_ids = [e.id for e in produced]

        named = [t for t in targets if t]
        if produced and named:
            # One decision naming every claim the evidence was filed against, so the trace
            # shows a shared experiment as the single act it was rather than one act per claim.
            state.record_decision(AgentDecision(
                id=idgen.make("dec-evd", len(state.decisions)), decision_type=DecisionType.update_evidence,
                rationale=f"recorded {len(produced)} evidence item(s) for {request.tool_name}",
                targets=[EntityRef(kind=EntityKind.hypothesis, id=t) for t in named],
                provenance=_prov("evidence_updater")))
        return produced

    @staticmethod
    def _evidence_sign(e: Evidence) -> int:
        if e.statistics is not None:
            for key in ("slope", "shift", "mean_diff"):
                if key in e.statistics.diagnostics:
                    return _sign(e.statistics.diagnostics[key])
        return 0

    @staticmethod
    def _signal_sign(record: ExperimentExecutionRecord) -> int:
        for key in ("slope", "mean_slope", "shift", "mean_diff", "effect_size"):
            if key in record.metrics:
                return _sign(record.metrics[key])
        for s in record.statistics:
            for key in ("slope", "shift", "mean_diff"):
                if key in s.diagnostics:
                    return _sign(s.diagnostics[key])
        return 0


# ---------------------------------------------------------------------------
# 7. HypothesisUpdater
# ---------------------------------------------------------------------------


class HypothesisUpdater:
    SUPPORT_THRESHOLD = 0.5

    def update(self, state: InvestigationState, request: ExperimentRequest, idgen: DeterministicIds) -> None:
        # Score every claim the experiment was raised to test. Reading only the first was the
        # last of three places that did so: the planner then named both claims and the
        # evidence updater filed against both, but the rival was still never scored, so it
        # sat at `proposed` and the loop kept declining with an untested alternative standing.
        for target in request.target_hypothesis_ids or []:
            self._score(state, target, request, idgen)

    def _score(
        self, state: InvestigationState, target: str, request: ExperimentRequest,
        idgen: DeterministicIds,
    ) -> None:
        h = state.find_hypothesis(target)
        if h is None or h.is_terminal():
            return
        if h.status is HypothesisStatus.proposed:
            h.set_status(HypothesisStatus.active)

        ev = state.evidence_for(h.id)
        supports = [e for e in ev if e.direction is EvidenceDirection.supports]
        refutes = [e for e in ev if e.direction is EvidenceDirection.refutes]
        s_str = max((e.strength for e in supports), default=0.0)
        r_str = max((e.strength for e in refutes), default=0.0)

        if supports and refutes:
            h.set_status(HypothesisStatus.weakened)
            h.set_confidence(round(max(0.25, 0.5 + 0.2 * (s_str - r_str)), 4))
        elif supports and not refutes:
            if s_str >= self.SUPPORT_THRESHOLD:
                h.set_status(HypothesisStatus.supported)
                h.set_confidence(round(min(0.95, 0.55 + s_str / 2), 4))
            else:
                h.set_confidence(round(0.45 + s_str / 4, 4))
        elif refutes and not supports:
            if r_str >= self.SUPPORT_THRESHOLD:
                h.set_status(HypothesisStatus.rejected)
                h.set_confidence(round(max(0.05, 0.4 - r_str / 2), 4))
            else:
                h.set_status(HypothesisStatus.weakened)
                h.set_confidence(round(max(0.2, 0.45 - r_str / 4), 4))
        state.record_decision(AgentDecision(
            id=idgen.make("dec-rev", len(state.decisions)), decision_type=DecisionType.revise_confidence,
            rationale=f"{h.status.value} (support={len(supports)}, refute={len(refutes)})",
            targets=[EntityRef(kind=EntityKind.hypothesis, id=h.id)], provenance=_prov("hypothesis_updater")))

        # follow-up question when evidence is present but weak
        if h.status is HypothesisStatus.active and 0.0 < s_str < self.SUPPORT_THRESHOLD:
            state.add_open_question(OpenQuestion(
                id=idgen.make("q-follow", len(state.open_questions)),
                question=f"Stronger evidence needed for: {h.statement}",
                related_hypothesis_ids=[h.id], provenance=_prov("hypothesis_updater")))


# ---------------------------------------------------------------------------
# 8. Critic
# ---------------------------------------------------------------------------


class Critic:
    #: Ceiling applied to both sides of a contradiction. Two claims that cannot both hold
    #: must not sit at high confidence while the conflict is open; capping rather than
    #: zeroing keeps whatever evidence each has, because the conflict says one of them is
    #: wrong, not which one.
    CONTRADICTION_CONFIDENCE_CAP = 0.5

    def __init__(self, policy: AgentPolicy) -> None:
        self._policy = policy

    def challenge(
        self, state: InvestigationState, interpretation: GoalInterpretation, manifest: DatasetManifest,
        executed_tools: set[str], tracker: BudgetTracker, idgen: DeterministicIds,
    ) -> None:
        asked = self._claim_to_challenge(state)
        if asked is None:
            return
        h, reason = asked
        already = any(c.target.id == h.id for c in state.critiques)
        tools = list(INTENT_TOOLS.get(interpretation.intent, []))
        if is_edgar_manifest(manifest):
            tools = EDGAR_INTENT_TOOLS.get(interpretation.intent, []) + tools
        available = [t for t in tools if t not in executed_tools and self._registry_ok(t)]
        # Every supported claim, so the policy can see that two of them disagree. Nothing
        # else in the loop compares claims to each other: the hypothesis updater scores each
        # one against its own evidence, which is exactly how a claim and its negation both
        # reached `supported`.
        supported_claims = [
            {"id": x.id, "statement": x.statement, "confidence": x.confidence}
            for x in state.hypotheses
            if x.status is HypothesisStatus.supported
        ]
        proposal = _invoke_policy(
            tracker, self._policy,
            lambda: self._policy.critique(
                strongest_claim={"id": h.id, "status": h.status.value, "confidence": h.confidence,
                                 "already_critiqued": already,
                                 # Carried inside the claim rather than as a new argument: the
                                 # caller always passes it, so a new keyword would break every
                                 # existing `AgentPolicy` implementation rather than defaulting
                                 # for it. It is a fact about this claim's evidence anyway.
                                 "challenge_reason": reason.value},
                available_tools=available,
                supported_claims=supported_claims))

        if self._record_contradiction(state, proposal, idgen, asked_about=h):
            return

        if not proposal.should_challenge or not proposal.target_hypothesis_id or not proposal.falsification_tool:
            return
        state.add_critique(Critique(
            id=idgen.make("crit", len(state.critiques)), critique_type=CritiqueType.competing_explanation,
            severity=CritiqueSeverity.major, target=EntityRef(kind=EntityKind.hypothesis, id=proposal.target_hypothesis_id),
            message=proposal.message, suggested_action=proposal.falsification_tool, provenance=_prov("critic")))
        state.record_decision(AgentDecision(
            id=idgen.make("dec-crit", len(state.decisions)), decision_type=DecisionType.request_critique,
            rationale=proposal.rationale or "challenge strongest claim",
            targets=[EntityRef(kind=EntityKind.hypothesis, id=proposal.target_hypothesis_id)],
            chosen_option=proposal.falsification_tool, provenance=_prov("critic")))

    def _record_contradiction(
        self, state: InvestigationState, proposal: CritiqueProposal, idgen: DeterministicIds,
        *, asked_about: Hypothesis,
    ) -> bool:
        """
        Act on a reported conflict between two supported claims. True when one was recorded.

        The model supplies only the judgement that two statements are mutually exclusive.
        Everything that follows is computed here: which claims are affected, what status they
        take, and what confidence they carry. Both sides are weakened rather than one being
        picked, because the conflict establishes that they cannot both hold — not which of
        them is wrong. Choosing a winner on the model's say-so would be the loop asserting a
        finding no evidence supports.
        """
        other_id = proposal.contradicts_hypothesis_id
        # A contradiction is reported independently of an ordinary challenge, and in practice
        # arrives *with* a decline: once the claim has been critiqued and no tools are left,
        # the policy answers `should_challenge: false` with a null target and still names the
        # conflict. Requiring a target here dropped exactly the report this exists to catch.
        # The claim the critic was asked about is the other side of the pair by construction.
        target_id = proposal.target_hypothesis_id or asked_about.id
        if not other_id or other_id == target_id:
            return False

        target = state.find_hypothesis(target_id)
        other = state.find_hypothesis(other_id)
        # Only a live conflict counts. A claim already rejected or weakened is not being
        # asserted, so there is nothing to contradict.
        if target is None or other is None:
            return False
        # Both must still be asserted. This also dedupes: recording weakens the pair, so a
        # later call cannot re-fire on it unless fresh evidence restores both to supported —
        # at which point recording again is the right behaviour.
        if target.status is not HypothesisStatus.supported or other.status is not HypothesisStatus.supported:
            return False

        state.add_critique(Critique(
            id=idgen.make("crit", len(state.critiques)),
            critique_type=CritiqueType.contradiction,
            severity=CritiqueSeverity.major,
            target=EntityRef(kind=EntityKind.hypothesis, id=target.id),
            conflicts_with_id=other.id,
            message=proposal.message or (
                f"'{target.statement}' and '{other.statement}' cannot both hold; "
                f"the evidence gathered so far supports each independently."
            ),
            suggested_action=proposal.falsification_tool,
            provenance=_prov("critic"),
        ))

        for claim in (target, other):
            conflicting = other if claim is target else target
            claim.set_status(HypothesisStatus.weakened)
            claim.set_confidence(round(min(claim.confidence, self.CONTRADICTION_CONFIDENCE_CAP), 4))
            state.record_decision(AgentDecision(
                id=idgen.make("dec-contra", len(state.decisions)),
                decision_type=DecisionType.revise_confidence,
                # The id of the conflicting claim goes in `targets`, where a reader can follow
                # it and a client can link it. A rationale is prose and is read as prose, so
                # naming the claim beats printing its primary key.
                rationale=f"weakened: cannot hold at the same time as {_quote(conflicting.statement)}",
                targets=[
                    EntityRef(kind=EntityKind.hypothesis, id=claim.id),
                    EntityRef(kind=EntityKind.hypothesis, id=conflicting.id),
                ],
                provenance=_prov("critic"),
            ))

        state.add_open_question(OpenQuestion(
            id=idgen.make("q-contra", len(state.open_questions)),
            question=f"Which of these holds, if either: '{target.statement}' or '{other.statement}'?",
            related_hypothesis_ids=[target.id, other.id],
            provenance=_prov("critic"),
        ))
        return True

    @staticmethod
    def _claim_to_challenge(
        state: InvestigationState,
    ) -> tuple[Hypothesis, ChallengeReason] | None:
        """
        The claim most worth challenging and why, or ``None`` when nothing is.

        Four tiers, most urgent first, each a shape the run's own evidence has actually taken.
        The reason is returned rather than inferred downstream because it decides what a useful
        challenge even is — "you may be overconfident" and "your methods are not separating
        anything" are different questions with different right answers.

        A ``supported`` claim comes first: guarding against false confidence is the critic's
        oldest job, and it is what the agency suite's ``challenges_before_concluding`` property
        measures.

        The three below it exist because the gate used to stop at supported-or-mixed, and the
        published corpus shows what that cost. Of five runs that ran any experiment, three
        never consulted the critic at all — including the flagship, which ran seven experiments,
        rejected both of its claims and concluded `refuted` without one. The two shapes it could
        not see are the two where the run has *failed to learn something*:

        - every measurement came back neutral, so the run measured three times and separated
          nothing, then declined;
        - evidence refutes the claim and nothing supports it, and it has not been dropped yet.

        Both are worth a second reading, and neither is "false confidence" — which is why the
        reason has to travel with the ask rather than the critic guessing from a status.

        What this still cannot reach is the flagship's shape: a run whose claims are all
        *rejected*, where the useful question is "what explanation did you never propose?".
        A critique produces a falsification experiment aimed at an existing hypothesis, so the
        loop can only ever test the claims it already holds. Widening the gate further would
        not fix that; a different mechanism would.

        A claim with **no** evidence is still left alone: nothing has been run against it, so a
        critique would be a note rather than a challenge. So is one whose evidence only
        supports it without having reached ``supported`` — the first tier catches that on
        promotion, and challenging it earlier would fire every iteration of a healthy run.
        """
        supported = [h for h in state.hypotheses if h.status is HypothesisStatus.supported]
        if supported:
            return max(supported, key=lambda x: x.confidence), ChallengeReason.false_confidence

        conflicting: list[Hypothesis] = []
        undiscriminating: list[Hypothesis] = []
        unexplained: list[Hypothesis] = []
        for h in state.hypotheses:
            # Terminal claims are excluded on purpose. A critique's suggested tool becomes a
            # falsification candidate *targeting that claim*, and spending an experiment to
            # further disconfirm something already rejected buys nothing.
            if h.status in (HypothesisStatus.rejected, HypothesisStatus.unresolved):
                continue
            directions = {e.direction for e in state.evidence if h.id in e.hypothesis_ids}
            if not directions:
                continue
            supports = EvidenceDirection.supports in directions
            refutes = EvidenceDirection.refutes in directions
            if supports and refutes:
                conflicting.append(h)
            elif not supports and not refutes:
                undiscriminating.append(h)
            elif refutes:
                unexplained.append(h)

        for tier, reason in (
            (conflicting, ChallengeReason.conflicting_evidence),
            (undiscriminating, ChallengeReason.undiscriminating_evidence),
            (unexplained, ChallengeReason.unexplained_refutation),
        ):
            if tier:
                return max(tier, key=lambda x: x.confidence), reason
        return None

    def _registry_ok(self, _tool: str) -> bool:
        return True


# ---------------------------------------------------------------------------
# 9. TerminationPolicy
# ---------------------------------------------------------------------------


class TerminationPolicy:
    SUFFICIENT_CONFIDENCE = 0.6

    def decide(
        self, state: InvestigationState, tracker: BudgetTracker, iterations: int, *,
        executed_tools: set[str], intent_tools: list[str], user_stop: bool,
    ) -> tuple[bool, TerminationReason | None]:
        if user_stop or tracker.user_stop_requested:
            return True, TerminationReason.user_stop
        if tracker.safety_violated(iterations):
            return True, TerminationReason.safety_constraint
        if tracker.repeated_failure():
            return True, TerminationReason.repeated_failure
        if tracker.budget_exhausted():
            return True, TerminationReason.budget_exhausted
        # Sufficiency means the *investigation* is done, not that one claim landed. Firing on the
        # first supported hypothesis leaves every other claim stranded at `proposed` — the same
        # single-metric bug this phase removes, relocated one step later.
        #
        # The bar is "nothing is still `proposed`", not `is_terminal()`. Only `rejected` is
        # terminal in the transition graph — a supported claim may still be weakened — so
        # requiring terminality would mean requiring every claim to be *rejected*. A claim past
        # `proposed` has had evidence brought to bear on it; one still at `proposed` has had
        # nothing run against it at all.
        if any(h.status is HypothesisStatus.proposed for h in state.hypotheses):
            return False, None

        # A live contradiction is disqualifying on its own. The loop is holding two claims
        # that cannot both be true, so whatever else it has established, it has not
        # established this — and reporting `sufficient_evidence` here would publish a
        # conclusion the run's own record refutes. Run the discriminating experiment if the
        # critic named one and it has not been tried; otherwise say so plainly.
        contradiction = self._unresolved_contradiction(state)
        if contradiction is not None:
            suggested = contradiction.suggested_action or ""
            if suggested and suggested not in executed_tools:
                return False, None
            return True, TerminationReason.insufficient_evidence

        supported = [h for h in state.hypotheses
                     if h.status is HypothesisStatus.supported and h.confidence >= self.SUFFICIENT_CONFIDENCE]
        if supported:
            # Every supported claim must have been challenged, or have no challenge left to run.
            # One claim surviving a falsification says nothing about the others.
            unused = [t for t in intent_tools if t not in executed_tools]
            for h in supported:
                crits = [c for c in state.critiques if c.target.id == h.id]
                tested = any((c.suggested_action or "") in executed_tools for c in crits)
                if not tested and unused:
                    return False, None
            return True, TerminationReason.sufficient_evidence
        return False, None

    @staticmethod
    def _unresolved_contradiction(state: InvestigationState) -> Critique | None:
        return next(iter(open_contradictions(state)), None)

    def finalize_no_candidates(self, state: InvestigationState, ran_any: bool) -> TerminationReason:
        # Checked before `supported`: running out of experiments does not settle a
        # contradiction, and a third claim standing does not make the conflicting pair go away.
        if self._unresolved_contradiction(state) is not None:
            return TerminationReason.insufficient_evidence
        # Same bar `decide` applies: a claim still at `proposed` has had nothing run against
        # it. Without this the two termination paths disagreed, and running out of candidate
        # experiments could report `sufficient_evidence` with a rival explanation untested —
        # a real run concluded "a genuine change rather than a seasonal artifact" at 0.95
        # while the seasonality claim it raised was never examined. Ruling out an alternative
        # the loop never tested is exactly the overreach this system exists to not commit.
        if any(h.status is HypothesisStatus.proposed for h in state.hypotheses):
            return TerminationReason.insufficient_evidence if ran_any else TerminationReason.no_valid_experiment
        supported = [h for h in state.hypotheses
                     if h.status is HypothesisStatus.supported and h.confidence >= self.SUFFICIENT_CONFIDENCE]
        if supported:
            return TerminationReason.sufficient_evidence
        if ran_any:
            return TerminationReason.insufficient_evidence
        return TerminationReason.no_valid_experiment


# ---------------------------------------------------------------------------
# 10. ConclusionSynthesizer
# ---------------------------------------------------------------------------


class ConclusionSynthesizer:
    def synthesize(
        self,
        state: InvestigationState,
        reason: TerminationReason,
        idgen: DeterministicIds,
        *,
        policy: object | None = None,
        question: str = "",
        tracker: BudgetTracker | None = None,
    ) -> Conclusion:
        supported = [h for h in state.hypotheses if h.status is HypothesisStatus.supported]
        rejected = [h for h in state.hypotheses if h.status is HypothesisStatus.rejected]
        weakened = [h for h in state.hypotheses if h.status is HypothesisStatus.weakened]

        # any still-active hypothesis is left explicitly unresolved
        for h in state.hypotheses:
            if h.status is HypothesisStatus.active:
                h.set_status(HypothesisStatus.unresolved)

        contradicting = [e.id for e in state.evidence if e.direction is EvidenceDirection.refutes]
        # `unresolved` counts as not-held. It is not a refutation, but reporting a run as
        # "supported" when half its claims could not be determined tells the user their whole
        # question was answered favourably, which is the more misleading of the two errors.
        unresolved = [h for h in state.hypotheses if h.status is HypothesisStatus.unresolved]
        opposed = rejected + weakened + unresolved
        if supported and opposed:
            # Checked before the `supported` branch on purpose. Falling through to it would
            # report a run that found one thing true and another false as simply "supported",
            # dropping the refutation from the headline and averaging its confidence away.
            disposition = ConclusionDisposition.mixed
            conf = round(
                sum(h.confidence for h in supported + opposed) / len(supported + opposed), 4
            )
            statement = (
                "Supported: " + "; ".join(h.statement for h in supported)
                + " | Not supported: " + "; ".join(h.statement for h in opposed)
            )
            hyp_ids = [h.id for h in supported + opposed]
            key_ev = [e.id for e in state.evidence]
        elif supported:
            disposition = ConclusionDisposition.supported
            conf = round(sum(h.confidence for h in supported) / len(supported), 4)
            statement = "; ".join(h.statement for h in supported)
            hyp_ids = [h.id for h in supported]
            key_ev = [e.id for e in state.evidence if e.direction is EvidenceDirection.supports]
        elif rejected and not weakened:
            disposition = ConclusionDisposition.refuted
            conf = 0.6
            statement = "Rejected: " + "; ".join(h.statement for h in rejected)
            hyp_ids = [h.id for h in rejected]
            key_ev = contradicting
        elif weakened:
            disposition = ConclusionDisposition.inconclusive
            conf = 0.4
            statement = "Mixed evidence for: " + "; ".join(h.statement for h in weakened)
            hyp_ids = [h.id for h in weakened]
            key_ev = [e.id for e in state.evidence]
        elif reason is TerminationReason.unanswerable_premise:
            # Checked before the generic `insufficient_evidence` fallback and reported as its
            # own disposition. "The analysis was inconclusive" invites the user to run more of
            # it; "this data cannot answer that" tells them to bring different data. Reporting
            # the second as the first is the substitution failure wearing a hedge.
            disposition = ConclusionDisposition.unanswerable
            conf = 0.0
            statement = (
                "This dataset cannot answer the question as asked: "
                f"{_unanswerable_detail(state)}"
            )
            hyp_ids = []
            key_ev = []
        else:
            disposition = ConclusionDisposition.insufficient_evidence
            conf = 0.2
            statement = "Insufficient evidence to resolve the investigation goal."
            hyp_ids = [h.id for h in state.hypotheses]
            key_ev = []

        caveats = [f"termination: {reason.value}"]
        if contradicting:
            caveats.append(f"{len(contradicting)} contradicting evidence item(s) preserved")
        if state.unresolved_questions():
            caveats.append(f"{len(state.unresolved_questions())} open question(s) remain")

        conclusion = Conclusion(
            id=idgen.make("concl", 0), statement=statement, disposition=disposition, confidence=conf,
            narrative=self._narrate(state, policy, question, disposition, conf, reason, tracker),
            supporting_hypothesis_ids=hyp_ids, key_evidence_ids=key_ev, caveats=caveats,
            open_question_ids=[q.id for q in state.open_questions], provenance=_prov("conclusion_synthesizer"))
        state.set_conclusion(conclusion)
        state.record_decision(AgentDecision(
            id=idgen.make("dec-concl", len(state.decisions)), decision_type=DecisionType.conclude,
            rationale=f"{disposition.value} ({reason.value})",
            targets=[EntityRef(kind=EntityKind.conclusion, id=conclusion.id)], provenance=_prov("conclusion_synthesizer")))
        return conclusion

    @staticmethod
    def _narrate(
        state: InvestigationState,
        policy: object | None,
        question: str,
        disposition: ConclusionDisposition,
        confidence: float,
        reason: TerminationReason,
        tracker: BudgetTracker | None,
    ) -> str | None:
        """
        The finding as prose, when a policy can write one that survives verification.

        The findings handed over are already-computed values, and the check on the way back
        is what matters: a figure the run never recorded discards the whole narrative and
        the caller keeps its deterministic statement. Returning ``None`` is an ordinary
        outcome here, not a failure.
        """
        if policy is None:
            return None

        # Per-claim evidence, split by direction. A writer asked for more than a couple of
        # sentences needs something true to say in them; without this the only material is
        # the claim statements, and the extra length turns into padding or into figures the
        # run never produced.
        # Seeded at zero for every claim, not just the ones with evidence. A claim with
        # nothing against it genuinely has *zero* refuting items, and that is a fact the run
        # holds — leaving it out of the allowed set made "no refuting evidence" unsayable.
        supporting: dict[str, int] = {h.id: 0 for h in state.hypotheses}
        refuting: dict[str, int] = {h.id: 0 for h in state.hypotheses}
        for e in state.evidence:
            bucket = refuting if e.direction is EvidenceDirection.refutes else supporting
            for hid in e.hypothesis_ids:
                if hid in bucket:
                    bucket[hid] += 1

        claims = [
            {
                "statement": h.statement,
                "status": h.status.value,
                "confidence": h.confidence,
                "supporting_evidence": supporting.get(h.id, 0),
                "refuting_evidence": refuting.get(h.id, 0),
            }
            for h in state.hypotheses
        ]
        counts = {
            "hypotheses": len(state.hypotheses),
            "evidence": len(state.evidence),
            "experiments": len(state.completed_experiments),
            "supported": sum(1 for h in state.hypotheses if h.status is HypothesisStatus.supported),
            "open_questions": len(state.open_questions),
        }
        dataset = state.datasets[0] if state.datasets else None
        findings = {
            "disposition": disposition.value,
            "confidence": confidence,
            "claims": claims,
            "counts": counts,
            # What was actually run, and what the run refused to settle. Both are things a
            # reader wants and neither can be inferred from the claims alone.
            "experiments_run": [x.tool_name for x in state.completed_experiments],
            "stopped_because": reason.value,
            "open_questions": [q.question for q in state.open_questions],
            "dataset": (
                {"name": dataset.name, "row_count": dataset.row_count} if dataset else None
            ),
        }

        # Counted like every other policy call. Writing the answer is a real model call that
        # costs real money, and a budget that cannot see it reports a run as cheaper than it
        # was. It is recorded, never gated: the run is already over by the time this happens,
        # so refusing it here would only cost the answer, not save the spend.
        if tracker is not None:
            written = _invoke_policy(tracker, policy, lambda: narrate_answer(  # type: ignore[arg-type]
                policy, question=question, findings=findings))
        else:
            written = narrate_answer(policy, question=question, findings=findings)
        if written is None:
            return None

        # Every figure handed over is a figure the prose may state — as the *kind of thing*
        # it actually is, and nothing else. `counts` keys are already the role names, so a
        # count of experiments is admissible in a clause about experiments and nowhere else.
        allowed = AllowedFigures().add_counts(counts).add_confidence(confidence)
        for h in state.hypotheses:
            allowed.add_confidence(h.confidence)
        for role, per_claim in (("supporting_evidence", supporting), ("refuting_evidence", refuting)):
            for value in per_claim.values():
                allowed.add(role, value)
        if dataset is not None:
            allowed.add("rows", dataset.row_count)
        # The columns this run measured. Nothing in `findings` carries a measured value, so a
        # figure the prose attaches to one of these names is invented whatever it collides
        # with — naming them here is what makes that refusable.
        allowed.add_metric_terms(_metric_vocabulary(state))
        return verify_narrative(written, allowed)


def _contradiction_is_settled(state: InvestigationState, critique: Critique) -> bool:
    """
    True when evidence has separated the two claims a contradiction was recorded over.

    The conflict says one of the pair is wrong, not which. It is answered when the run can
    tell them apart — exactly one still standing — and not before: two claims that are both
    still weakened have had the question put to them and not answered it.

    Derived from the claims rather than read from a flag. The flag existed for a year and was
    never once set to ``True``, which made ``sufficient_evidence`` unreachable for the rest of
    any run that recorded a conflict — including one the discriminating experiment had already
    settled. State cannot drift out of step with itself in the same way.
    """
    pair = [state.find_hypothesis(i) for i in (critique.target.id, critique.conflicts_with_id or "")]
    claims = [h for h in pair if h is not None]
    if len(claims) < 2:
        # Only one side is identifiable, so nothing here can establish that it was settled.
        return False
    standing = [h for h in claims if h.status is HypothesisStatus.supported]
    return len(standing) == 1


def reconcile_contradictions(state: InvestigationState) -> None:
    """Mark contradictions the evidence has since settled, so the trace shows what happened."""
    for critique in state.critiques:
        if critique.critique_type is CritiqueType.contradiction and not critique.resolved:
            critique.resolved = _contradiction_is_settled(state, critique)


def open_contradictions(state: InvestigationState) -> list[Critique]:
    """Conflicts the run is still holding, computed from the claims themselves."""
    return [
        c
        for c in state.critiques
        if c.critique_type is CritiqueType.contradiction and not _contradiction_is_settled(state, c)
    ]


def enforce_mutual_exclusivity(
    state: InvestigationState, idgen: DeterministicIds,
) -> list[Critique]:
    """
    Record a contradiction for any rival pair the evidence has left both standing.

    The rivalry was established by the goal's own phrasing, so this needs no model and cannot
    be missed. It is the same consequence the critic applies to a conflict a model reports —
    both sides weakened, neither picked — because the conflict establishes that they cannot
    both hold, not which of them is wrong.
    """
    recorded: list[Critique] = []
    seen: set[frozenset[str]] = set()
    for claim in state.hypotheses:
        if claim.status is not HypothesisStatus.supported:
            continue
        for rival_id in claim.mutually_exclusive_with:
            rival = state.find_hypothesis(rival_id)
            if rival is None or rival.status is not HypothesisStatus.supported:
                continue
            # Only the symmetric duplicate within this call needs suppressing. Recording
            # weakens both sides, so the pair cannot qualify again unless fresh evidence
            # restores both to `supported` — at which point it is a live conflict again and
            # recording it a second time is the correct behaviour, not a repeat.
            pair = frozenset({claim.id, rival.id})
            if pair in seen:
                continue
            seen.add(pair)
            recorded.append(_record_exclusive_conflict(state, claim, rival, idgen))
    return recorded


def _record_exclusive_conflict(
    state: InvestigationState, claim: Hypothesis, rival: Hypothesis, idgen: DeterministicIds,
) -> Critique:
    critique = Critique(
        id=idgen.make("crit", len(state.critiques)),
        critique_type=CritiqueType.contradiction,
        severity=CritiqueSeverity.major,
        target=EntityRef(kind=EntityKind.hypothesis, id=claim.id),
        conflicts_with_id=rival.id,
        message=(
            f"The goal asked which of these holds, and both are currently supported: "
            f"{_quote(claim.statement)} and {_quote(rival.statement)}. They cannot both be "
            f"the explanation."
        ),
        provenance=_prov("mutual_exclusivity"),
    )
    state.add_critique(critique)
    for side, conflicting in ((claim, rival), (rival, claim)):
        side.set_status(HypothesisStatus.weakened)
        side.set_confidence(round(min(side.confidence, Critic.CONTRADICTION_CONFIDENCE_CAP), 4))
        state.record_decision(AgentDecision(
            id=idgen.make("dec-contra", len(state.decisions)),
            decision_type=DecisionType.revise_confidence,
            rationale=f"weakened: cannot hold at the same time as {_quote(conflicting.statement)}",
            targets=[
                EntityRef(kind=EntityKind.hypothesis, id=side.id),
                EntityRef(kind=EntityKind.hypothesis, id=conflicting.id),
            ],
            provenance=_prov("mutual_exclusivity"),
        ))
    state.add_open_question(OpenQuestion(
        id=idgen.make("q-contra", len(state.open_questions)),
        question=(
            "Which of the two competing explanations does the evidence actually favour?"
        ),
        related_hypothesis_ids=[claim.id, rival.id],
        provenance=_prov("mutual_exclusivity"),
    ))
    return critique


def _unanswerable_detail(state: InvestigationState) -> str:
    """The reason the loop declined, recovered from the decision it wrote before stopping."""
    for decision in reversed(state.decisions):
        if decision.rationale.startswith("declined: "):
            return decision.rationale[len("declined: "):]
    return "the data holds no measure of what the goal asks about"


def _metric_vocabulary(state: InvestigationState) -> list[str]:
    """
    Every column name this run's datasets expose, plus the metrics its claims reference.

    Used only to *refuse* figures, so breadth is the safe direction: a name missing from here
    is a clause the narrative check will not veto, while an extra name costs at most a
    readable sentence that falls back to the deterministic statement.
    """
    names: list[str] = []
    for dataset in state.datasets:
        if dataset.manifest is not None:
            names.extend(c.name for c in dataset.manifest.columns)
    for hypothesis in state.hypotheses:
        names.extend(hypothesis.metric_refs)
    return names


def make_termination(
    reason: TerminationReason, state: InvestigationState, idgen: DeterministicIds, *,
    policy_failure: PolicyFailure | None = None,
) -> TerminationDecision:
    return TerminationDecision(
        should_stop=True, reason=reason,
        rationale=f"terminated: {reason.value}", at_iteration=state.budget.iterations_used,
        provenance=_prov("termination_policy"), policy_failure=policy_failure)
