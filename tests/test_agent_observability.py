"""
Backend observability + cost accounting for the agentic investigation loop.

Covers the three signals :class:`BackendAgentObserver` produces (spans, structured
logs, ``edgar_agent_*`` metrics), the guarantee that observation can never fail a run,
and the token→USD path that makes ``LoopBudget.max_cost_usd`` bind on real usage.

All offline: the loop runs against an in-memory dataset with a deterministic policy.
"""

from __future__ import annotations

import pandas as pd
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import REGISTRY

from agentic.adapters import AdapterRequest, InMemoryDatasetAdapter
from agentic.agent import (
    InMemoryInvestigationStore,
    InvestigationLoop,
    LoopBudget,
    LoopComponent,
    PolicyTransportError,
)
from agentic.agent.observer import (
    ComponentCompleted,
    HypothesisTransitioned,
    InvestigationEnded,
)
from agentic.domain.enums import (
    ColumnRole,
    HypothesisStatus,
    InvestigationStatus,
    TerminationReason,
)
from backend.agents.agentic_model_policy import (
    CostAwareModelPolicy,
    CostTrackingResponder,
    build_agent_policy,
)
from backend.config.settings import Settings
from backend.llm.exceptions import ChatCompletionProviderError
from backend.llm.pricing import ModelPrice, estimate_cost_usd, parse_model_prices
from backend.llm.types import ChatCompletionResult
from backend.observability import agent_observer as agent_observer_module
from backend.observability.agent_observer import BackendAgentObserver

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sample(name: str, **labels: str) -> float:
    """Current value of one Prometheus sample, or 0.0 when it has not been recorded."""
    for metric in REGISTRY.collect():
        for s in metric.samples:
            if s.name == name and all(s.labels.get(k) == v for k, v in labels.items()):
                return s.value
    return 0.0


def _trending_up(n: int = 8) -> pd.DataFrame:
    periods = [f"2021-Q{i % 4 + 1}-{i // 4}" for i in range(n)]
    return pd.DataFrame({"entity": ["A"] * n, "period": periods,
                         "revenue": [10.0 + 5.0 * i for i in range(n)]})


def _trend_case():
    df = _trending_up()
    manifest = InMemoryDatasetAdapter(
        frame=df, time_field="period", entity_id_fields=["entity"],
        role_hints={"revenue": ColumnRole.metric},
    ).build_manifest(AdapterRequest())
    return df, manifest


GOAL = "revenue is increasing over time"


@pytest.fixture
def spans() -> InMemorySpanExporter:
    """Isolated tracer provider, so span assertions don't depend on global OTel state."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


@pytest.fixture
def traced_observer(spans, monkeypatch) -> type[BackendAgentObserver]:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(spans))
    monkeypatch.setattr(agent_observer_module, "get_tracer", lambda _name: provider.get_tracer("test"))
    return BackendAgentObserver


def _run_loop(observer: BackendAgentObserver, *, goal: str = GOAL, **kwargs):
    df, manifest = _trend_case()
    return InvestigationLoop(observer=observer, **kwargs).start(
        goal, manifest=manifest, frame=df, seed="obs-test", store=InMemoryInvestigationStore())


class _StubProvider:
    """Chat provider returning fixed usage, or raising, without touching a network."""

    def __init__(self, *, prompt_tokens=1000, completion_tokens=500, raises=False, model="m1") -> None:
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._raises = raises
        self._model = model
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        if self._raises:
            raise ChatCompletionProviderError("provider down")
        # The policy's typed models forbid extra fields, so the reply has to match the
        # schema the caller asked for. Only GoalInterpretation has a required field.
        system = request.messages[0]["content"]
        payload = '{"intent": "trend"}' if "GoalInterpretation" in system else "{}"
        return ChatCompletionResult(
            model=self._model, assistant_text=payload, latency_ms=1,
            prompt_tokens=self._prompt_tokens, completion_tokens=self._completion_tokens)


# ---------------------------------------------------------------------------
# pricing
# ---------------------------------------------------------------------------


def test_parse_model_prices_reads_a_valid_table() -> None:
    table = parse_model_prices({"m1": {"input_per_1m": 0.15, "output_per_1m": 0.6}})
    assert table["m1"] == ModelPrice(input_per_1m=0.15, output_per_1m=0.6)


@pytest.mark.parametrize("raw", [
    {"m1": "not-an-object"},
    {"m1": {"input_per_1m": "abc"}},
    {"m1": {"input_per_1m": -1.0}},
    "not-a-mapping",
    None,
])
def test_parse_model_prices_drops_malformed_entries_without_raising(raw: object) -> None:
    """A typo in operator config must degrade cost tracking, never fail a run."""
    assert parse_model_prices(raw) == {}


def test_estimate_cost_uses_per_million_token_pricing() -> None:
    prices = parse_model_prices({"m1": {"input_per_1m": 2.0, "output_per_1m": 10.0}})
    # 1M prompt tokens at $2 + 0.5M completion tokens at $10 = $2 + $5
    cost = estimate_cost_usd(prices, model="m1", prompt_tokens=1_000_000, completion_tokens=500_000)
    assert cost == pytest.approx(7.0)


def test_unpriced_model_costs_nothing() -> None:
    """Unpriced models must return 0 rather than a guess, so budgets never bind on invented numbers."""
    assert estimate_cost_usd({}, model="unknown", prompt_tokens=10_000, completion_tokens=10_000) == 0.0


def test_missing_usage_costs_nothing() -> None:
    prices = parse_model_prices({"m1": {"input_per_1m": 2.0, "output_per_1m": 10.0}})
    assert estimate_cost_usd(prices, model="m1", prompt_tokens=None, completion_tokens=None) == 0.0


# ---------------------------------------------------------------------------
# cost-tracking responder / policy
# ---------------------------------------------------------------------------


def test_responder_accrues_cost_and_drain_resets_it() -> None:
    prices = parse_model_prices({"m1": {"input_per_1m": 1.0, "output_per_1m": 2.0}})
    responder = CostTrackingResponder(_StubProvider(), model="m1", prices=prices)

    responder("system", "user")  # 1000 prompt @ $1/1M + 500 completion @ $2/1M
    assert responder.drain_cost_usd() == pytest.approx(0.001 + 0.001)
    assert responder.drain_cost_usd() == 0.0, "drain must reset, so cost is never double-charged"


def test_responder_raises_transport_error_and_no_cost_on_provider_error() -> None:
    """An outage must not look like a model that answered with nothing — see PolicyFailureKind."""
    responder = CostTrackingResponder(_StubProvider(raises=True), model="m1", prices={})
    with pytest.raises(PolicyTransportError, match="provider down"):
        responder("system", "user")
    assert responder.drain_cost_usd() == 0.0


def test_build_agent_policy_returns_a_cost_aware_policy(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.agents.agentic_model_policy.get_chat_completion_provider",
        lambda _s: _StubProvider(),
    )
    settings = Settings(llm_model_prices={"m1": {"input_per_1m": 1.0, "output_per_1m": 2.0}})
    policy = build_agent_policy(settings)
    assert isinstance(policy, CostAwareModelPolicy)
    assert policy.drain_cost_usd() == 0.0


def test_cost_flows_from_provider_usage_into_the_loop_budget(monkeypatch) -> None:
    """End-to-end: provider tokens → price table → drain → LoopBudget.max_cost_usd."""
    monkeypatch.setattr(
        "backend.agents.agentic_model_policy.get_chat_completion_provider",
        lambda _s: _StubProvider(prompt_tokens=1_000_000, completion_tokens=0),
    )
    # $1 per policy call, against a $1.5 budget: the run must stop on cost.
    settings = Settings(llm_model_prices={"m1": {"input_per_1m": 1.0, "output_per_1m": 0.0}})
    df, manifest = _trend_case()
    inv = InvestigationLoop(policy=build_agent_policy(settings)).start(
        GOAL, manifest=manifest, frame=df, seed="cost-e2e",
        budget=LoopBudget(max_cost_usd=1.5), store=InMemoryInvestigationStore())
    assert inv.state.termination.reason is TerminationReason.budget_exhausted


# ---------------------------------------------------------------------------
# spans
# ---------------------------------------------------------------------------


def test_span_tree_nests_components_under_iterations_under_the_investigation(traced_observer, spans) -> None:
    _run_loop(traced_observer())
    by_name = {s.name: s for s in spans.get_finished_spans()}

    investigation = by_name.get("agent.investigation")
    assert investigation is not None, "an investigation span must be recorded"

    iterations = [s for s in spans.get_finished_spans() if s.name.startswith("agent.iteration.")]
    components = [s for s in spans.get_finished_spans() if s.name.startswith("agent.component.")]
    assert iterations and components

    inv_span_id = investigation.context.span_id
    assert all(s.parent.span_id == inv_span_id for s in iterations), "iterations hang off the investigation"

    iteration_ids = {s.context.span_id for s in iterations}
    parents = {s.parent.span_id for s in components}
    assert parents <= (iteration_ids | {inv_span_id}), "components hang off an iteration or the investigation"


def test_investigation_span_carries_the_outcome(traced_observer, spans) -> None:
    inv = _run_loop(traced_observer())
    span = next(s for s in spans.get_finished_spans() if s.name == "agent.investigation")
    assert span.attributes["agent.status"] == inv.status.value
    assert span.attributes["agent.termination.reason"] == inv.state.termination.reason.value
    assert span.attributes["agent.iterations"] == inv.state.budget.iterations_used
    assert span.attributes["agent.hypotheses"] == len(inv.state.hypotheses)
    assert span.attributes["agent.partial"] is False


def test_failed_component_span_is_marked_as_an_error(traced_observer, spans) -> None:
    from opentelemetry.trace import StatusCode

    observer = traced_observer()
    observer.on_component_completed(ComponentCompleted(
        investigation_id="i1", component=LoopComponent.selector,
        duration_seconds=0.01, error="MalformedPolicyResponse"))
    span = next(s for s in spans.get_finished_spans() if s.name == "agent.component.selector")
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["agent.error_type"] == "MalformedPolicyResponse"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_investigation_metrics_are_recorded(traced_observer) -> None:
    before = _sample("edgar_agent_investigation_iterations_count")
    inv = _run_loop(traced_observer())
    reason = inv.state.termination.reason.value

    assert _sample("edgar_agent_investigations_total",
                   status=inv.status.value, termination_reason=reason) >= 1
    assert _sample("edgar_agent_terminations_total", reason=reason) >= 1
    assert _sample("edgar_agent_investigation_iterations_count") == before + 1


def test_component_and_experiment_metrics_are_recorded(traced_observer) -> None:
    inv = _run_loop(traced_observer())
    assert _sample("edgar_agent_component_duration_seconds_count",
                   component=LoopComponent.executor.value) >= 1
    tool = inv.state.completed_experiments[0].tool_name
    assert _sample("edgar_agent_experiments_total", tool_name=tool, status="succeeded") >= 1
    assert _sample("edgar_agent_experiment_duration_seconds_count", tool_name=tool) >= 1


def test_hypothesis_transition_metric_is_recorded(traced_observer) -> None:
    observer = traced_observer()
    before = _sample("edgar_agent_hypothesis_transitions_total",
                     from_status=HypothesisStatus.active.value,
                     to_status=HypothesisStatus.supported.value)
    observer.on_hypothesis_transition(HypothesisTransitioned(
        investigation_id="i1", hypothesis_id="h1",
        from_status=HypothesisStatus.active, to_status=HypothesisStatus.supported))
    assert _sample("edgar_agent_hypothesis_transitions_total",
                   from_status=HypothesisStatus.active.value,
                   to_status=HypothesisStatus.supported.value) == before + 1


def test_component_error_metric_is_recorded(traced_observer) -> None:
    observer = traced_observer()
    before = _sample("edgar_agent_component_errors_total",
                     component=LoopComponent.critic.value, error_type="BoomError")
    observer.on_component_completed(ComponentCompleted(
        investigation_id="i1", component=LoopComponent.critic,
        duration_seconds=0.01, error="BoomError"))
    assert _sample("edgar_agent_component_errors_total",
                   component=LoopComponent.critic.value, error_type="BoomError") == before + 1


def test_partial_run_is_not_counted_as_a_terminal_investigation(traced_observer) -> None:
    """A resumable pause must not inflate terminal-outcome counters."""
    before = _sample("edgar_agent_investigations_total",
                     status=InvestigationStatus.running.value, termination_reason="none")
    before_iterations = _sample("edgar_agent_investigation_iterations_count")

    observer = traced_observer()
    observer.on_investigation_end(InvestigationEnded(
        investigation_id="i1", status=InvestigationStatus.running, termination_reason=None,
        iterations=1, experiments_completed=1, experiments_failed=0, hypotheses=1, evidence=1,
        elapsed_seconds=0.5, cost_usd=0.0, model_calls=2, partial=True))

    assert _sample("edgar_agent_investigations_total",
                   status=InvestigationStatus.running.value, termination_reason="none") == before
    assert _sample("edgar_agent_investigation_iterations_count") == before_iterations


def test_model_call_cost_metric_is_recorded(monkeypatch, traced_observer) -> None:
    monkeypatch.setattr(
        "backend.agents.agentic_model_policy.get_chat_completion_provider",
        lambda _s: _StubProvider(prompt_tokens=1_000_000, completion_tokens=0),
    )
    settings = Settings(llm_model_prices={"m1": {"input_per_1m": 1.0, "output_per_1m": 0.0}})
    before = _sample("edgar_agent_cost_usd_total", component=LoopComponent.goal_interpreter.value)

    df, manifest = _trend_case()
    InvestigationLoop(policy=build_agent_policy(settings), observer=traced_observer()).start(
        GOAL, manifest=manifest, frame=df, seed="cost-metric",
        budget=LoopBudget(max_cost_usd=1.5), store=InMemoryInvestigationStore())

    after = _sample("edgar_agent_cost_usd_total", component=LoopComponent.goal_interpreter.value)
    assert after > before
    assert _sample("edgar_agent_model_calls_total",
                   component=LoopComponent.goal_interpreter.value) >= 1


# ---------------------------------------------------------------------------
# resilience
# ---------------------------------------------------------------------------


def test_observer_hooks_never_raise(monkeypatch, traced_observer) -> None:
    """A broken tracer must degrade observability, not fail the investigation."""

    class _BrokenTracer:
        def start_span(self, *a, **kw):
            raise RuntimeError("tracer exploded")

    monkeypatch.setattr(agent_observer_module, "get_tracer", lambda _name: _BrokenTracer())
    inv = _run_loop(BackendAgentObserver(analysis_run_id="run-1"))
    assert inv.is_terminal(), "the run completes even though every span call raised"
    assert inv.state.termination.reason is not TerminationReason.error


def test_observer_survives_a_broken_metric(monkeypatch, traced_observer) -> None:
    def _boom(*_a, **_kw):
        raise RuntimeError("metric exploded")

    monkeypatch.setattr(agent_observer_module.AGENT_EXPERIMENTS_TOTAL, "labels", _boom)
    inv = _run_loop(traced_observer())
    assert inv.is_terminal()
