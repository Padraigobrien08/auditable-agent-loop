"""
Integration tests for the adaptive investigation loop.

Proves the loop is genuinely adaptive (not a renamed static pipeline): execution
paths differ by goal, intermediate results steer selection, hypotheses are
supported/weakened/rejected/unresolved, contradictory evidence is preserved,
the critic drives falsification, insufficient evidence is a valid outcome, the
EDGAR demo completes, and a checkpointed run resumes to the same state. All offline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from agentic.adapters import AdapterRequest, EDGARAdapter, InMemoryDatasetAdapter
from agentic.agent import (
    InMemoryInvestigationStore,
    InvestigationLoop,
    LoopBudget,
    ModelAgentPolicy,
)
from agentic.domain.enums import ColumnRole, ConclusionDisposition, HypothesisStatus, InvestigationStatus

REPO = Path(__file__).resolve().parents[2]
EDGAR_FIXTURE = REPO / "edgar_project/evaluation/fixtures/data/01_simple_anomaly_features.csv"


def _manifest(df: pd.DataFrame, **hints):
    return InMemoryDatasetAdapter(frame=df, **hints).build_manifest(AdapterRequest())


def _executed(inv) -> list[str]:
    return [r.tool_name for r in inv.state.completed_experiments + inv.state.failed_experiments]


def _trending_up(entity: str = "A", n: int = 8, start: float = 10.0, step: float = 5.0) -> pd.DataFrame:
    # Labels must sort chronologically: the tools order periods by sorting them. The previous
    # "2021-Q1-0 … 2021-Q1-1" form sorted to Q1-0, Q1-1, Q2-0, … and scrambled the series, which
    # only went unnoticed because an uncalibrated change-point score stood in for the trend fit.
    periods = [f"{2021 + i // 4}-Q{i % 4 + 1}" for i in range(n)]
    return pd.DataFrame({"entity": [entity] * n, "period": periods,
                         "revenue": [start + step * i for i in range(n)]})


def _loop() -> InvestigationLoop:
    return InvestigationLoop()


# 1. a trend question selects trend experiments -----------------------------


def test_trend_goal_selects_trend_experiments() -> None:
    df = _trending_up()
    m = _manifest(df, time_field="period", entity_id_fields=["entity"], role_hints={"revenue": ColumnRole.metric})
    inv = _loop().start("what is the revenue trend over time?", manifest=m, frame=df, seed="trend",
                        store=InMemoryInvestigationStore())
    tools = _executed(inv)
    assert "analyze_time_series_trend" in tools
    assert "compare_groups" not in tools  # a comparison tool was NOT chosen for a trend goal


# 2. a group comparison selects comparison experiments ----------------------


def test_comparison_goal_selects_comparison_experiments() -> None:
    df = pd.DataFrame({
        "grp": ["g1", "g1", "g1", "g1", "g2", "g2", "g2", "g2"],
        "period": ["2022-Q1", "2022-Q2", "2022-Q3", "2022-Q4"] * 2,
        "revenue": [10.0, 11, 12, 13, 40, 41, 42, 43],
    })
    m = _manifest(df, role_hints={"revenue": ColumnRole.metric, "grp": ColumnRole.dimension})
    inv = _loop().start("compare revenue between groups", manifest=m, frame=df, seed="cmp",
                        store=InMemoryInvestigationStore())
    tools = _executed(inv)
    assert "compare_groups" in tools
    assert "analyze_time_series_trend" not in tools


def test_execution_paths_differ_across_goals() -> None:
    df = _trending_up()
    m = _manifest(df, time_field="period", entity_id_fields=["entity"], role_hints={"revenue": ColumnRole.metric})
    a = _loop().start("show the trend of revenue over time", manifest=m, frame=df, seed="a", store=InMemoryInvestigationStore())
    b = _loop().start("rank entities by revenue", manifest=m, frame=df, seed="b", store=InMemoryInvestigationStore())
    assert set(_executed(a)) != set(_executed(b))


# 3. a weak result triggers a follow-up experiment --------------------------


def test_weak_result_triggers_followup() -> None:
    # noisy / flat-ish series -> weak trend evidence -> not sufficient -> follow-up runs
    df = pd.DataFrame({"entity": ["A"] * 8, "period": [f"2021-{i}" for i in range(8)],
                       "revenue": [10, 10.2, 9.8, 10.1, 9.9, 10.05, 9.95, 10.0]})
    m = _manifest(df, time_field="period", entity_id_fields=["entity"], role_hints={"revenue": ColumnRole.metric})
    inv = _loop().start("is revenue increasing over time?", manifest=m, frame=df, seed="weak",
                        store=InMemoryInvestigationStore())
    assert len(_executed(inv)) >= 2  # a single experiment was not treated as conclusive


# 4. contradictory evidence weakens a hypothesis ----------------------------


def test_contradictory_evidence_weakens_hypothesis() -> None:
    # entity A rises, entity B falls -> one trend experiment yields supporting + refuting evidence
    up = _trending_up("A", n=6, start=10, step=4)
    down = pd.DataFrame({"entity": ["B"] * 6, "period": up["period"].tolist(),
                         "revenue": [40, 34, 28, 22, 16, 10.0]})
    df = pd.concat([up, down], ignore_index=True)
    m = _manifest(df, time_field="period", entity_id_fields=["entity"], role_hints={"revenue": ColumnRole.metric})
    inv = _loop().start("revenue is increasing over time", manifest=m, frame=df, seed="contra",
                        store=InMemoryInvestigationStore())
    h = inv.state.hypotheses[0]
    assert h.status is HypothesisStatus.weakened
    # both supporting and contradicting evidence are preserved
    assert h.supporting_evidence_ids and h.contradicting_evidence_ids


# 5. the critic selects a falsification experiment --------------------------


def test_critic_selects_falsification_experiment() -> None:
    df = _trending_up("A", n=8, start=5, step=6)  # strong, clean up-trend -> supported
    m = _manifest(df, time_field="period", entity_id_fields=["entity"], role_hints={"revenue": ColumnRole.metric})
    inv = _loop().start("revenue is increasing over time", manifest=m, frame=df, seed="falsify",
                        store=InMemoryInvestigationStore())
    assert inv.state.critiques  # the critic challenged the strongest claim
    suggested = {c.suggested_action for c in inv.state.critiques}
    assert suggested & set(_executed(inv))  # the suggested falsification tool actually ran
    assert any(d.decision_type.value == "request_critique" for d in inv.state.decisions)


# 6. insufficient data ends with an unresolved conclusion -------------------


def test_insufficient_data_unresolved_conclusion() -> None:
    df = pd.DataFrame({"entity": ["A"] * 6, "period": [f"2021-{i}" for i in range(6)],
                       "revenue": [5.0] * 6})  # perfectly flat -> no directional signal
    m = _manifest(df, time_field="period", entity_id_fields=["entity"], role_hints={"revenue": ColumnRole.metric})
    inv = _loop().start("is revenue increasing over time?", manifest=m, frame=df, seed="insuf",
                        store=InMemoryInvestigationStore())
    assert inv.state.current_conclusion.disposition is ConclusionDisposition.insufficient_evidence
    assert all(h.status is HypothesisStatus.unresolved for h in inv.state.hypotheses)


# 7. the EDGAR demo still completes successfully ----------------------------


def test_edgar_demo_completes() -> None:
    df = pd.read_csv(EDGAR_FIXTURE)
    m = EDGARAdapter().build_manifest(AdapterRequest(parameters={"panel_csv": str(EDGAR_FIXTURE)}))
    inv = _loop().start("find unusual revenue changes", manifest=m, frame=df, adapter_id="edgar",
                        seed="edgar", store=InMemoryInvestigationStore())
    assert inv.is_terminal()
    assert inv.status is not InvestigationStatus.failed
    assert inv.state.current_conclusion is not None
    assert any(t.startswith("edgar_") for t in _executed(inv))  # EDGAR domain tools were used
    assert inv.state.termination is not None


# 8. resuming from a checkpoint produces the same subsequent state ----------


def _signature(inv) -> dict:
    return {
        "status": inv.status.value,
        "hypotheses": [(h.id, h.status.value, round(h.confidence, 6)) for h in inv.state.hypotheses],
        "evidence": [(e.id, e.direction.value, round(e.strength, 6)) for e in inv.state.evidence],
        "experiments": _executed(inv),
        "conclusion": inv.state.current_conclusion.disposition.value if inv.state.current_conclusion else None,
        "termination": inv.state.termination.reason.value if inv.state.termination else None,
        "decisions": [d.decision_type.value for d in inv.state.decisions],
    }


def test_resume_from_checkpoint_matches_uninterrupted() -> None:
    df = _trending_up("A", n=8, start=5, step=6)
    m = _manifest(df, time_field="period", entity_id_fields=["entity"], role_hints={"revenue": ColumnRole.metric})
    goal = "revenue is increasing over time"

    full = _loop().start(goal, manifest=m, frame=df, seed="resume", store=InMemoryInvestigationStore())

    store = InMemoryInvestigationStore()
    partial = _loop().start(goal, manifest=m, frame=df, seed="resume", store=store, max_new_experiments=1)
    assert partial.state.termination is None  # stopped mid-run, not terminal
    reloaded = store.load(partial.id)
    resumed = _loop().resume(reloaded, goal_text=goal, manifest=m, frame=df, store=store)

    assert _signature(resumed) == _signature(full)


# safe failure on malformed model output ------------------------------------


def test_malformed_model_response_fails_safely() -> None:
    bad_policy = ModelAgentPolicy(respond=lambda system, user: "{ this is : not valid json")
    df = _trending_up()
    m = _manifest(df, time_field="period", entity_id_fields=["entity"], role_hints={"revenue": ColumnRole.metric})
    inv = InvestigationLoop(policy=bad_policy).start("trend of revenue", manifest=m, frame=df, seed="bad",
                                                     store=InMemoryInvestigationStore())
    # no crash; terminates safely with a recorded reason and a conclusion
    assert inv.status is InvestigationStatus.failed
    assert inv.state.termination is not None and inv.state.termination.reason.value == "error"
    assert inv.state.current_conclusion is not None


def test_budget_limits_bound_the_run() -> None:
    df = _trending_up("A", n=12, start=1, step=3)
    m = _manifest(df, time_field="period", entity_id_fields=["entity"], role_hints={"revenue": ColumnRole.metric})
    inv = _loop().start("describe revenue", manifest=m, frame=df, seed="budget",
                        budget=LoopBudget(max_experiments=1), store=InMemoryInvestigationStore())
    assert len(_executed(inv)) <= 1
    assert inv.is_terminal()
