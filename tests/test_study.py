"""
The study's analysis and runner.

The analysis exists because the scoreboard's pass rate credits a model that never answered:
a stub that only ever returned prose scored 50% raw on core under the loop. These tests pin
that the honest pass does not, that each failure gets the right single class, that intervals
are over cases and reproducible, and that comparisons only ever use cases both sides scored.
"""

from __future__ import annotations

import json

import pytest

from agentic.agent import FixtureAgentPolicy, MalformedPolicyResponse
from agentic.agent.policy import GoalInterpretation
from agentic.domain import PolicyDecisionKind, PolicyFailure, PolicyFailureKind
from agentic.evaluation.agency import AgencyCaseResult, AgencyProperty, AgencyReport, PropertyOutcome
from agentic.evaluation.cases import AGENCY_CASES, CaseTier
from agentic.evaluation.study import (
    FailureClass,
    Observation,
    analyse,
    cell,
    classify,
    observations,
    paired,
)
from backend.config.settings import Settings
from backend.dev.study import CONDITIONS, cell_path, condition_settings, load_observations, run_cell, write_report

_BY_ID = {c.case_id: c for c in AGENCY_CASES}
_CLEAR = _BY_ID["clear_rising_is_concluded"]
_FLAT = _BY_ID["flat_data_is_not_a_trend"]


def _result(case_id: str, *, passed: bool, disposition: str | None, statuses: list[str],
            confidence: float = 0.9, termination: str | None = "sufficient_evidence",
            failure: PolicyFailureKind | None = None,
            prop: AgencyProperty = AgencyProperty.reaches_the_right_disposition) -> AgencyCaseResult:
    return AgencyCaseResult(
        case_id=case_id, passed=passed, outcomes=[PropertyOutcome(property=prop, passed=passed)],
        observed_disposition=disposition, observed_hypothesis_statuses=statuses,
        observed_confidence=confidence, observed_termination=termination,
        observed_policy_failure=PolicyFailure(decision=PolicyDecisionKind.interpret_goal, kind=failure)
        if failure else None,
    )


# -- taxonomy -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "result", "expected"),
    [
        (_CLEAR, _result(_CLEAR.case_id, passed=True, disposition="supported", statuses=["supported"]), None),
        (_CLEAR, _result(_CLEAR.case_id, passed=False, disposition=None, statuses=[],
                         failure=PolicyFailureKind.transport), FailureClass.transport),
        (_CLEAR, _result(_CLEAR.case_id, passed=False, disposition=None, statuses=[],
                         failure=PolicyFailureKind.invalid_json), FailureClass.structural),
        (_CLEAR, _result(_CLEAR.case_id, passed=False, disposition="insufficient_evidence", statuses=["active"],
                         termination="budget_exhausted"), FailureClass.budget_exhaustion),
        (_FLAT, _result(_FLAT.case_id, passed=False, disposition="supported", statuses=["supported"]),
         FailureClass.overclaim),
        (_CLEAR, _result(_CLEAR.case_id, passed=False, disposition="insufficient_evidence", statuses=["unresolved"]),
         FailureClass.underclaim),
        (_CLEAR, _result(_CLEAR.case_id, passed=False, disposition="refuted", statuses=["rejected"]),
         FailureClass.reasoning),
    ],
    ids=["pass", "transport", "structural", "budget", "overclaim", "underclaim", "reasoning"],
)
def test_each_failure_gets_one_class(case, result, expected) -> None:  # noqa: ANN001
    assert classify(result, case.expectations) is expected


def test_a_structural_failure_is_structural_even_when_the_scoreboard_passed_it() -> None:
    """The measured artefact: an empty answer passes an overclaim case on the raw scoreboard."""
    lucky = _result(_FLAT.case_id, passed=True, disposition=None, statuses=[],
                    failure=PolicyFailureKind.schema)

    assert classify(lucky, _FLAT.expectations) is FailureClass.structural


def test_route_properties_do_not_affect_the_honest_pass() -> None:
    route_only_fail = AgencyCaseResult(
        case_id=_CLEAR.case_id, passed=False, observed_disposition="supported",
        observed_hypothesis_statuses=["supported"], observed_confidence=0.9,
        outcomes=[PropertyOutcome(property=AgencyProperty.reaches_the_right_disposition, passed=True),
                  PropertyOutcome(property=AgencyProperty.challenges_before_concluding, passed=False)],
    )

    assert classify(route_only_fail, _CLEAR.expectations) is None


# -- observations ---------------------------------------------------------------------------------


def _report(*results: AgencyCaseResult) -> AgencyReport:
    return AgencyReport(suite_id="s", total=len(results), passed=sum(r.passed for r in results),
                        results=list(results))


def test_transport_runs_are_excluded_not_counted() -> None:
    obs = observations([_report(
        _result(_CLEAR.case_id, passed=False, disposition=None, statuses=[], failure=PolicyFailureKind.transport),
        _result(_FLAT.case_id, passed=True, disposition="insufficient_evidence", statuses=["unresolved"]),
    )], model="m", size_b=1, condition="A", tier="core")

    summary = cell(obs)
    assert summary.excluded_transport == 1
    assert summary.n_runs == 1
    assert summary.rate == 1.0


def test_route_only_cases_are_not_observed() -> None:
    route_only = _BY_ID["comparison_goal_uses_comparison_tools"]
    obs = observations([_report(_result(route_only.case_id, passed=True, disposition=None, statuses=[]))],
                       model="m", size_b=1, condition="A", tier="core")

    assert obs == []


# -- intervals and pairing ---------------------------------------------------------------------------


def _obs(condition: str, outcomes: dict[str, bool], *, trials: int = 1) -> list[Observation]:
    return [Observation(model="m", condition=condition, tier="t", trial=t, case_id=cid, honest_pass=ok,
                        raw_pass=ok) for cid, ok in outcomes.items() for t in range(trials)]


def test_interval_is_over_cases_and_reproducible() -> None:
    outcomes = {f"c{i}": i % 3 != 0 for i in range(30)}

    first, second = cell(_obs("A", outcomes)), cell(_obs("A", outcomes, trials=5))
    assert first.ci_low < first.rate < first.ci_high
    # More trials of a deterministic case add no information, so they must not narrow it.
    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)
    assert first == cell(_obs("A", outcomes))


def test_a_clear_difference_excludes_zero_and_no_difference_does_not() -> None:
    cases = [f"c{i}" for i in range(30)]
    better = _obs("A", {c: True for c in cases})
    worse = _obs("C", {c: i % 2 == 0 for i, c in enumerate(cases)})

    assert paired(better, worse).excludes_zero
    assert not paired(better, _obs("B", {c: True for c in cases})).excludes_zero


def test_pairing_uses_only_cases_both_conditions_scored() -> None:
    a = _obs("A", {"shared": True, "route_only": False})
    c = _obs("C", {"shared": True})

    diff = paired(a, c)
    assert diff.n_cases == 1
    assert diff.diff == 0.0


def test_every_condition_is_compared_with_the_full_loop() -> None:
    obs = _obs("A", {"x": True}) + _obs("B", {"x": False}) + _obs("C", {"x": True})

    assert {(d.a, d.b) for d in analyse(obs).comparisons} == {("A", "B"), ("A", "C")}


# -- the runner ------------------------------------------------------------------------------------


def test_conditions_map_to_the_bench() -> None:
    assert condition_settings("A") == ("loop", None)
    assert condition_settings("B")[1].removed == ["critic", "typed_termination", "mutual_exclusivity"]
    assert condition_settings("B-critic")[1].removed == ["critic"]
    assert condition_settings("C") == ("bare", None)
    assert set(CONDITIONS) >= {"A", "B", "B-critic", "B-typed_termination", "B-mutual_exclusivity", "C"}


def test_a_cell_is_persisted_once_and_resumes(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    kwargs = dict(model="fixture", size_b=0.0, condition="A", tiers=(CaseTier.core,), trials=1)

    written = run_cell(tmp_path, **kwargs)
    assert written == cell_path(tmp_path, "fixture", "A")
    payload = json.loads(written.read_text())
    assert payload["prompt_version"] and payload["tiers"]["core"]["reports"]

    assert run_cell(tmp_path, **kwargs) is None, "an existing cell must be skipped, not re-run"


def test_the_rule_based_policy_has_no_bare_form(tmp_path) -> None:  # noqa: ANN001
    with pytest.raises(SystemExit, match="no bare form"):
        run_cell(tmp_path, model="fixture", size_b=0.0, condition="C", tiers=(CaseTier.core,), trials=1)


class _NeverAnswers(FixtureAgentPolicy):
    def interpret_goal(self, goal_text: str, *, capability_summary: dict) -> GoalInterpretation:
        raise MalformedPolicyResponse("not JSON", kind=PolicyFailureKind.invalid_json)

    def drain_cost_usd(self) -> float:
        return 0.0


def test_a_model_that_never_answers_scores_zero_honestly(tmp_path) -> None:  # noqa: ANN001
    """End to end through the runner: the raw scoreboard credits it, the honest pass does not."""
    free = Settings(agent_completion_model="m", llm_model_prices={"m": {"input_per_1m": 0, "output_per_1m": 0}})
    run_cell(tmp_path, model="m", size_b=1.0, condition="A", tiers=(CaseTier.core,), trials=1,
             policy_factory=lambda kind, s: _NeverAnswers(), settings=free)

    [summary] = analyse(load_observations(tmp_path)).cells
    assert summary.rate == 0.0
    assert summary.raw_rate > 0.0
    assert set(summary.failures) == {"structural"}
    assert "Honest pass rate" in write_report(tmp_path)
    assert (tmp_path / "study.json").exists()
