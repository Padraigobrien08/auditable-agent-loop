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
from agentic.evaluation.runner import SweepPoint
from agentic.evaluation.study import (
    FailureClass,
    Observation,
    analyse,
    cell,
    classify,
    observations,
    paired,
    sweep_curve,
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


# -- the signal-strength curve -----------------------------------------------------------------------

def _points(rates: dict[float, list[bool]], *, failed: int = 0) -> list[SweepPoint]:
    points = [SweepPoint(case_id=f"t{t}-{i}", target_t=t, realised_t=t, claimed=c)
              for t, claims in rates.items() for i, c in enumerate(claims)]
    points += [SweepPoint(case_id=f"f{i}", target_t=0.0, realised_t=0.0, claimed=False,
                          policy_failure="bare_answer:invalid_json") for i in range(failed)]
    return points


def test_t50_is_interpolated_where_the_claim_rate_crosses_half() -> None:
    curve = sweep_curve(_points({0.0: [False, False], 2.0: [False, False], 4.0: [True, True]}),
                        model="m", size_b=1, condition="A")

    assert curve.t50 == pytest.approx(3.0)
    assert [lv.claim_rate for lv in curve.levels] == [0.0, 0.0, 1.0]


@pytest.mark.parametrize(("claims", "note"), [(True, "claims at every level"), (False, "never claims")])
def test_a_curve_that_never_crosses_says_which_way(claims: bool, note: str) -> None:
    curve = sweep_curve(_points({0.0: [claims], 4.0: [claims]}), model="m", size_b=1, condition="A")

    assert curve.t50 is None
    assert curve.t50_note == note


def test_failed_calls_are_left_out_of_the_curve_not_counted_as_declining() -> None:
    curve = sweep_curve(_points({0.0: [True]}, failed=3), model="m", size_b=1, condition="C")

    assert curve.excluded == 3
    assert curve.levels[0].claim_rate == 1.0


def test_a_sweep_is_persisted_once_and_reported(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    from backend.dev.study import load_sweeps, run_sweep, sweep_path

    written = run_sweep(tmp_path, model="fixture", size_b=0.0, condition="A")
    assert written == sweep_path(tmp_path, "fixture", "A")
    assert run_sweep(tmp_path, model="fixture", size_b=0.0, condition="A") is None

    [curve] = load_sweeps(tmp_path)
    assert curve.levels[-1].claim_rate == 1.0, "an unmistakable trend must be claimed"
    assert "Signal-strength curve" in write_report(tmp_path)


def test_a_bare_sweep_produces_the_same_points(tmp_path) -> None:  # noqa: ANN001
    from backend.dev.study import load_sweeps, run_sweep

    answer = json.dumps({"disposition": "supported", "confidence": 0.9,
                         "claims": [{"statement": "value is increasing over time", "status": "supported"}]})
    free = Settings(agent_completion_model="m", llm_model_prices={"m": {"input_per_1m": 0, "output_per_1m": 0}})
    run_sweep(tmp_path, model="m", size_b=1.0, condition="C", settings=free,
              bare_factory=lambda s: (lambda _s, _u: answer, lambda: 0.0))

    [curve] = load_sweeps(tmp_path)
    assert curve.condition == "C"
    assert curve.t50_note == "claims at every level"
    assert sum(lv.n for lv in curve.levels) == 36


def test_a_sweep_where_the_provider_never_answered_is_not_written(tmp_path) -> None:  # noqa: ANN001
    from agentic.agent import PolicyTransportError
    from backend.dev.study import run_sweep, sweep_path

    def respond(_s: str, _u: str) -> str:
        raise PolicyTransportError("provider error: 401")

    free = Settings(agent_completion_model="m", llm_model_prices={"m": {"input_per_1m": 0, "output_per_1m": 0}})
    with pytest.raises(SystemExit, match="never answered"):
        run_sweep(tmp_path, model="m", size_b=1.0, condition="C", settings=free,
                  bare_factory=lambda s: (respond, lambda: 0.0))
    assert not sweep_path(tmp_path, "m", "C").exists()
