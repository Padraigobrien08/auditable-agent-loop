"""
Condition C: the model alone, scored by the loop's own answer checks.

The harness must not give the model hints the loop does not get, must record a failed call the
same way the loop does, and must score with the same checks rather than a copy. The parity
test below is what keeps the last of those true.
"""

from __future__ import annotations

import json

import pytest

from agentic.agent import FixtureAgentPolicy, PolicyTransportError
from agentic.domain import PolicyDecisionKind, PolicyFailureKind
from agentic.evaluation.agency import ANSWER_PROPERTIES, answer_is_scorable, score_answer
from agentic.evaluation.cases import AGENCY_CASES, CaseTier
from agentic.evaluation.generated import GENERATED_CASES
from agentic.evaluation.runner import run_case
from backend.config.settings import Settings
from backend.dev.agency_bench import BARE, run_policy_rows
from backend.dev.bare_model import bare_cases, run_bare_case, run_bare_suite, user_prompt

_CLEAR = next(c for c in AGENCY_CASES if c.case_id == "clear_rising_is_concluded")
_FLAT = next(c for c in AGENCY_CASES if c.case_id == "flat_data_is_not_a_trend")


def _answer(disposition: str, *, statuses=("supported",), confidence: float = 0.9) -> str:
    return json.dumps({
        "disposition": disposition, "confidence": confidence, "rationale": "r",
        "claims": [{"statement": "value is increasing over time", "status": s} for s in statuses],
    })


# -- what the model is shown -----------------------------------------------------------------


def test_the_prompt_carries_the_question_and_the_whole_table() -> None:
    payload = json.loads(user_prompt(_CLEAR))

    assert payload["question"] == _CLEAR.goal
    assert payload["table_csv"].count("\n") == 11  # header + 10 rows


@pytest.mark.parametrize("case", AGENCY_CASES + GENERATED_CASES, ids=lambda c: c.case_id)
def test_the_prompt_never_leaks_the_answer_key(case) -> None:  # noqa: ANN001
    prompt = user_prompt(case)

    assert case.case_id not in prompt
    assert case.description not in prompt
    assert set(json.loads(prompt)) == {"question", "table_csv"}


# -- scoring -----------------------------------------------------------------------------------


def test_a_correct_answer_passes_and_an_overclaim_fails() -> None:
    concluded = run_bare_case(_CLEAR, lambda _s, _u: _answer("supported"))
    overclaimed = run_bare_case(_FLAT, lambda _s, _u: _answer("supported"))

    assert concluded.passed
    assert not overclaimed.passed
    assert concluded.observed_policy_failure is None


@pytest.mark.parametrize(
    ("respond", "kind"),
    [
        (lambda _s, _u: "Sure, the trend is up.", PolicyFailureKind.invalid_json),
        (lambda _s, _u: json.dumps({"disposition": "probably"}), PolicyFailureKind.schema),
    ],
    ids=["prose", "schema"],
)
def test_a_malformed_answer_is_recorded_like_a_loop_failure(respond, kind) -> None:  # noqa: ANN001
    result = run_bare_case(_CLEAR, respond)

    assert not result.passed
    failure = result.observed_policy_failure
    assert failure is not None
    assert (failure.decision, failure.kind) == (PolicyDecisionKind.bare_answer, kind)


def test_an_unreachable_provider_is_transport_not_the_model() -> None:
    def respond(_s: str, _u: str) -> str:
        raise PolicyTransportError("provider error: connection refused")

    result = run_bare_case(_CLEAR, respond)

    assert result.observed_policy_failure is not None
    assert result.observed_policy_failure.kind is PolicyFailureKind.transport


@pytest.mark.parametrize("case", AGENCY_CASES + GENERATED_CASES, ids=lambda c: c.case_id)
def test_score_answer_is_exactly_the_loops_answer_checks(case) -> None:  # noqa: ANN001
    """
    The parity guarantee. Scoring a loop run's conclusion with `score_answer` must give the same
    outcomes, in the same order, as the answer-property subset of `score_case`. A divergence
    would mean condition C is held to a different bar than conditions A and B.
    """
    loop = run_case(case, policy=FixtureAgentPolicy())
    bare_view = score_answer(
        case.expectations, disposition=loop.observed_disposition,
        statuses=loop.observed_hypothesis_statuses, confidence=loop.observed_confidence)

    assert bare_view == [o for o in loop.outcomes if o.property in ANSWER_PROPERTIES]


def test_route_only_cases_are_not_scored_bare() -> None:
    """There is no route to score, so a case that asserts only the route is skipped."""
    skipped = {c.case_id for c in AGENCY_CASES} - {c.case_id for c in bare_cases()}

    assert skipped
    assert all(not answer_is_scorable(c.expectations) for c in AGENCY_CASES if c.case_id in skipped)
    assert "comparison_goal_uses_comparison_tools" in skipped


def test_every_generated_case_is_scorable_bare() -> None:
    assert len(bare_cases(CaseTier.generated)) == len(GENERATED_CASES)


def test_a_suite_trial_reports_cost_per_call() -> None:
    spend = iter([0.002] * 100)
    report, metrics = run_bare_suite(lambda _s, _u: _answer("supported"), tier=CaseTier.core,
                                     drain_cost=lambda: next(spend))

    assert report.total == len(metrics) == len(bare_cases(CaseTier.core))
    assert all(m.model_calls == 1 and m.cost_usd == 0.002 for m in metrics)


# -- the bench -----------------------------------------------------------------------------------


def _priced() -> Settings:
    return Settings(agent_completion_model="test-model",
                    llm_model_prices={"m": {"input_per_1m": 0.15, "output_per_1m": 0.6}})


def test_bench_bare_rows_are_labelled_and_score_only_answer_properties() -> None:
    rows = run_policy_rows(
        ["model"], model="m", trials=1, settings=_priced(), condition=BARE,
        bare_factory=lambda s: (lambda _s, _u: _answer("supported"), lambda: 0.001),
    )

    assert [(r.label, r.tier) for r in rows] == [("m [bare]", "core"), ("m [bare]", "hard")]
    assert all(set(r.property_means) <= {p.value for p in ANSWER_PROPERTIES} for r in rows)


def test_bench_refuses_bare_with_a_fixture_row_or_ablations() -> None:
    from agentic.agent import LoopAblations

    with pytest.raises(SystemExit, match="only --policy model"):
        run_policy_rows(["fixture"], trials=1, settings=_priced(), condition=BARE)
    with pytest.raises(SystemExit, match="--ablate does not apply"):
        run_policy_rows(["model"], model="m", trials=1, settings=_priced(), condition=BARE,
                        ablations=LoopAblations.without("critic"))


def test_bench_bare_reports_an_unreachable_provider() -> None:
    def respond(_s: str, _u: str) -> str:
        raise PolicyTransportError("provider error: 401")

    with pytest.raises(SystemExit, match="provider never answered"):
        run_policy_rows(["model"], model="m", trials=2, settings=_priced(), condition=BARE,
                        bare_factory=lambda s: (respond, lambda: 0.0))
