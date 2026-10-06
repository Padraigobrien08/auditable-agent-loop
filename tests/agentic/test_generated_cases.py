"""
The generated tier: its answer key comes from the data, its cases come in twins, and it can
see the scaffold component the hand-written suite cannot.

Ground truth here is the generating process plus a t-statistic computed independently of the
experiment tools. These tests check that each case really sits in the band its expectations
assume, so the answer key cannot drift away from the data it describes.
"""

from __future__ import annotations

import numpy as np
import pytest

from agentic.agent import FixtureAgentPolicy, LoopAblations
from agentic.agent.alternatives import poses_alternatives
from agentic.agent.policy import HypothesisProposal, HypothesisProposals
from agentic.evaluation.cases import AGENCY_CASES, SUPPORTED, CaseTier
from agentic.evaluation.fixtures import build_fixture
from agentic.evaluation.generated import (
    CLEAR_MIN_T,
    DOMAINS,
    GENERATED_CASES,
    GENERATED_SUITE_ID,
    NULL_MAX_T,
    SIGNAL_SWEEP,
    parse_fixture_id,
    realised_t,
    series_values,
)
from agentic.evaluation.runner import run_agency_suite, run_case, run_signal_sweep

_BY_ID = {c.case_id: c for c in GENERATED_CASES}


def _t(case) -> float:  # noqa: ANN001 - AgencyCase
    return realised_t(series_values(parse_fixture_id(case.fixture_id)))


def _goal_sign(case) -> int:  # noqa: ANN001 - AgencyCase
    return 1 if "increasing" in case.goal or "rising" in case.goal else -1


# -- shape ---------------------------------------------------------------------------------


def test_the_tier_is_about_thirty_cases() -> None:
    assert 25 <= len(GENERATED_CASES) <= 40


def test_generated_cases_never_join_the_published_suite() -> None:
    """`suite_agency_v2` numbers are published. Growing it in place would change their meaning."""
    assert all(c.tier is CaseTier.generated for c in GENERATED_CASES)
    assert not any(c.tier is CaseTier.generated for c in AGENCY_CASES)
    assert not {c.case_id for c in GENERATED_CASES} & {c.case_id for c in AGENCY_CASES}


def test_case_ids_are_unique() -> None:
    ids = [c.case_id for c in GENERATED_CASES + SIGNAL_SWEEP]
    assert len(ids) == len(set(ids))


def test_cases_span_several_domains() -> None:
    """Different column names per domain, so the tier also exercises input-agnosticism."""
    assert len({c.metric_field for c in GENERATED_CASES}) >= 3


# -- reproducible from the id alone ------------------------------------------------------------


@pytest.mark.parametrize("case", GENERATED_CASES + SIGNAL_SWEEP, ids=lambda c: c.case_id)
def test_fixture_id_round_trips_and_builds_deterministically(case) -> None:  # noqa: ANN001
    assert parse_fixture_id(case.fixture_id).fixture_id == case.fixture_id
    first, second = build_fixture(case.fixture_id), build_fixture(case.fixture_id)
    assert first.equals(second)
    assert list(first.columns) == [case.entity_id_fields[0], case.time_field, case.metric_field]


def test_an_unknown_generated_id_is_refused() -> None:
    with pytest.raises(KeyError):
        build_fixture("gen:trend:nowhere:n10:t3:seed0")


# -- the answer key matches the data -----------------------------------------------------------


@pytest.mark.parametrize("case", [c for c in GENERATED_CASES if c.case_id.endswith("_clear")],
                         ids=lambda c: c.case_id)
def test_clear_cases_are_unambiguous_in_the_stated_direction(case) -> None:  # noqa: ANN001
    assert _t(case) * _goal_sign(case) >= CLEAR_MIN_T


@pytest.mark.parametrize("case", [c for c in GENERATED_CASES if c.case_id.endswith("_null")],
                         ids=lambda c: c.case_id)
def test_null_cases_have_no_slope_to_find(case) -> None:  # noqa: ANN001
    assert parse_fixture_id(case.fixture_id).target_t == 0
    assert abs(_t(case)) < NULL_MAX_T


@pytest.mark.parametrize("case", [c for c in GENERATED_CASES if c.case_id.endswith("_contradicted")],
                         ids=lambda c: c.case_id)
def test_contradicted_cases_run_clearly_the_other_way(case) -> None:  # noqa: ANN001
    assert -_t(case) * _goal_sign(case) >= CLEAR_MIN_T


def test_realised_t_agrees_with_a_textbook_fit() -> None:
    """Computed independently of the tools, so check it against an independent formula."""
    rng = np.random.default_rng(0)
    y = 3.0 + 0.4 * np.arange(20) + rng.normal(0, 1.0, 20)
    x = np.arange(20, dtype=float)
    xc = x - x.mean()
    slope = (xc @ (y - y.mean())) / (xc @ xc)
    resid = y - (y.mean() + slope * xc)
    se = np.sqrt(resid @ resid / 18 / (xc @ xc))
    assert realised_t(y) == pytest.approx(slope / se)


# -- twins -------------------------------------------------------------------------------------


def test_every_overclaiming_case_has_an_underclaiming_twin() -> None:
    """An always-hedging agent passes every negative case. Its twin is what makes it fail."""
    for case in GENERATED_CASES:
        for suffix, twin_suffix in (("_null", "_clear"), ("_contradicted", "_clear"), ("_rivals", "_plain")):
            if case.case_id.endswith(suffix):
                twin = _BY_ID[case.case_id.removesuffix(suffix) + twin_suffix]
                assert SUPPORTED in twin.expectations.disposition_in
                assert twin.metric_field == case.metric_field


def test_rival_cases_pose_alternatives_and_share_data_with_their_twin() -> None:
    rivals = [c for c in GENERATED_CASES if c.case_id.endswith("_rivals")]
    assert rivals
    for case in rivals:
        assert poses_alternatives(case.goal)
        assert case.expectations.max_supported_claims == 1
        assert _BY_ID[case.case_id.removesuffix("_rivals") + "_plain"].fixture_id == case.fixture_id


# -- what the tier can see ---------------------------------------------------------------------


class _BothExplanations(FixtureAgentPolicy):
    """Proposes both branches of an either/or goal, as a competent model would."""

    def generate_hypotheses(self, interpretation, *, metric_names, dimension_names, goal_text=""):  # noqa: ANN001
        if not poses_alternatives(goal_text):
            return super().generate_hypotheses(interpretation, metric_names=metric_names,
                                               dimension_names=dimension_names, goal_text=goal_text)
        metric = metric_names[0]
        return HypothesisProposals(hypotheses=[
            HypothesisProposal(statement=f"{metric} is increasing steadily over time", metric=metric, direction="up"),
            HypothesisProposal(statement=f"{metric} had a one-off level shift", metric=metric, direction="none"),
        ])


@pytest.mark.parametrize("case", [c for c in GENERATED_CASES if c.case_id.endswith("_rivals")],
                         ids=lambda c: c.case_id)
def test_rival_cases_detect_the_mutual_exclusivity_ablation(case) -> None:  # noqa: ANN001
    """The hand-written suite could not tell this check was missing. This tier can."""
    full = run_case(case, policy=_BothExplanations())
    ablated = run_case(case, policy=_BothExplanations(), ablations=LoopAblations.without("mutual_exclusivity"))

    assert full.passed
    assert not ablated.passed
    assert ablated.observed_hypothesis_statuses.count("supported") == 2


def test_the_tier_runs_under_its_own_suite_id() -> None:
    report = run_agency_suite(policy=FixtureAgentPolicy(), tier=CaseTier.generated)

    assert report.suite_id == GENERATED_SUITE_ID
    assert report.total == len(GENERATED_CASES)


# -- the sweep -----------------------------------------------------------------------------------


def test_sweep_cases_assert_nothing() -> None:
    """Between the bands the answer depends on the evidence standard, so nothing is scored."""
    assert all(c.expectations.model_dump(exclude_defaults=True) == {} for c in SIGNAL_SWEEP)


def test_sweep_reports_points_not_a_pass_rate() -> None:
    points = run_signal_sweep(policy=FixtureAgentPolicy())

    assert len(points) == len(SIGNAL_SWEEP)
    assert {p.target_t for p in points} >= {0.0, 8.0}
    strongest = [p for p in points if p.target_t == max(p.target_t for p in points)]
    assert all(p.claimed for p in strongest), "an unmistakable trend must be claimed"


def test_every_domain_is_used() -> None:
    used = {parse_fixture_id(c.fixture_id).domain for c in GENERATED_CASES}
    assert used == set(DOMAINS)
