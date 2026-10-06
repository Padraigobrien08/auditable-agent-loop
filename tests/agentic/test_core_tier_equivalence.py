"""
The frozen core tier's investigation *paths*, pinned.

`docs/agent/agency-scoreboard.md` publishes a measurement over these 13 cases, and
`data/evaluation/agency/scoreboard-2026-08-07*.json` holds the raw result. A change to how the
loop plans can alter which tools run and in what order — which alters experiment ids, which
alters replay diffs — while the pass rate stays a flawless 13/13. The published measurement
would then describe investigations the code no longer performs, and nothing would say so.

Pass rate is therefore not enough. This pins the route: the ordered tool sequence, the
termination reason, the disposition, and the confidence, for every frozen case.

**A failure here is not a test to update.** It means the published scoreboard has gone stale and
the phase making the change has to stop and decide whether to re-measure. Regenerating these
values to make the suite green would silently invalidate a committed result.

The expectations below are literal data, captured from `FixtureAgentPolicy` on 2026-08-08
before phase 33 touched `InvestigationPlanner`. They are deliberately not computed at runtime —
a generated baseline compares the new behaviour against itself and can never fail.
"""

from __future__ import annotations

import pytest

from agentic.agent.fixture_policy import FixtureAgentPolicy
from agentic.evaluation.cases import SUITE_V1_CASES
from agentic.evaluation.runner import run_case

#: case_id -> (ordered tools, termination reason, disposition, confidence)
EXPECTED_PATHS: dict[str, tuple[list[str], str, str, float]] = {
    "clear_rising_is_concluded": (
        ["analyze_time_series_trend", "detect_change_points"],
        "sufficient_evidence", "supported", 0.95,
    ),
    "clear_falling_is_concluded": (
        ["analyze_time_series_trend", "detect_change_points"],
        "sufficient_evidence", "supported", 0.95,
    ),
    "flat_data_is_not_a_trend": (
        ["analyze_time_series_trend", "detect_change_points", "summarize_distribution"],
        "insufficient_evidence", "insufficient_evidence", 0.2,
    ),
    # Re-pinned deliberately on 2026-10-05 with detect_change_points 1.1. Was ("refuted", 0.6):
    # the uncalibrated change-point score of pure noise refuted "increasing" at strength 1.0,
    # which is itself an overclaim of the opposite direction. Same route, same termination,
    # same pass. Only the conclusion is now the honest one. See docs/agent/agency-scoreboard.md.
    "noise_is_not_a_trend": (
        ["analyze_time_series_trend", "detect_change_points", "summarize_distribution"],
        "insufficient_evidence", "insufficient_evidence", 0.2,
    ),
    "two_points_are_not_a_trend": (
        ["summarize_distribution"],
        "insufficient_evidence", "insufficient_evidence", 0.2,
    ),
    "contradicted_claim_is_revised": (
        ["analyze_time_series_trend", "detect_change_points", "summarize_distribution"],
        "insufficient_evidence", "refuted", 0.6,
    ),
    "opposing_entities_are_not_cherry_picked": (
        ["analyze_time_series_trend", "detect_change_points", "summarize_distribution"],
        "insufficient_evidence", "inconclusive", 0.4,
    ),
    "comparison_goal_uses_comparison_tools": (
        ["rank_entities"],
        "insufficient_evidence", "insufficient_evidence", 0.2,
    ),
    "trend_goal_uses_trend_tools": (
        ["analyze_time_series_trend", "detect_change_points"],
        "sufficient_evidence", "supported", 0.95,
    ),
    "non_financial_trend_is_concluded": (
        ["analyze_time_series_trend", "detect_change_points"],
        "sufficient_evidence", "supported", 0.95,
    ),
    "non_financial_flat_is_not_a_trend": (
        ["analyze_time_series_trend", "detect_change_points", "summarize_distribution"],
        "insufficient_evidence", "insufficient_evidence", 0.2,
    ),
    "supported_claim_is_challenged_before_concluding": (
        ["analyze_time_series_trend", "detect_change_points"],
        "sufficient_evidence", "supported", 0.95,
    ),
    "budget_is_respected": (
        ["analyze_time_series_trend", "detect_change_points"],
        "budget_exhausted", "supported", 0.95,
    ),
}

_STALE_SCOREBOARD = (
    "\n\nThis backs the published measurement in docs/agent/agency-scoreboard.md. If the change "
    "that caused this is intended, the scoreboard needs re-measuring — do not regenerate these "
    "expectations to go green."
)


def test_the_pinned_set_covers_every_frozen_case() -> None:
    """A case missing from the map would be silently unguarded."""
    frozen = {c.case_id for c in SUITE_V1_CASES}

    assert set(EXPECTED_PATHS) == frozen, (
        f"missing from EXPECTED_PATHS: {sorted(frozen - set(EXPECTED_PATHS))}; "
        f"unknown case ids pinned: {sorted(set(EXPECTED_PATHS) - frozen)}"
    )


@pytest.mark.parametrize("case", SUITE_V1_CASES, ids=lambda c: c.case_id)
def test_the_investigation_takes_the_same_route(case) -> None:
    tools, termination, disposition, confidence = EXPECTED_PATHS[case.case_id]
    result = run_case(case, policy=FixtureAgentPolicy())

    assert result.observed_tools == tools, (
        f"{case.case_id}: tool sequence changed\n"
        f"  expected {tools}\n  observed {result.observed_tools}" + _STALE_SCOREBOARD
    )
    assert result.observed_termination == termination, (
        f"{case.case_id}: terminated {result.observed_termination!r}, "
        f"expected {termination!r}" + _STALE_SCOREBOARD
    )
    assert result.observed_disposition == disposition, (
        f"{case.case_id}: concluded {result.observed_disposition!r}, "
        f"expected {disposition!r}" + _STALE_SCOREBOARD
    )
    assert result.observed_confidence == pytest.approx(confidence), (
        f"{case.case_id}: confidence {result.observed_confidence}, "
        f"expected {confidence}" + _STALE_SCOREBOARD
    )
