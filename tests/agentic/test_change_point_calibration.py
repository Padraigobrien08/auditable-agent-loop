"""
A change point counts as evidence only when noise could not have produced it.

`detect_change_points` 1.0 scored the *largest* standardised mean shift over every split and
treated it as a pre-chosen Cohen's d. On pure noise that maximum routinely reached evidence
strength 1.0. The evidence updater then read support or refutation off the shift's sign alone,
so the loop supported "value is decreasing" at 0.95 on a series whose trend fit explained
R² = 0.12. Whether a directional claim about noise stood depended on which way its largest
wobble happened to point.

Two fixes, pinned here: the score gets a permutation test of the same maximum statistic, and a
tool's own "not significant" verdict is no longer overridden by the sign of what it measured.
"""

from __future__ import annotations

import numpy as np
import pytest

from agentic.agent.fixture_policy import FixtureAgentPolicy
from agentic.domain.enums import EvidenceDirection, HypothesisStatus
from agentic.evaluation.cases import SUITE_V1_CASES
from agentic.evaluation.runner import run_case
from agentic.experiments.stats import max_mean_shift, max_mean_shift_p_value

_NOISE_CASE = next(c for c in SUITE_V1_CASES if c.case_id == "noise_is_not_a_trend")


def _naive_max_shift(y: np.ndarray, min_segment: int) -> tuple[int, float, float]:
    """Version 1.0's loop, kept as the reference the vectorised statistic must reproduce."""
    n, best = y.size, (-1, -1.0, 0.0)
    for k in range(min_segment, n - min_segment + 1):
        left, right = y[:k], y[k:]
        pooled = np.sqrt((np.var(left) * left.size + np.var(right) * right.size) / n) or 1.0
        shift = float(np.mean(right) - np.mean(left))
        if abs(shift) / pooled > best[1]:
            best = (k, abs(shift) / pooled, shift)
    return best


# -- the statistic is unchanged ---------------------------------------------------------------


@pytest.mark.parametrize("seed", range(25))
def test_vectorised_score_matches_the_original_loop(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 40))
    y = rng.normal(100, 5, n) + np.where(np.arange(n) >= rng.integers(2, n - 2), rng.normal(0, 15), 0.0)

    split, score, shift = _naive_max_shift(y, 2)
    got = max_mean_shift(y, 2)

    assert int(got["split"]) == split
    assert got["score"] == pytest.approx(score, rel=1e-9)
    assert got["shift"] == pytest.approx(shift, rel=1e-9)


def test_an_exact_step_keeps_the_zero_variance_fallback() -> None:
    """Flat segments have zero pooled variance; the score falls back to |shift|, as before."""
    got = max_mean_shift(np.array([100.0] * 4 + [300.0] * 4), 2)

    assert (got["split"], got["score"], got["shift"]) == (4.0, 200.0, 200.0)


# -- and is now tested against noise ------------------------------------------------------------


def test_p_value_is_calibrated_on_noise() -> None:
    """Under the null, about 5% of series should fall below 0.05. Version 1.0 had no test at all."""
    p_values = np.array([
        max_mean_shift_p_value(np.random.default_rng(seed).normal(0, 1, 16), 2, permutations=199)
        for seed in range(200)
    ])

    assert 0.0 < (p_values < 0.05).mean() <= 0.10
    assert 0.35 <= (p_values < 0.5).mean() <= 0.65


def test_a_real_step_is_significant() -> None:
    rng = np.random.default_rng(1)
    y = 100 + np.where(np.arange(16) >= 8, 60.0, 0.0) + rng.normal(0, 5, 16)

    assert max_mean_shift_p_value(y, 2) < 0.01


def test_p_value_is_reproducible() -> None:
    """A run must be reproducible from persisted state, so the permutation stream is fixed."""
    y = np.random.default_rng(7).normal(0, 1, 20)

    assert max_mean_shift_p_value(y, 2) == max_mean_shift_p_value(y, 2)


# -- end to end: noise no longer settles a directional claim ----------------------------------------


@pytest.mark.parametrize("direction", ["increasing", "decreasing"])
def test_noise_does_not_settle_a_directional_claim_either_way(direction: str) -> None:
    """
    The original failure. On this fixture "decreasing" ended supported at 0.95 and
    "increasing" was refuted, both off the sign of one insignificant shift.
    """
    case = _NOISE_CASE.model_copy(update={"goal": f"value is {direction} over time"})
    result = run_case(case, policy=FixtureAgentPolicy())

    assert result.observed_hypothesis_statuses == [HypothesisStatus.unresolved.value]
    assert result.observed_disposition == "insufficient_evidence"


def test_the_change_point_on_noise_is_reported_as_not_significant() -> None:
    from agentic.adapters import AdapterRequest, InMemoryDatasetAdapter
    from agentic.agent.loop import InvestigationLoop
    from agentic.domain.enums import ColumnRole
    from agentic.evaluation.fixtures import build_fixture

    frame = build_fixture(_NOISE_CASE.fixture_id)
    manifest = InMemoryDatasetAdapter(
        frame=frame, time_field="period", entity_id_fields=["entity"],
        role_hints={"value": ColumnRole.metric},
    ).build_manifest(AdapterRequest())
    state = InvestigationLoop().start(_NOISE_CASE.goal, manifest=manifest, frame=frame, seed="cp").state

    shifts = [e for e in state.evidence if "shifts by" in e.claim]
    assert shifts, "the case must run detect_change_points, or this test proves nothing"
    for evidence in shifts:
        assert evidence.direction is EvidenceDirection.neutral
        assert evidence.statistics is not None and evidence.statistics.p_value is not None
        assert evidence.statistics.p_value >= 0.05
        assert any("Not significant" in w for w in evidence.statistics.warnings)
