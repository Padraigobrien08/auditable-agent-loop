"""
Adversarial datasets for agency evaluation.

Each fixture is built so that a *plausible but wrong* agent fails it. Flat and noisy series
punish an agent that manufactures a trend; contradicting and opposing series punish one that
confirms whatever the goal asserts; the clear-signal fixtures punish the opposite failure —
an agent that hedges everything into "insufficient evidence" and is therefore never wrong and
never useful.

That pairing is the point. A suite of only negative cases is passed perfectly by an agent that
always refuses to conclude, so positive controls are load-bearing, not filler.

Deterministic: every series is generated from a fixed formula or a seeded RNG, so a case's
verdict never depends on the run.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

#: Entity/period column names are deliberately generic — the loop must not depend on EDGAR
#: vocabulary to reason about a dataset.
ENTITY = "entity"
PERIOD = "period"
METRIC = "value"

_SEED = 20260806


def _periods(n: int) -> list[str]:
    return [f"20{21 + i // 4}-Q{i % 4 + 1}" for i in range(n)]


def _series(entity: str, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        ENTITY: [entity] * len(values),
        PERIOD: _periods(len(values)),
        METRIC: values,
    })


def clear_rising(n: int = 10) -> pd.DataFrame:
    """Unambiguous upward trend. A competent agent must be able to *conclude* here."""
    return _series("A", [100.0 + 12.0 * i for i in range(n)])


def clear_falling(n: int = 10) -> pd.DataFrame:
    """Unambiguous downward trend — the mirror positive control."""
    return _series("A", [220.0 - 12.0 * i for i in range(n)])


def flat(n: int = 10) -> pd.DataFrame:
    """No movement at all. Any directional claim here is manufactured."""
    return _series("A", [100.0] * n)


def noisy_no_trend(n: int = 16) -> pd.DataFrame:
    """Mean-reverting noise around a constant: variation without direction."""
    rng = np.random.default_rng(_SEED)
    noise = rng.normal(0.0, 3.0, size=n)
    return _series("A", [100.0 + float(v) for v in noise])


def too_short() -> pd.DataFrame:
    """Two points. Enough to draw a line through, nowhere near enough to claim a trend."""
    return _series("A", [100.0, 118.0])


def opposing_entities(n: int = 10) -> pd.DataFrame:
    """Two entities moving in opposite directions.

    The failure mode this targets is cherry-picking: an agent that reports the entity
    agreeing with the goal and discards the one that contradicts it.
    """
    up = _series("A", [100.0 + 12.0 * i for i in range(n)])
    down = _series("B", [220.0 - 12.0 * i for i in range(n)])
    return pd.concat([up, down], ignore_index=True)


def separated_groups(n: int = 8) -> pd.DataFrame:
    """Two clearly different groups, flat in time.

    The signal is *between* groups, not across time, so a goal about comparing groups must
    not be answered with a trend experiment.
    """
    frames = []
    for entity, level in (("A", 100.0), ("B", 300.0)):
        frames.append(_series(entity, [level] * n))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Non-financial fixtures — proving input-agnosticism
# ---------------------------------------------------------------------------
#
# These deliberately use a different domain *and* different column names. Nothing in the
# loop may depend on EDGAR vocabulary, or even on this module's own generic
# entity/period/value names: the adapter declares structure, and the loop reasons over roles.


def rainfall_rising(n: int = 12) -> pd.DataFrame:
    """Monthly rainfall at one weather station, trending up. Columns are domain-native."""
    return pd.DataFrame({
        "station": ["Braemar"] * n,
        "month": [f"2024-{m + 1:02d}" for m in range(n)],
        "rainfall_mm": [40.0 + 6.0 * i for i in range(n)],
    })


def response_latency_flat(n: int = 12) -> pd.DataFrame:
    """Service response latency that is not moving — a non-financial flat control."""
    return pd.DataFrame({
        "service": ["checkout-api"] * n,
        "day": [f"2024-03-{d + 1:02d}" for d in range(n)],
        "latency_ms": [180.0] * n,
    })


#: Fixture id -> builder. Cases reference these by name so a suite stays declarative.
def support_desk_slowing(n: int = 12) -> pd.DataFrame:
    """
    A support desk whose throughput is steady but which is getting steadily slower.

    Three metrics, and the one the question is about is deliberately **not first**:
    ``tickets_opened`` and ``tickets_closed`` are flat, while ``median_resolution_hours``
    climbs from 4h to roughly 15h. A goal like "are we getting slower at resolving customer
    issues?" names none of these columns, so choosing between them is an act of
    interpretation rather than string matching.
    """
    return pd.DataFrame(
        {
            "team": ["support"] * n,
            "week": [f"2025-W{w + 1:02d}" for w in range(n)],
            "tickets_opened": [120.0 + (w % 3) for w in range(n)],
            "tickets_closed": [118.0 + (w % 3) for w in range(n)],
            "median_resolution_hours": [4.0 + 1.0 * w for w in range(n)],
        }
    )


def regional_revenue_spread(n: int = 8) -> pd.DataFrame:
    """
    Four regions whose revenue trajectories differ sharply, with one clear laggard.

    In aggregate the picture is unremarkable — the strong regions offset the weak one — so a
    question about *which* region is worst is only answerable by comparing entities. Reading
    it as a question about the overall trend produces a confident answer to a different
    question.
    """
    rows = []
    # north declines; the rest grow at varying rates, so the pooled series looks flat-ish.
    slopes = {"north": -6.0, "south": 5.0, "east": 3.0, "west": 4.0}
    for region, slope in slopes.items():
        for q in range(n):
            rows.append(
                {
                    "region": region,
                    "quarter": f"2024-Q{q + 1}",
                    "revenue": 100.0 + slope * q,
                }
            )
    return pd.DataFrame(rows)


def plan_tier_separated(n: int = 8) -> pd.DataFrame:
    """
    Usage that separates cleanly by plan tier and not at all by signup month.

    Two dimensions are available and the question names neither. Grouping by the first one
    offered (``signup_month``) finds nothing, which is a true statement about a question
    nobody asked; grouping by ``plan_tier`` finds the real difference.
    """
    rows = []
    for i in range(n):
        for tier, base in (("enterprise", 900.0), ("self_serve", 120.0)):
            rows.append(
                {
                    "account": f"{tier}-{i % 4}",
                    "signup_month": f"2024-{(i % 4) + 1:02d}",
                    "plan_tier": tier,
                    "monthly_active_minutes": base + (i % 3) * 5.0,
                }
            )
    return pd.DataFrame(rows)


def api_latency_rising(n: int = 12) -> pd.DataFrame:
    """
    Infrastructure telemetry with a genuine rising trend — and nothing about complaints.

    Used for a goal the dataset cannot answer. The trend here is real and strong, so a policy
    that silently substitutes an available metric for the one that was asked about will reach
    a confident, well-evidenced conclusion about the wrong question.
    """
    return pd.DataFrame(
        {
            "service": ["gateway"] * n,
            "day": [f"2024-05-{d + 1:02d}" for d in range(n)],
            "p99_latency_ms": [220.0 + 18.0 * d for d in range(n)],
            "requests_per_second": [1400.0 + (d % 3) * 10.0 for d in range(n)],
        }
    )


def growth_slowing_margin_steady(n: int = 10) -> pd.DataFrame:
    """
    Two metrics whose honest answers differ: growth is clearly slowing, margin is clearly not.

    A two-clause question over this data can only be answered correctly by investigating both
    columns. Investigating just the first — the single-metric behaviour that existed before
    phase 33 — produces a confident answer to half the question and silently drops the other
    half, which reads as a complete answer.
    """
    return pd.DataFrame(
        {
            "company": ["acme"] * n,
            "quarter": [f"2024-Q{q + 1}" for q in range(n)],
            "revenue_growth_pct": [20.0 - 1.6 * q for q in range(n)],
            "margin_pct": [31.0 + (q % 2) * 0.2 for q in range(n)],
        }
    )


FIXTURES: dict[str, Callable[[], pd.DataFrame]] = {
    "clear_rising": clear_rising,
    "clear_falling": clear_falling,
    "flat": flat,
    "noisy_no_trend": noisy_no_trend,
    "too_short": too_short,
    "opposing_entities": opposing_entities,
    "separated_groups": separated_groups,
    "rainfall_rising": rainfall_rising,
    "response_latency_flat": response_latency_flat,
    "support_desk_slowing": support_desk_slowing,
    "regional_revenue_spread": regional_revenue_spread,
    "plan_tier_separated": plan_tier_separated,
    "api_latency_rising": api_latency_rising,
    "growth_slowing_margin_steady": growth_slowing_margin_steady,
}


def build_fixture(fixture_id: str) -> pd.DataFrame:
    if fixture_id.startswith("gen:"):
        # Imported here: the generated module builds its cases at import, and a fixture id is
        # the only thing a generated case needs from this one.
        from agentic.evaluation.generated import build_generated_fixture

        return build_generated_fixture(fixture_id)
    try:
        return FIXTURES[fixture_id]()
    except KeyError as exc:  # pragma: no cover - guard
        raise KeyError(f"Unknown agency fixture: {fixture_id}") from exc
