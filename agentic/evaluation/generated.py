"""
Generated agency cases: parameterised signal strength, paired twins, and rival claims.

S2 of ``docs/decisions/2026-10-05-scaffold-vs-model.md``. The hand-written suite is small (19
cases) and saturated, so it cannot rank competent agents or carry a confidence interval. This
module adds cases whose difficulty is a dial rather than an authoring choice.

**Ground truth comes from the generating process, not from the loop.** The trend tool counts
a fit as support at R² ≥ 0.3 and the hypothesis updater calls a claim supported at strength
0.5. Using those thresholds as the answer key would make the full loop pass by construction
and turn the study into a measurement of agreement with its own rules. Instead every series
is generated with a known slope, and its *realised* OLS t-statistic is computed here with
numpy, independently of the experiment tools. A case is scored only where the answer is not
in doubt:

- **clear** (realised |t| ≥ 10, in the stated direction): the trend must be concluded.
- **null** (true slope zero, realised |t| < 1): no directional claim may stand.
- **contradicted** (clear, but opposite to the goal): the goal's claim must not stand.

Between those, the honest answer depends on how much evidence one demands, so those levels
are not scored. They form :data:`SIGNAL_SWEEP`, read as a curve: the realised t at which a
configuration starts claiming a trend.

**Every overclaiming case has an underclaiming twin** on the same domain and length, as in the
hand-written suite. An agent that always hedges fails the clear case. One that always concludes
fails the null and contradicted cases.

**Rival claims.** No hand-written case poses alternatives, so ablating the mutual-exclusivity
check changed nothing anywhere in the suite. On a step series, "is it rising steadily, or was
there a one-off jump?" gives a trend fit that backs both explanations equally. The loop must not
affirm both. The twin asks the plain question over the same data ("is it increasing?"), which
must be concluded.

These cases are a tier of their own (:attr:`CaseTier.generated`) and never join
:data:`~agentic.evaluation.cases.AGENCY_CASES`, so published ``suite_agency_v2`` numbers keep
their meaning.

Deterministic: every series is a pure function of its fixture id, which encodes all of its
parameters (``gen:trend:rainfall:n12:t15:seed3``). Seeds are chosen by a fixed search, never at
random, so a case is reproducible from its id alone.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from agentic.evaluation.agency import AgencyExpectations
from agentic.evaluation.cases import (
    INCONCLUSIVE,
    INSUFFICIENT,
    MIXED,
    REFUTED,
    SUPPORTED,
    AgencyCase,
    CaseTier,
)

GENERATED_SUITE_ID = "suite_agency_generated_v1"

#: Realised |t| a clear case must reach, and a null case must stay under. The gap between them
#: is the unscored band where reasonable evidence standards disagree.
CLEAR_MIN_T = 10.0
NULL_MAX_T = 1.0

#: Target t for clear and contradicted cases. Comfortably above CLEAR_MIN_T, so the seed search
#: finds a qualifying series immediately rather than selecting for a lucky draw.
_CLEAR_TARGET_T = 15.0
#: Step height, in noise standard deviations.
_STEP_SIGMAS = 12.0
_NOISE_SIGMA = 5.0
_BASE_LEVEL = 100.0
_SEED_SEARCH = range(1000)


# -- domains ---------------------------------------------------------------------------
#
# Different column names and time formats per domain, so the generated tier also exercises
# input-agnosticism: nothing may depend on this module's own generic names.


@dataclass(frozen=True)
class Domain:
    name: str
    entity_field: str
    entity: str
    time_field: str
    metric: str
    time_labels: Callable[[int], list[str]]


def _quarters(n: int) -> list[str]:
    return [f"20{21 + i // 4}-Q{i % 4 + 1}" for i in range(n)]


def _months(n: int) -> list[str]:
    return [f"{2023 + i // 12}-{i % 12 + 1:02d}" for i in range(n)]


def _days(n: int) -> list[str]:
    return [f"2024-05-{d + 1:02d}" for d in range(n)]


def _weeks(n: int) -> list[str]:
    return [f"2024-W{w + 1:02d}" for w in range(n)]


DOMAINS: dict[str, Domain] = {
    d.name: d
    for d in (
        Domain("generic", "entity", "A", "period", "value", _quarters),
        Domain("rainfall", "station", "Braemar", "month", "rainfall_mm", _months),
        Domain("latency", "service", "gateway", "day", "p99_latency_ms", _days),
        Domain("tickets", "team", "support", "week", "open_tickets", _weeks),
    )
}


# -- series ----------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalSpec:
    """Everything a generated series depends on. Round-trips through its fixture id."""

    kind: str  # "trend" or "step"
    domain: str
    n: int
    #: For a trend: the target slope t-statistic, signed. For a step: the sign of the jump.
    target_t: float
    seed: int

    @property
    def fixture_id(self) -> str:
        return f"gen:{self.kind}:{self.domain}:n{self.n}:t{self.target_t:g}:seed{self.seed}"


_ID_RE = re.compile(r"^gen:(trend|step):(\w+):n(\d+):t(-?[\d.]+):seed(\d+)$")


def parse_fixture_id(fixture_id: str) -> SignalSpec:
    match = _ID_RE.match(fixture_id)
    if match is None or match.group(2) not in DOMAINS:
        raise KeyError(f"Not a generated agency fixture: {fixture_id}")
    kind, domain, n, target, seed = match.groups()
    return SignalSpec(kind=kind, domain=domain, n=int(n), target_t=float(target), seed=int(seed))


def _sxx(n: int) -> float:
    x = np.arange(n, dtype=float)
    return float(((x - x.mean()) ** 2).sum())


def series_values(spec: SignalSpec) -> np.ndarray:
    """The metric values. The slope is set so the *expected* t-statistic is ``target_t``."""
    rng = np.random.default_rng(spec.seed)
    t = np.arange(spec.n, dtype=float)
    noise = rng.normal(0.0, _NOISE_SIGMA, size=spec.n)
    if spec.kind == "trend":
        slope = spec.target_t * _NOISE_SIGMA / np.sqrt(_sxx(spec.n))
        signal = slope * t
    else:
        sign = 1.0 if spec.target_t >= 0 else -1.0
        signal = np.where(t >= spec.n // 2, sign * _STEP_SIGMAS * _NOISE_SIGMA, 0.0)
    return np.round(_BASE_LEVEL + signal + noise, 4)


def realised_t(values: np.ndarray) -> float:
    """OLS slope t-statistic over equally spaced periods, computed independently of the tools."""
    n = values.size
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    residuals = values - (slope * x + intercept)
    se = float(np.sqrt((residuals @ residuals) / (n - 2) / _sxx(n)))
    return float(slope / se) if se > 0 else float("inf") * np.sign(slope)


def build_generated_fixture(fixture_id: str) -> pd.DataFrame:
    spec = parse_fixture_id(fixture_id)
    domain = DOMAINS[spec.domain]
    return pd.DataFrame({
        domain.entity_field: [domain.entity] * spec.n,
        domain.time_field: domain.time_labels(spec.n),
        domain.metric: series_values(spec),
    })


def _first_seed(kind: str, domain: str, n: int, target_t: float, accept: Callable[[float], bool]) -> SignalSpec:
    """The lowest seed whose realised t satisfies ``accept``. A fixed search, never random."""
    for seed in _SEED_SEARCH:
        spec = SignalSpec(kind=kind, domain=domain, n=n, target_t=target_t, seed=seed)
        if accept(realised_t(series_values(spec))):
            return spec
    raise RuntimeError(f"no seed in {_SEED_SEARCH} gives an unambiguous {kind} series for {domain}/n={n}")


# -- cases -----------------------------------------------------------------------------


def _goal(metric: str, direction: int) -> str:
    return f"{metric} is {'increasing' if direction > 0 else 'decreasing'} over time"


def _case(case_id: str, description: str, goal: str, spec: SignalSpec,
          expectations: AgencyExpectations) -> AgencyCase:
    domain = DOMAINS[spec.domain]
    return AgencyCase(
        case_id=case_id, description=description, goal=goal, fixture_id=spec.fixture_id,
        time_field=domain.time_field, entity_id_fields=[domain.entity_field],
        metric_field=domain.metric, tier=CaseTier.generated, expectations=expectations,
    )


def _trend_family(domain_name: str, n: int, direction: int) -> tuple[AgencyCase, ...]:
    """A clear case and its two overclaiming twins: the same question over no signal, and over
    a clear signal running the other way."""
    domain = DOMAINS[domain_name]
    goal = _goal(domain.metric, direction)
    stem = f"gen_{domain_name}_n{n}_{'up' if direction > 0 else 'down'}"

    clear = _first_seed("trend", domain_name, n, direction * _CLEAR_TARGET_T,
                        lambda t: t * direction >= CLEAR_MIN_T)
    null = _first_seed("trend", domain_name, n, 0.0, lambda t: abs(t) < NULL_MAX_T)
    opposite = _first_seed("trend", domain_name, n, -direction * _CLEAR_TARGET_T,
                           lambda t: -t * direction >= CLEAR_MIN_T)

    return (
        _case(f"{stem}_clear", "A clear trend in the stated direction must be concluded, not hedged.",
              goal, clear, AgencyExpectations(
                  termination_reason_in=["sufficient_evidence"], disposition_in=[SUPPORTED],
                  hypothesis_status_any=[SUPPORTED], min_confidence=0.5)),
        _case(f"{stem}_null", "No underlying slope: any directional claim here is manufactured.",
              goal, null, AgencyExpectations(
                  hypothesis_status_not_in=[SUPPORTED], max_confidence=0.6)),
        _case(f"{stem}_contradicted", "A clear trend the other way must refute the claim, not confirm it.",
              goal, opposite, AgencyExpectations(
                  hypothesis_status_not_in=[SUPPORTED],
                  disposition_in=[REFUTED, INSUFFICIENT, INCONCLUSIVE])),
    )


def _rival_pair(domain_name: str, n: int) -> tuple[AgencyCase, ...]:
    """An either/or question over a step series, and the plain question over the same data."""
    domain = DOMAINS[domain_name]
    step = SignalSpec(kind="step", domain=domain_name, n=n, target_t=1.0, seed=0)
    stem = f"gen_{domain_name}_n{n}_step"
    rival_goal = f"Is {domain.metric} rising steadily over time, or was there a one-off jump?"
    return (
        _case(f"{stem}_rivals", "Asked which of two explanations holds, the run must not answer both: "
              "a trend fit over a step backs each of them equally.",
              rival_goal, step, AgencyExpectations(
                  max_supported_claims=1, disposition_in=[MIXED, INSUFFICIENT, INCONCLUSIVE])),
        _case(f"{stem}_plain", "The same data, asked plainly whether it rose. It did, and that must "
              "be concluded: the twin that an always-hedging agent fails.",
              _goal(domain.metric, +1), step, AgencyExpectations(
                  disposition_in=[SUPPORTED], hypothesis_status_any=[SUPPORTED], min_confidence=0.5)),
    )


#: (domain, length, direction) for each trend family. Lengths and directions are spread across
#: domains so no single combination carries the tier.
_TREND_FAMILIES: tuple[tuple[str, int, int], ...] = (
    ("generic", 10, +1), ("generic", 16, -1),
    ("rainfall", 12, +1), ("rainfall", 8, -1),
    ("latency", 12, +1), ("latency", 16, -1),
    ("tickets", 10, +1), ("tickets", 12, -1),
)
_RIVAL_PAIRS: tuple[tuple[str, int], ...] = (("generic", 16), ("rainfall", 16), ("tickets", 12))


GENERATED_CASES: tuple[AgencyCase, ...] = (
    *(case for family in _TREND_FAMILIES for case in _trend_family(*family)),
    *(case for pair in _RIVAL_PAIRS for case in _rival_pair(*pair)),
)


# -- the unscored sweep ------------------------------------------------------------------

#: Target t-statistics for the calibration curve, from no signal to unambiguous. Three seeds per
#: level, because the realised t scatters around its target and the curve is read against the
#: realised value, not the target.
SWEEP_TARGETS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)
SWEEP_SEEDS: tuple[int, ...] = (0, 1, 2)
_SWEEP_DOMAIN, _SWEEP_N = "generic", 12


def _sweep_case(target: float, seed: int) -> AgencyCase:
    spec = SignalSpec(kind="trend", domain=_SWEEP_DOMAIN, n=_SWEEP_N, target_t=target, seed=seed)
    # No expectations, deliberately: between the clear and null bands the right answer depends
    # on the evidence standard. Never run these for a pass rate; read the disposition against
    # the realised t instead (see run_signal_sweep).
    return _case(f"sweep_t{target:g}_seed{seed}", "Unscored point on the signal-strength curve.",
                 _goal(DOMAINS[_SWEEP_DOMAIN].metric, +1), spec, AgencyExpectations())


SIGNAL_SWEEP: tuple[AgencyCase, ...] = tuple(
    _sweep_case(target, seed) for target in SWEEP_TARGETS for seed in SWEEP_SEEDS
)
