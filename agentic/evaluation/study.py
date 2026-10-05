"""
The scaffold-vs-model study's analysis: one honest headline, a failure taxonomy, and intervals.

S5 of ``docs/decisions/2026-10-05-scaffold-vs-model.md``. The scoreboard reports a pass rate per
row. That is the wrong number for this study, for two measured reasons:

1. **A run that produced no valid output scores as cautious.** A stub that never returned JSON
   passed 36% of core under the loop and 40% bare, entirely from cases built to punish
   overclaiming, which an empty answer cannot fail. Comparing small models on that number
   would credit them for failing to answer.
2. **Conditions do not assert the same properties.** The loop is scored on its route (tools,
   critique, budget); a bare answer has none. Only the answer properties are common to all
   three conditions.

So the headline here is the **honest pass**: a run passes when every answer property its case
asserts holds *and* no policy decision failed structurally. A transport failure (the provider
never answered) measures the machine, not the model, and is excluded rather than counted.

Each failing run is also given exactly one :class:`FailureClass`, so "the 1.7B model failed"
can be told apart from "the 1.7B model cannot write JSON".

Intervals come from a bootstrap over **cases**, not runs: at temperature 0 the trials of a case
barely vary, so the uncertainty that matters is which cases were chosen. Comparisons between
conditions resample the same cases for both sides (paired), which is what lets a modest case
count detect a real difference.

Pure: no I/O, no provider. Built from reports the runner persisted, so a study can be
re-analysed without re-running a single model call.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from enum import Enum

import numpy as np
from pydantic import Field

from agentic.domain.common import DomainModel
from agentic.evaluation.agency import (
    ANSWER_PROPERTIES,
    AgencyCaseResult,
    AgencyExpectations,
    AgencyReport,
    answer_is_scorable,
)
from agentic.evaluation.cases import AGENCY_CASES, SUPPORTED, AgencyCase
from agentic.evaluation.runner import SweepPoint

#: Policy failures that mean the model did not produce a usable decision. ``ungrounded`` is
#: included: a candidate index that does not exist is as much a failure to answer as bad JSON.
_STRUCTURAL = frozenset({"invalid_json", "schema", "ungrounded"})
_BUDGET_TERMINATIONS = frozenset({"budget_exhausted", "max_iterations", "safety_constraint"})
_HEDGES = frozenset({"insufficient_evidence", "inconclusive", None})


class FailureClass(str, Enum):
    """Why a run failed, exactly one per failing run, checked in this order."""

    transport = "transport"
    """The provider never answered. Excluded from rates: it measures the machine."""
    structural = "structural"
    """A decision came back as invalid JSON, the wrong schema, or a reference to nothing."""
    budget_exhaustion = "budget_exhaustion"
    """The run stopped on a resource limit before its answer held."""
    overclaim = "overclaim"
    """Concluded more than the data supports: a forbidden claim stood, a rival pair both stood,
    or confidence exceeded the case's ceiling."""
    underclaim = "underclaim"
    """Declined to conclude what the data establishes."""
    reasoning = "reasoning"
    """Any other wrong answer, e.g. the opposite direction, or `refuted` where `mixed` was right."""


def all_cases() -> dict[str, AgencyCase]:
    """Every case a study can contain, by id: hand-written and generated."""
    from agentic.evaluation.generated import GENERATED_CASES

    return {c.case_id: c for c in (*AGENCY_CASES, *GENERATED_CASES)}


def answer_passed(result: AgencyCaseResult) -> bool:
    """Every answer property the case asserts held. Route properties are ignored."""
    return all(o.passed for o in result.outcomes if o.property in ANSWER_PROPERTIES)


def _overclaimed(result: AgencyCaseResult, exp: AgencyExpectations) -> bool:
    statuses = result.observed_hypothesis_statuses
    disposition = result.observed_disposition
    if SUPPORTED in exp.hypothesis_status_not_in and SUPPORTED in statuses:
        return True
    if exp.disposition_in and SUPPORTED not in exp.disposition_in and disposition == SUPPORTED:
        return True
    if exp.max_confidence is not None and result.observed_confidence > exp.max_confidence:
        return True
    return exp.max_supported_claims is not None and statuses.count(SUPPORTED) > exp.max_supported_claims


def _underclaimed(result: AgencyCaseResult, exp: AgencyExpectations) -> bool:
    wants_conclusion = (SUPPORTED in exp.disposition_in) or (SUPPORTED in exp.hypothesis_status_any)
    if wants_conclusion and result.observed_disposition in _HEDGES:
        return True
    return exp.min_confidence is not None and result.observed_confidence < exp.min_confidence


def classify(result: AgencyCaseResult, expectations: AgencyExpectations) -> FailureClass | None:
    """The single reason a run failed on its answer, or ``None`` when it did not."""
    failure = result.observed_policy_failure
    if failure is not None and failure.kind.value == "transport":
        return FailureClass.transport
    if failure is not None and failure.kind.value in _STRUCTURAL:
        return FailureClass.structural
    if answer_passed(result):
        return None
    if result.observed_termination in _BUDGET_TERMINATIONS:
        return FailureClass.budget_exhaustion
    if _overclaimed(result, expectations):
        return FailureClass.overclaim
    if _underclaimed(result, expectations):
        return FailureClass.underclaim
    return FailureClass.reasoning


class Observation(DomainModel):
    """One run of one case in one cell of the study design."""

    model: str
    #: Parameters in billions, for the size axis. ``None`` for a reference model of unknown
    #: size; ``0`` for the rule-based policy (the scaffold with no model).
    size_b: float | None = None
    condition: str
    tier: str
    trial: int
    case_id: str
    #: ``None`` when the run is excluded (transport): it says nothing about the model.
    honest_pass: bool | None
    #: The scoreboard's own verdict, kept beside the honest one so the gap is visible.
    raw_pass: bool
    failure: FailureClass | None = None


def observations(
    reports: Sequence[AgencyReport], *, model: str, size_b: float | None, condition: str, tier: str,
    cases: dict[str, AgencyCase] | None = None,
) -> list[Observation]:
    """Flatten one cell's trial reports, keeping only cases that assert something about the answer."""
    known = cases if cases is not None else all_cases()
    out: list[Observation] = []
    for trial, report in enumerate(reports):
        for result in report.results:
            case = known.get(result.case_id)
            if case is None or not answer_is_scorable(case.expectations):
                continue
            failure = classify(result, case.expectations)
            honest = None if failure is FailureClass.transport else failure is None
            out.append(Observation(
                model=model, size_b=size_b, condition=condition, tier=tier, trial=trial,
                case_id=result.case_id, honest_pass=honest, raw_pass=result.passed, failure=failure,
            ))
    return out


# -- intervals -------------------------------------------------------------------------------

#: Fixed so a report is reproducible from the persisted runs alone.
BOOTSTRAP_SEED = 20261005
BOOTSTRAP_RESAMPLES = 4000


def _per_case(obs: Iterable[Observation]) -> dict[str, float]:
    """Mean honest pass per case over its trials, excluding transport-failed runs."""
    by_case: dict[str, list[bool]] = {}
    for o in obs:
        if o.honest_pass is not None:
            by_case.setdefault(o.case_id, []).append(o.honest_pass)
    return {cid: float(np.mean(v)) for cid, v in by_case.items() if v}


def _interval(samples: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(samples, [2.5, 97.5])
    return round(float(low), 4), round(float(high), 4)


class Cell(DomainModel):
    """One model × condition × tier: the honest pass rate with a 95% case-bootstrap interval."""

    model: str
    size_b: float | None = None
    condition: str
    tier: str
    rate: float
    ci_low: float
    ci_high: float
    #: The scoreboard's pass rate over the same runs, for contrast.
    raw_rate: float
    n_cases: int
    n_runs: int
    excluded_transport: int = 0
    failures: dict[str, int] = Field(default_factory=dict)


def cell(obs: Sequence[Observation]) -> Cell:
    """Summarise one cell. All observations must share model, condition and tier."""
    first = obs[0]
    per_case = _per_case(obs)
    values = np.array(list(per_case.values()))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    if values.size:
        draws = rng.integers(0, values.size, size=(BOOTSTRAP_RESAMPLES, values.size))
        low, high = _interval(values[draws].mean(axis=1))
        rate = round(float(values.mean()), 4)
    else:
        low = high = rate = 0.0
    counted = [o for o in obs if o.honest_pass is not None]
    return Cell(
        model=first.model, size_b=first.size_b, condition=first.condition, tier=first.tier,
        rate=rate, ci_low=low, ci_high=high,
        raw_rate=round(float(np.mean([o.raw_pass for o in counted])), 4) if counted else 0.0,
        n_cases=len(per_case), n_runs=len(counted),
        excluded_transport=sum(1 for o in obs if o.honest_pass is None),
        failures=dict(sorted(Counter(o.failure.value for o in obs if o.failure is not None).items())),
    )


class PairedDifference(DomainModel):
    """Condition ``a`` minus condition ``b`` for one model and tier, over the cases both ran."""

    model: str
    tier: str
    a: str
    b: str
    diff: float
    ci_low: float
    ci_high: float
    n_cases: int

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


def paired(a: Sequence[Observation], b: Sequence[Observation]) -> PairedDifference:
    """
    Honest pass rate of ``a`` minus ``b``, resampling the *same* cases for both.

    Only cases both conditions scored are compared, so a condition that skips route-only cases
    (C) is compared with A and B on the cases it shares with them, never on a different set.
    """
    pa, pb = _per_case(a), _per_case(b)
    shared = sorted(set(pa) & set(pb))
    deltas = np.array([pa[c] - pb[c] for c in shared])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    if deltas.size:
        draws = rng.integers(0, deltas.size, size=(BOOTSTRAP_RESAMPLES, deltas.size))
        low, high = _interval(deltas[draws].mean(axis=1))
        diff = round(float(deltas.mean()), 4)
    else:
        low = high = diff = 0.0
    return PairedDifference(model=a[0].model, tier=a[0].tier, a=a[0].condition, b=b[0].condition,
                            diff=diff, ci_low=low, ci_high=high, n_cases=len(shared))


# -- the signal-strength curve --------------------------------------------------------------------


class SweepLevel(DomainModel):
    """One target signal level: how often a trend was claimed, at the realised strength."""

    target_t: float
    #: Mean realised t over the level's points. The curve is read against this, not the target.
    mean_realised_t: float
    claim_rate: float
    n: int


class SweepCurve(DomainModel):
    """
    Where a configuration starts claiming a trend, from no signal to unambiguous.

    Unscored on purpose: between clear and null the right answer depends on the evidence
    standard, so a curve is compared with other curves (the full loop's, the rule-based
    policy's), not with an answer key. ``claim_rate`` at target 0 is the rate of claims made
    about pure noise.
    """

    model: str
    size_b: float | None = None
    condition: str
    levels: list[SweepLevel] = Field(default_factory=list)
    #: Realised t at which the claim rate first reaches 50%, interpolated between levels.
    #: ``None`` when it never crosses inside the sweep; ``t50_note`` then says which way.
    t50: float | None = None
    t50_note: str = ""
    #: Points whose call failed (transport or structural). A failure is not a judgement about
    #: the data, so it is left out of the curve rather than counted as "did not claim".
    excluded: int = 0


def sweep_curve(points: Sequence[SweepPoint], *, model: str, size_b: float | None, condition: str) -> SweepCurve:
    decided = [p for p in points if p.policy_failure is None]
    by_target: dict[float, list[SweepPoint]] = {}
    for p in decided:
        by_target.setdefault(p.target_t, []).append(p)
    levels = [
        SweepLevel(target_t=t, mean_realised_t=round(float(np.mean([p.realised_t for p in pts])), 3),
                   claim_rate=round(float(np.mean([p.claimed for p in pts])), 4), n=len(pts))
        for t, pts in sorted(by_target.items())
    ]
    t50, note = None, ""
    crossing = next((i for i, lv in enumerate(levels) if lv.claim_rate >= 0.5), None)
    if not levels:
        note = "no decided points"
    elif crossing is None:
        note = "never claims"
    elif crossing == 0:
        note = "claims at every level"
    else:
        lo, hi = levels[crossing - 1], levels[crossing]
        frac = (0.5 - lo.claim_rate) / (hi.claim_rate - lo.claim_rate)
        t50 = round(lo.mean_realised_t + frac * (hi.mean_realised_t - lo.mean_realised_t), 3)
    return SweepCurve(model=model, size_b=size_b, condition=condition, levels=levels,
                      t50=t50, t50_note=note, excluded=len(points) - len(decided))


# -- the report ---------------------------------------------------------------------------------


class StudyReport(DomainModel):
    cells: list[Cell] = Field(default_factory=list)
    comparisons: list[PairedDifference] = Field(default_factory=list)
    sweeps: list[SweepCurve] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "## Honest pass rate",
            "",
            "Answer properties only, with structural failures counted as failed and transport "
            "failures excluded. 95% intervals from a case bootstrap. `raw` is the scoreboard's "
            "pass rate over the same runs.",
            "",
            "| model | size (B) | condition | tier | honest | 95% CI | raw | cases | runs | failures |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for c in sorted(self.cells, key=lambda c: (c.tier, c.size_b is None, c.size_b or 0, c.model, c.condition)):
            size = "—" if c.size_b is None else f"{c.size_b:g}"
            failures = ", ".join(f"{k} {v}" for k, v in c.failures.items()) or "—"
            if c.excluded_transport:
                failures += f" (excluded: {c.excluded_transport})"
            lines.append(
                f"| {c.model} | {size} | {c.condition} | {c.tier} | {c.rate:.0%} | "
                f"{c.ci_low:.0%}–{c.ci_high:.0%} | {c.raw_rate:.0%} | {c.n_cases} | {c.n_runs} | {failures} |")
        if self.comparisons:
            lines += [
                "",
                "## Paired comparisons",
                "",
                "Honest pass rate of the first condition minus the second, over the cases both "
                "scored. **Bold** where the 95% interval excludes zero.",
                "",
                "| model | tier | comparison | difference | 95% CI | cases |",
                "|---|---|---|---|---|---|",
            ]
            for d in self.comparisons:
                diff = f"{d.diff:+.0%}"
                lines.append(
                    f"| {d.model} | {d.tier} | {d.a} − {d.b} | {f'**{diff}**' if d.excludes_zero else diff} | "
                    f"{d.ci_low:+.0%} to {d.ci_high:+.0%} | {d.n_cases} |")
        if self.sweeps:
            targets = sorted({lv.target_t for c in self.sweeps for lv in c.levels})
            lines += [
                "",
                "## Signal-strength curve",
                "",
                "Share of runs claiming a trend at each target signal level (slope t-statistic; "
                "the realised value scatters around it). Unscored: compare curves with each "
                "other, not with an answer key. `t50` is the realised t where the claim rate "
                "first reaches 50%; the `0` column is claims made about pure noise.",
                "",
                "| model | size (B) | condition | t50 | " + " | ".join(f"{t:g}" for t in targets) + " | excluded |",
                "|---|---|---|---|" + "---|" * len(targets) + "---|",
            ]
            for c in sorted(self.sweeps, key=lambda c: (c.size_b is None, c.size_b or 0, c.model, c.condition)):
                rates = {lv.target_t: lv.claim_rate for lv in c.levels}
                size = "—" if c.size_b is None else f"{c.size_b:g}"
                t50 = f"{c.t50:.2f}" if c.t50 is not None else c.t50_note
                cells = " | ".join(f"{rates[t]:.0%}" if t in rates else "—" for t in targets)
                lines.append(f"| {c.model} | {size} | {c.condition} | {t50} | {cells} | {c.excluded} |")
        return "\n".join(lines)


#: The full loop. Every other condition is compared against it.
REFERENCE_CONDITION = "A"


def analyse(obs: Sequence[Observation], sweeps: Sequence[SweepCurve] = ()) -> StudyReport:
    """Every cell, and each condition compared with the full loop for the same model and tier."""
    groups: dict[tuple[str, str, str], list[Observation]] = {}
    for o in obs:
        groups.setdefault((o.model, o.condition, o.tier), []).append(o)
    cells = [cell(g) for g in groups.values()]
    comparisons = []
    for (model, condition, tier), group in sorted(groups.items()):
        reference = groups.get((model, REFERENCE_CONDITION, tier))
        if condition != REFERENCE_CONDITION and reference:
            comparisons.append(paired(reference, group))
    return StudyReport(cells=cells, comparisons=comparisons, sweeps=list(sweeps))
