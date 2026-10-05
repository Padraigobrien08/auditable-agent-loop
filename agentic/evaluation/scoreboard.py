"""
Reducing many suite runs to one honest scorecard.

A single pass of the agency suite is a measurement only when the policy under test is
deterministic. A model-backed policy is not: the same case can pass on one trial and fail on
the next, and averaging that into a bare pass rate hides exactly the thing a reader needs to
know. So this module keeps two facts side by side — how often each property held, and whether
the verdict was *stable* — and reports a case as unstable rather than quietly averaging it.

Cost and latency come from :class:`~agentic.agent.observer.InvestigationEnded` events captured
by :class:`MetricsObserver`. They are recorded by the loop itself, so quality and spend are
measured on the same run rather than stitched together from separate ones.

Everything here is pure: no I/O, no clock, no provider. Aggregation is a function of the
reports and metrics handed to it, which is what makes the scoreboard testable from synthetic
inputs and keeps this package free of the backend.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from agentic.agent.observer import AgentObserver, InvestigationEnded
from agentic.domain.common import DomainModel
from agentic.evaluation.agency import AgencyReport


class RunMetrics(DomainModel):
    """Resource cost of one investigation, as the loop reported it."""

    investigation_id: str = ""
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    model_calls: int = 0


class MetricsObserver(AgentObserver):
    """
    Captures one :class:`RunMetrics` per finished investigation.

    Only terminal events are recorded — a partial (resumable) return is not a completed
    investigation and would double-count the run it belongs to.
    """

    def __init__(self) -> None:
        self.runs: list[RunMetrics] = []

    def on_investigation_end(self, event: InvestigationEnded) -> None:
        if event.partial:
            return
        self.runs.append(
            RunMetrics(
                investigation_id=event.investigation_id,
                cost_usd=event.cost_usd,
                elapsed_seconds=event.elapsed_seconds,
                model_calls=event.model_calls,
            )
        )

    def drain(self) -> list[RunMetrics]:
        """Return everything captured so far and reset, so one observer can span trials."""
        captured, self.runs = self.runs, []
        return captured


class CaseStability(DomainModel):
    """How often one case passed across the trials that ran it."""

    case_id: str
    passed_trials: int = 0
    total_trials: int = 0

    @property
    def stable(self) -> bool:
        """True when every trial agreed — all passed or all failed."""
        return self.passed_trials in (0, self.total_trials)

    @property
    def pass_rate(self) -> float:
        return (self.passed_trials / self.total_trials) if self.total_trials else 0.0


class PolicyScorecard(DomainModel):
    """One policy's result over N trials of one tier."""

    label: str
    #: Which tier this row measures. Core and hard mean different things — core is saturated
    #: by design history, hard is where the headroom is — so they are never averaged into one
    #: number. An empty value means the row spans every tier.
    tier: str = ""
    trials: int = 0
    mean_pass_rate: float = 0.0
    property_means: dict[str, float] = Field(default_factory=dict)
    #: Only cases whose verdict was **not** unanimous; a stable suite leaves this empty.
    unstable_cases: list[CaseStability] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    mean_cost_usd: float = 0.0
    p95_latency_seconds: float = 0.0
    #: Set when trials were cut short, e.g. by a suite-level cost ceiling. A truncated row
    #: is still reportable but must not be read as an N-trial result.
    truncated: bool = False
    #: Runs a policy failure ended, keyed ``decision:kind`` (e.g. ``critique:schema``), summed
    #: over trials. Beside the pass rate because a failing row means different things
    #: depending on what is in here: a model that reasons badly, one that cannot produce valid
    #: output, or a provider that never answered.
    policy_failures: dict[str, int] = Field(default_factory=dict)

    @property
    def fully_stable(self) -> bool:
        return not self.unstable_cases


def _p95(values: Sequence[float]) -> float:
    """Nearest-rank p95, which stays meaningful at the small trial counts used here."""
    if not values:
        return 0.0
    ordered = sorted(values)
    # ceil(0.95 * n) as an index, clamped — for n < 20 this is simply the maximum.
    rank = max(1, -(-95 * len(ordered) // 100))
    return ordered[rank - 1]


def aggregate_trials(
    label: str,
    reports: Sequence[AgencyReport],
    metrics: Sequence[RunMetrics] = (),
    *,
    truncated: bool = False,
    tier: str = "",
) -> PolicyScorecard:
    """
    Reduce N suite reports (and the metrics captured alongside them) to one scorecard.

    Property means are taken across the reports that asserted each property, so a property no
    case exercises is absent rather than counted as zero.
    """
    if not reports:
        return PolicyScorecard(label=label, tier=tier, truncated=truncated)

    stability: dict[str, CaseStability] = {}
    for report in reports:
        for result in report.results:
            entry = stability.setdefault(result.case_id, CaseStability(case_id=result.case_id))
            entry.total_trials += 1
            entry.passed_trials += 1 if result.passed else 0

    property_totals: dict[str, list[float]] = {}
    for report in reports:
        for name, score in report.property_scores().items():
            property_totals.setdefault(name, []).append(score)

    failures: dict[str, int] = {}
    for report in reports:
        for result in report.results:
            failure = result.observed_policy_failure
            if failure is not None:
                key = f"{failure.decision.value}:{failure.kind.value}"
                failures[key] = failures.get(key, 0) + 1

    latencies = [m.elapsed_seconds for m in metrics]
    total_cost = sum(m.cost_usd for m in metrics)

    return PolicyScorecard(
        label=label,
        tier=tier,
        trials=len(reports),
        mean_pass_rate=round(sum(r.pass_rate for r in reports) / len(reports), 4),
        property_means={
            name: round(sum(scores) / len(scores), 4)
            for name, scores in sorted(property_totals.items())
        },
        unstable_cases=[s for _, s in sorted(stability.items()) if not s.stable],
        total_cost_usd=round(total_cost, 6),
        mean_cost_usd=round(total_cost / len(reports), 6),
        p95_latency_seconds=round(_p95(latencies), 4),
        truncated=truncated,
        policy_failures=dict(sorted(failures.items())),
    )


class Scoreboard(DomainModel):
    """Several policies measured on the same suite, ready to publish."""

    suite_id: str
    rows: list[PolicyScorecard] = Field(default_factory=list)

    def property_names(self) -> list[str]:
        """Every property any row scored, so the table has stable columns."""
        names: set[str] = set()
        for row in self.rows:
            names.update(row.property_means)
        return sorted(names)

    def to_markdown(self) -> str:
        """A publishable table: headline quality, then spend, then stability."""
        if not self.rows:
            return f"_No results for {self.suite_id}._"

        properties = self.property_names()
        tiered = any(row.tier for row in self.rows)
        header = ["policy", *(["tier"] if tiered else []), "trials", "pass rate",
                  *properties, "mean $", "p95 s"]
        lines = [
            f"| {' | '.join(header)} |",
            f"|{'|'.join(['---'] * len(header))}|",
        ]
        for row in self.rows:
            label = f"{row.label} ⚠️" if row.truncated else row.label
            cells = [
                label,
                *([row.tier or "all"] if tiered else []),
                str(row.trials),
                f"{row.mean_pass_rate:.0%}",
                *[
                    f"{row.property_means[p]:.0%}" if p in row.property_means else "—"
                    for p in properties
                ],
                f"{row.mean_cost_usd:.4f}",
                f"{row.p95_latency_seconds:.2f}",
            ]
            lines.append(f"| {' | '.join(cells)} |")

        unstable = [(r.label, c) for r in self.rows for c in r.unstable_cases]
        if unstable:
            lines.append("")
            lines.append("**Unstable cases** — verdict changed across trials:")
            lines.append("")
            for label, case in unstable:
                lines.append(
                    f"- `{label}` / `{case.case_id}` — passed "
                    f"{case.passed_trials}/{case.total_trials} trials"
                )

        failing = [r for r in self.rows if r.policy_failures]
        if failing:
            lines.append("")
            lines.append("**Policy failures**: runs a policy decision ended, as `decision:kind`:")
            lines.append("")
            for row in failing:
                counts = ", ".join(f"`{k}` × {v}" for k, v in row.policy_failures.items())
                lines.append(f"- `{row.label}` / {row.tier or 'all'}: {counts}")

        truncated = [r.label for r in self.rows if r.truncated]
        if truncated:
            lines.append("")
            lines.append(
                "⚠️ Truncated by the cost ceiling (fewer trials than requested): "
                + ", ".join(f"`{label}`" for label in truncated)
            )
        return "\n".join(lines)
