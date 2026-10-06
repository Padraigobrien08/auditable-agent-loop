"""
Run the agency suite against the investigation loop.

Offline and deterministic by construction: fixtures are generated from fixed formulas or a
seeded RNG, the default policy is the deterministic fixture policy, and every check reads
persisted typed state. The same suite can be pointed at a model-backed policy to ask whether
a real model reasons as well as the deterministic baseline — which is the point of having it.
"""

from __future__ import annotations

from agentic.adapters import AdapterRequest, InMemoryDatasetAdapter
from agentic.agent.budget import LoopBudget, SafetyLimits
from agentic.agent.loop import InvestigationLoop
from agentic.agent.observer import AgentObserver
from agentic.agent.policy import AgentPolicy
from agentic.domain.enums import ColumnRole
from agentic.evaluation.agency import AgencyCaseResult, AgencyReport, score_case
from agentic.evaluation.cases import AGENCY_CASES, SUITE_ID, AgencyCase, CaseTier, cases_for_tier
from agentic.evaluation.fixtures import build_fixture


def run_case(
    case: AgencyCase,
    *,
    policy: AgentPolicy | None = None,
    budget: LoopBudget | None = None,
    safety: SafetyLimits | None = None,
    observer: AgentObserver | None = None,
) -> AgencyCaseResult:
    """Run one agency case end to end and score it.

    ``observer`` is forwarded to the loop; the benchmark uses it to capture each run's
    cost and wall time from ``InvestigationEnded`` rather than measuring them separately.
    """
    frame = build_fixture(case.fixture_id)
    manifest = InMemoryDatasetAdapter(
        frame=frame,
        time_field=case.time_field,
        entity_id_fields=list(case.entity_id_fields),
        role_hints={case.metric_field: ColumnRole.metric},
    ).build_manifest(AdapterRequest())

    # A case's own max_experiments is part of what it asserts (see `budget_is_respected`), so
    # it wins over a caller-supplied budget rather than being overwritten by it. A caller
    # bounding cost or time therefore does not silently disarm the budget cases.
    effective_budget = budget
    if case.max_experiments is not None:
        effective_budget = (
            LoopBudget(max_experiments=case.max_experiments)
            if budget is None
            else budget.model_copy(update={"max_experiments": case.max_experiments})
        )

    # Built as kwargs so an omitted policy/observer keeps the loop's own field default
    # rather than being overwritten with None.
    loop_kwargs: dict[str, object] = {}
    if policy is not None:
        loop_kwargs["policy"] = policy
    if observer is not None:
        loop_kwargs["observer"] = observer
    loop = InvestigationLoop(**loop_kwargs)  # type: ignore[arg-type]
    investigation = loop.start(
        case.goal,
        manifest=manifest,
        frame=frame,
        # Seeded from the case id so ids — and therefore any diff against a prior run —
        # are stable across executions.
        seed=f"agency-{case.case_id}",
        budget=effective_budget,
        safety=safety,
    )
    return score_case(
        case.case_id, investigation, case.expectations, description=case.description
    )


def run_agency_suite(
    *,
    policy: AgentPolicy | None = None,
    cases: tuple[AgencyCase, ...] = AGENCY_CASES,
    suite_id: str = SUITE_ID,
    observer: AgentObserver | None = None,
    budget: LoopBudget | None = None,
    tier: CaseTier | None = None,
    safety: SafetyLimits | None = None,
) -> AgencyReport:
    """Run every case and aggregate a report.

    A single ``observer`` spans the whole suite, so one run collects one
    ``InvestigationEnded`` per case. ``budget`` applies to every case except where a case
    pins its own ``max_experiments`` — see :func:`run_case`.

    ``tier`` narrows to one tier. Core and hard measure different things — the core tier is
    saturated by design history, the hard tier is where the headroom is — so a caller
    comparing policies should report them separately rather than averaging them.
    """
    if tier is not None:
        cases = cases_for_tier(tier, cases)
    results = [
        run_case(case, policy=policy, observer=observer, budget=budget, safety=safety) for case in cases
    ]
    return AgencyReport(
        suite_id=suite_id,
        total=len(results),
        passed=sum(1 for r in results if r.passed),
        results=results,
    )


def format_report(report: AgencyReport) -> str:
    """Human-readable report for a CLI or CI log."""
    lines = [report.summary(), ""]
    lines.append("Per-property pass rate:")
    for name, score in report.property_scores().items():
        lines.append(f"  {score:>6.0%}  {name}")
    failures = [r for r in report.results if not r.passed]
    if failures:
        lines.append("")
        lines.append("Failures:")
        for result in failures:
            lines.append(f"  {result.case_id} — {result.description}")
            for outcome in result.failures:
                lines.append(f"      {outcome.property.value}: {outcome.detail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """
    ``python -m agentic.evaluation`` — run a tier and print a report.

    Defaults to the **core** tier, because this command's exit code is documented as a gate
    and the hard tier is designed for the deterministic baseline to fail. Running everything
    by default would leave the gate permanently red and say nothing when a real regression
    arrived.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="python -m agentic.evaluation")
    parser.add_argument(
        "--tier",
        choices=[t.value for t in CaseTier] + ["all"],
        default=CaseTier.core.value,
        help="Which tier to run (default: core, the tier the baseline is expected to pass).",
    )
    args = parser.parse_args(argv)

    tier = None if args.tier == "all" else CaseTier(args.tier)
    report = run_agency_suite(tier=tier)
    print(format_report(report))

    if tier is CaseTier.core:
        hard = len(cases_for_tier(CaseTier.hard))
        if hard:
            print(
                f"\n{hard} hard case(s) not run. The deterministic baseline is expected to "
                "fail these — that gap is the suite's headroom. `--tier hard` to see them."
            )
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
