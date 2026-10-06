"""
Condition C: the model alone. One call, given the table and the question, no loop.

Condition C of ``docs/decisions/2026-10-05-scaffold-vs-model.md``. The study asks how much of
an honest conclusion comes from the scaffold and how much from the model; this is the model
with the scaffold removed entirely. No tools run, no evidence is recorded, nothing is computed
deterministically: the model reads the raw rows and states a disposition, the claims it
considered, and a confidence.

**This deliberately breaks the project's invariant** that no number originates from a model:
the confidence here is the model's own, and any figure in its rationale is unchecked. That is
the point of a control, and it is why this lives in ``backend/dev`` and never in ``agentic/``.
Nothing here is persisted or reachable from a user's run.

It is scored by :func:`agentic.evaluation.agency.score_answer`, the same checks
:func:`~agentic.evaluation.agency.score_case` applies to the loop's conclusion, restricted to
the properties an answer can have (:data:`~agentic.evaluation.agency.ANSWER_PROPERTIES`).
Cases that assert only something about the route (which tools ran, whether a claim was
challenged) are skipped, because there is no route to score. A C row is therefore comparable
with an A or B row on ``reaches_the_right_disposition`` and the other answer properties, not
on the overall pass rate.

Transport is the same as the loop's: :class:`~backend.agents.agentic_model_policy.CostTrackingResponder`,
temperature 0, ``json_object`` mode. A failure is recorded with the same typed
:class:`~agentic.domain.PolicyFailure`, under ``PolicyDecisionKind.bare_answer``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Literal

from pydantic import Field, ValidationError

from agentic.agent.policy import AgentPolicyError, PolicyTransportError, Responder
from agentic.domain import PolicyDecisionKind, PolicyFailure, PolicyFailureKind
from agentic.domain.common import DomainModel
from agentic.evaluation.agency import AgencyCaseResult, AgencyReport, answer_is_scorable, score_answer
from agentic.evaluation.cases import AGENCY_CASES, SUITE_ID, AgencyCase, CaseTier, cases_for_tier
from agentic.evaluation.fixtures import build_fixture
from agentic.evaluation.runner import SweepPoint
from agentic.evaluation.scoreboard import RunMetrics

#: Recorded with every C row, as the loop's prompt version is with every A/B row. Bump on any
#: change to :data:`SYSTEM_PROMPT` or the answer schema.
BARE_PROMPT_VERSION = "1.0.0"

#: Kept close to the loop's prompts in what it asks for: the same dispositions, the same
#: permission to decline, and no hint about which answer a case expects.
SYSTEM_PROMPT = (
    "You are given a data table as CSV and a question about it. Answer the question from the "
    "data alone.\n"
    "Reply as JSON with exactly these fields:\n"
    '  "disposition": one of "supported", "refuted", "mixed", "inconclusive", '
    '"insufficient_evidence". Use "supported" when the data establishes what the question '
    'asks, "refuted" when it establishes the opposite, "mixed" when the question has several '
    'parts and only some hold, "inconclusive" when the evidence points both ways, and '
    '"insufficient_evidence" when the data cannot establish an answer.\n'
    '  "claims": a list of the claims you weighed, each {"statement": string, "status": one '
    'of "supported", "rejected", "weakened", "unresolved"}. If the question asks which of two '
    "explanations holds, list both.\n"
    '  "confidence": a number from 0 to 1 for the overall answer.\n'
    '  "rationale": one or two sentences.\n'
    "Do not claim more than the data shows. Declining to conclude is a valid answer."
)


class BareClaim(DomainModel):
    statement: str = Field(..., min_length=1)
    status: Literal["supported", "rejected", "weakened", "unresolved"]


class BareAnswer(DomainModel):
    disposition: Literal["supported", "refuted", "mixed", "inconclusive", "insufficient_evidence"]
    claims: list[BareClaim] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = ""


def user_prompt(case: AgencyCase) -> str:
    """The question and the full table. Nothing about the expected answer."""
    frame = build_fixture(case.fixture_id)
    return json.dumps({"question": case.goal, "table_csv": frame.to_csv(index=False)})


def _ask(respond: Responder, case: AgencyCase) -> BareAnswer:
    raw = respond(SYSTEM_PROMPT, user_prompt(case))
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise _BareFailure(PolicyFailureKind.invalid_json, f"not valid JSON: {exc}") from exc
    try:
        return BareAnswer.model_validate(data)
    except ValidationError as exc:
        raise _BareFailure(PolicyFailureKind.schema, f"failed BareAnswer validation: {exc}") from exc


class _BareFailure(AgentPolicyError):
    def __init__(self, kind: PolicyFailureKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind


_DETAIL_LIMIT = 500


def run_bare_case(case: AgencyCase, respond: Responder) -> AgencyCaseResult:
    """
    Ask once and score the answer.

    A failed call is scored as an empty answer: no disposition, no claims, zero confidence. That
    is the same thing a failed loop run concludes, so the raw rows stay comparable; the failure
    itself is recorded, so a headline metric can count it as failed rather than as caution.
    """
    failure: PolicyFailure | None = None
    try:
        answer: BareAnswer | None = _ask(respond, case)
    except PolicyTransportError as exc:
        answer, failure = None, PolicyFailure(
            decision=PolicyDecisionKind.bare_answer, kind=PolicyFailureKind.transport,
            detail=str(exc)[:_DETAIL_LIMIT])
    except _BareFailure as exc:
        answer, failure = None, PolicyFailure(
            decision=PolicyDecisionKind.bare_answer, kind=exc.kind, detail=str(exc)[:_DETAIL_LIMIT])

    disposition = answer.disposition if answer is not None else None
    statuses: list[str] = [str(c.status) for c in answer.claims] if answer is not None else []
    confidence = answer.confidence if answer is not None else 0.0
    outcomes = score_answer(case.expectations, disposition=disposition, statuses=statuses,
                            confidence=confidence)
    return AgencyCaseResult(
        case_id=case.case_id, description=case.description,
        passed=all(o.passed for o in outcomes), outcomes=outcomes,
        observed_disposition=disposition, observed_hypothesis_statuses=statuses,
        observed_confidence=round(confidence, 6), observed_policy_failure=failure,
    )


def bare_cases(tier: CaseTier | None = None) -> tuple[AgencyCase, ...]:
    """The cases C can be scored on: those asserting something about the answer."""
    if tier is CaseTier.generated:
        from agentic.evaluation.generated import GENERATED_CASES

        cases: tuple[AgencyCase, ...] = GENERATED_CASES
    else:
        cases = AGENCY_CASES if tier is None else cases_for_tier(tier)
    return tuple(c for c in cases if answer_is_scorable(c.expectations))


def run_bare_suite(
    respond: Responder,
    *,
    tier: CaseTier | None = None,
    drain_cost: Callable[[], float] = lambda: 0.0,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[AgencyReport, list[RunMetrics]]:
    """
    One trial of condition C over a tier, with per-case cost and latency.

    Returns the same :class:`AgencyReport` and :class:`RunMetrics` the loop's trials produce,
    so C rows aggregate through :func:`~agentic.evaluation.scoreboard.aggregate_trials`
    unchanged.
    """
    if tier is CaseTier.generated:
        from agentic.evaluation.generated import GENERATED_SUITE_ID

        suite_id = GENERATED_SUITE_ID
    else:
        suite_id = SUITE_ID
    results: list[AgencyCaseResult] = []
    metrics: list[RunMetrics] = []
    for case in bare_cases(tier):
        started = clock()
        results.append(run_bare_case(case, respond))
        metrics.append(RunMetrics(investigation_id=f"bare-{case.case_id}", cost_usd=drain_cost(),
                                  elapsed_seconds=clock() - started, model_calls=1))
    report = AgencyReport(suite_id=suite_id, total=len(results),
                          passed=sum(1 for r in results if r.passed), results=results)
    return report, metrics


def run_bare_sweep(respond: Responder) -> list[SweepPoint]:
    """
    Condition C over the unscored signal sweep: whether the model alone claims a trend, per point.

    The bare counterpart of :func:`agentic.evaluation.runner.run_signal_sweep`, producing the same
    points so the curves can be read side by side.
    """
    from agentic.evaluation.generated import SIGNAL_SWEEP, parse_fixture_id, realised_t, series_values

    points: list[SweepPoint] = []
    for case in SIGNAL_SWEEP:
        spec = parse_fixture_id(case.fixture_id)
        result = run_bare_case(case, respond)
        failure = result.observed_policy_failure
        points.append(SweepPoint(
            case_id=case.case_id, target_t=spec.target_t,
            realised_t=round(realised_t(series_values(spec)), 4),
            claimed="supported" in result.observed_hypothesis_statuses,
            disposition=result.observed_disposition,
            policy_failure=f"{failure.decision.value}:{failure.kind.value}" if failure else None,
        ))
    return points
