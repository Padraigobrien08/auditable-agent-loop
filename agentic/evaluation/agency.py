"""
Agency evaluation — measuring whether the loop reasons well, not whether it runs.

The existing benchmark suites check *outputs*: did the pipeline produce the right artifacts
with the right numbers. That is necessary and stays unchanged. It says nothing about the
question an agentic system actually has to answer well: given evidence, does it draw the right
conclusion, revise when contradicted, and decline when the data cannot support a claim?

This module scores those properties. Every check is **deterministic** — derived from persisted
typed state, never from a model judging a model — so a case's verdict is reproducible and a
regression is unambiguous.

The scoring deliberately treats two opposite failures as equally bad:

* **Overclaiming** — asserting a trend in flat or noisy data, or confirming a hypothesis the
  evidence contradicts.
* **Underclaiming** — concluding "insufficient evidence" on an unambiguous signal. An agent
  that always hedges is never wrong and never useful, and a suite without positive controls
  would score it perfectly.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from agentic.domain import Investigation, PolicyFailure
from agentic.domain.common import DomainModel


class AgencyProperty(str, Enum):
    """The behaviors a competent investigation loop must exhibit."""

    terminates_for_the_right_reason = "terminates_for_the_right_reason"
    """Stops with the typed reason the evidence warrants."""

    reaches_the_right_disposition = "reaches_the_right_disposition"
    """Concludes supported / refuted / insufficient in line with the data."""

    revises_under_contradiction = "revises_under_contradiction"
    """A hypothesis the evidence opposes does not end up supported.

    Includes a rival the goal itself opposes: asked *which* of two explanations holds, a run
    that ends with both supported has kept a claim the question rules out. Each can be scored
    honestly against its own evidence and still stand beside the other (the same trend fit
    backs "rising steadily" and "a one-off jump" on a step series), so only a check across
    claims catches it.
    """

    preserves_contradicting_evidence = "preserves_contradicting_evidence"
    """Opposing evidence is retained, not discarded in favour of a tidy story."""

    path_adapts_to_goal = "path_adapts_to_goal"
    """The experiments chosen — and the question put to them — reflect what was asked.

    Parameterisation counts, not only tool choice. An experiment aimed at the wrong end of a
    ranking runs the right tool and returns a correct number, so a tool-name assertion passes
    while the run answers the opposite question.
    """

    avoids_redundant_experiments = "avoids_redundant_experiments"
    """No tool is run twice for the same question."""

    respects_budget = "respects_budget"
    """Stays within the resource bounds it was given."""

    calibrated_confidence = "calibrated_confidence"
    """Confidence is proportional to the strength of the evidence."""

    challenges_before_concluding = "challenges_before_concluding"
    """A claim it accepted was first tested by an independent method it chose to run.

    This is the property that separates an adversarial loop from a pipeline that stops at the
    first agreeable result, and it needs its own name because the loop degrades *quietly*
    without it: a critic that never challenges still terminates ``sufficient_evidence`` once
    the candidate tools run out, reaching the same disposition at the same confidence by
    simply running everything. Only the presence of a critique whose falsification tool
    actually executed distinguishes the two.
    """


class AgencyExpectations(DomainModel):
    """Declarative, deterministic expectations for one case.

    Every field is optional; a case only asserts the properties it is designed to probe, so a
    fixture testing overclaiming does not accidentally assert things about tool choice.
    """

    termination_reason_in: list[str] = Field(default_factory=list)
    disposition_in: list[str] = Field(default_factory=list)
    #: Statuses no hypothesis may end in — e.g. ``["supported"]`` for a contradicted claim.
    hypothesis_status_not_in: list[str] = Field(default_factory=list)
    #: At least one hypothesis must end in one of these.
    hypothesis_status_any: list[str] = Field(default_factory=list)
    expect_any_tool: list[str] = Field(default_factory=list)
    forbid_tools: list[str] = Field(default_factory=list)
    #: The entity a ranking experiment must put first. Running the right tool is not the same
    #: as asking it the right question: ranked the wrong way round, ``rank_entities`` answers
    #: "which entity is weakest" with the strongest one — a true sentence about the opposite
    #: question, which no tool-choice assertion can catch.
    expect_ranked_first: str | None = None
    require_contradicting_evidence: bool = False
    #: A supported conclusion must have been challenged by a critique whose suggested
    #: falsification tool actually ran. Only meaningful on cases that converge — a case
    #: expected to end ``insufficient`` has nothing to challenge.
    require_challenge: bool = False
    no_repeated_tools: bool = True
    max_experiments: int | None = None
    max_confidence: float | None = None
    min_confidence: float | None = None
    #: At most this many hypotheses may end ``supported``. Set to 1 on a goal that poses two
    #: rival explanations, where affirming both is a failure to answer the question asked.
    max_supported_claims: int | None = None


class PropertyOutcome(DomainModel):
    """Whether one property held, and why not when it did not."""

    property: AgencyProperty
    passed: bool
    detail: str = ""


class AgencyCaseResult(DomainModel):
    case_id: str
    description: str = ""
    passed: bool
    outcomes: list[PropertyOutcome] = Field(default_factory=list)
    #: Observed facts, so a failure can be diagnosed without re-running the case.
    observed_termination: str | None = None
    observed_disposition: str | None = None
    observed_tools: list[str] = Field(default_factory=list)
    observed_hypothesis_statuses: list[str] = Field(default_factory=list)
    observed_confidence: float = 0.0
    #: Falsification tools that a critique proposed *and* the loop then ran.
    observed_challenge_tools: list[str] = Field(default_factory=list)
    #: Head of the run's ranking, when it ranked at all.
    observed_ranked_first: str | None = None
    #: The policy decision that ended the run, and how it failed. Kept beside the verdict so
    #: a model that cannot produce valid output is not read as one that reasons badly, and a
    #: provider outage is not charged to the model at all.
    observed_policy_failure: PolicyFailure | None = None

    @property
    def failures(self) -> list[PropertyOutcome]:
        return [o for o in self.outcomes if not o.passed]


class AgencyReport(DomainModel):
    """Suite-level result: overall pass rate plus a per-property breakdown."""

    suite_id: str
    total: int = 0
    passed: int = 0
    results: list[AgencyCaseResult] = Field(default_factory=list)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total) if self.total else 0.0

    def property_scores(self) -> dict[str, float]:
        """Pass rate per property, across the cases that asserted it.

        A per-property breakdown is what makes the suite diagnostic rather than a single
        number: "calibrated_confidence 0.4" points somewhere specific.
        """
        totals: dict[str, list[bool]] = {}
        for result in self.results:
            for outcome in result.outcomes:
                totals.setdefault(outcome.property.value, []).append(outcome.passed)
        return {
            name: round(sum(values) / len(values), 4)
            for name, values in sorted(totals.items())
            if values
        }

    def summary(self) -> str:
        line = f"{self.suite_id}: {self.passed}/{self.total} cases passed ({self.pass_rate:.0%})"
        if not self.failed:
            return line
        names = ", ".join(r.case_id for r in self.results if not r.passed)
        return f"{line} — failing: {names}"


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def _tools(investigation: Investigation) -> list[str]:
    return [r.tool_name for r in _results(investigation)]


def _statuses(investigation: Investigation) -> list[str]:
    return [h.status.value for h in investigation.state.hypotheses]


#: Parameter names that identify the column an experiment measured, most specific first.
_METRIC_PARAMS = ("value_column", "column", "metric_column", "x_column")


def _work_units(investigation: Investigation) -> list[tuple[str, str | None]]:
    """
    The unit of redundant work: ``(tool, measured column)``.

    Counting bare tool names was a proxy for "did the same work twice", and it was adequate
    while an investigation could only examine one metric. Once each claim is measured on its own
    column, running ``analyze_time_series_trend`` against two different metrics answers two
    different questions — the proxy misreads legitimate multi-claim work as redundancy.

    Falls back to tool names when ``executed_requests`` is empty, so an investigation persisted
    before that field existed still scores rather than silently passing.
    """
    requests = investigation.state.executed_requests
    if not requests:
        return [(name, None) for name in _tools(investigation)]
    units: list[tuple[str, str | None]] = []
    for request in requests:
        column = next(
            (request.parameters[p] for p in _METRIC_PARAMS if request.parameters.get(p)), None
        )
        units.append((request.tool_name, column))
    return units


def _describe_unit(unit: tuple[str, str | None]) -> str:
    tool, column = unit
    return f"{tool}({column})" if column else tool


def _challenge_tools(investigation: Investigation) -> list[str]:
    """
    Falsification tools a critique proposed that were actually executed.

    A critique the loop never acted on is not a challenge — it is a note. This mirrors the
    ``tested`` condition in ``TerminationPolicy.decide``, so the property measures the same
    thing the loop itself treats as "this claim has been challenged".
    """
    executed = {r.tool_name for r in _results(investigation)}
    return sorted(
        {
            c.suggested_action
            for c in investigation.state.critiques
            if c.suggested_action and c.suggested_action in executed
        }
    )


def _results(investigation: Investigation):
    return investigation.state.completed_experiments + investigation.state.failed_experiments


#: The tool whose ordering ``expect_ranked_first`` reads. Named here rather than inferred from
#: whichever experiment happened to emit an entity-scoped observation: several tools do, and a
#: check that silently fell back to one of them would assert something other than the ranking.
_RANKING_TOOL = "rank_entities"


def _ranked_first(investigation: Investigation) -> str | None:
    """The entity at the head of the run's ranking, or ``None`` if it never ranked.

    ``rank_entities`` emits its ordering as observations, best-first in whichever direction it
    was asked for, so the first entity-scoped observation of the first successful ranking *is*
    the answer the run would report.
    """
    for result in investigation.state.completed_experiments:
        if result.tool_name != _RANKING_TOOL:
            continue
        entity = next((o.entity_ref for o in result.observations if o.entity_ref), None)
        if entity is not None:
            return entity
    return None


def _check(
    prop: AgencyProperty, passed: bool, detail: str = "", outcomes: list[PropertyOutcome] | None = None
) -> None:
    if outcomes is not None:
        outcomes.append(PropertyOutcome(property=prop, passed=passed, detail="" if passed else detail))


# -- checks on the answer alone ------------------------------------------------------------
#
# These read only what a run concluded (disposition, claim statuses, confidence), never how it
# got there. They are separate functions so a run with no investigation behind it (condition C
# of docs/decisions/2026-10-05-scaffold-vs-model.md: one model call, no loop) is held to exactly
# the same checks, not a copy of them that could drift.

#: Properties a bare answer can be scored on. The rest describe the route (tools, critique,
#: budget, termination) and only exist where there was a route.
ANSWER_PROPERTIES: frozenset[AgencyProperty] = frozenset({
    AgencyProperty.reaches_the_right_disposition,
    AgencyProperty.revises_under_contradiction,
    AgencyProperty.calibrated_confidence,
})


def _check_disposition(
    expectations: AgencyExpectations, disposition: str | None, outcomes: list[PropertyOutcome],
) -> None:
    if expectations.disposition_in:
        _check(
            AgencyProperty.reaches_the_right_disposition,
            disposition in expectations.disposition_in,
            f"concluded {disposition!r}, expected one of {expectations.disposition_in}",
            outcomes,
        )


def _check_statuses(
    expectations: AgencyExpectations, statuses: list[str], outcomes: list[PropertyOutcome],
) -> None:
    if expectations.hypothesis_status_not_in:
        offending = [s for s in statuses if s in expectations.hypothesis_status_not_in]
        _check(
            AgencyProperty.revises_under_contradiction,
            not offending,
            f"hypotheses ended {offending}, which the evidence does not support",
            outcomes,
        )

    if expectations.hypothesis_status_any:
        _check(
            AgencyProperty.revises_under_contradiction,
            any(s in expectations.hypothesis_status_any for s in statuses),
            f"hypotheses ended {statuses}, expected any of {expectations.hypothesis_status_any}",
            outcomes,
        )


def _check_max_confidence(
    expectations: AgencyExpectations, confidence: float, outcomes: list[PropertyOutcome],
) -> None:
    if expectations.max_confidence is not None:
        _check(
            AgencyProperty.calibrated_confidence,
            confidence <= expectations.max_confidence,
            f"confidence {confidence:.2f} exceeds {expectations.max_confidence:.2f} for this evidence",
            outcomes,
        )


def _check_rivals(
    expectations: AgencyExpectations, statuses: list[str], outcomes: list[PropertyOutcome],
) -> None:
    if expectations.max_supported_claims is not None:
        supported = statuses.count("supported")
        _check(
            AgencyProperty.revises_under_contradiction,
            supported <= expectations.max_supported_claims,
            f"{supported} claims ended supported; the goal asked which of rival explanations "
            f"holds, so at most {expectations.max_supported_claims} may",
            outcomes,
        )


def _check_min_confidence(
    expectations: AgencyExpectations, confidence: float, outcomes: list[PropertyOutcome],
) -> None:
    if expectations.min_confidence is not None:
        _check(
            AgencyProperty.calibrated_confidence,
            confidence >= expectations.min_confidence,
            f"confidence {confidence:.2f} is below {expectations.min_confidence:.2f} "
            "despite an unambiguous signal",
            outcomes,
        )


def answer_is_scorable(expectations: AgencyExpectations) -> bool:
    """Whether a case asserts anything about the answer, rather than only about the route."""
    return bool(
        expectations.disposition_in or expectations.hypothesis_status_not_in
        or expectations.hypothesis_status_any or expectations.max_confidence is not None
        or expectations.min_confidence is not None or expectations.max_supported_claims is not None
    )


def score_answer(
    expectations: AgencyExpectations, *, disposition: str | None, statuses: list[str], confidence: float,
) -> list[PropertyOutcome]:
    """The answer-level checks of :func:`score_case`, in the same order, for a run with no route."""
    outcomes: list[PropertyOutcome] = []
    _check_disposition(expectations, disposition, outcomes)
    _check_statuses(expectations, statuses, outcomes)
    _check_max_confidence(expectations, confidence, outcomes)
    _check_rivals(expectations, statuses, outcomes)
    _check_min_confidence(expectations, confidence, outcomes)
    return outcomes


def score_case(
    case_id: str, investigation: Investigation, expectations: AgencyExpectations, *, description: str = ""
) -> AgencyCaseResult:
    """Score one finished investigation against a case's expectations."""
    outcomes: list[PropertyOutcome] = []
    state = investigation.state

    termination = state.termination.reason.value if state.termination is not None else None
    conclusion = state.current_conclusion
    disposition = conclusion.disposition.value if conclusion is not None else None
    confidence = float(conclusion.confidence) if conclusion is not None else 0.0
    tools = _tools(investigation)
    statuses = _statuses(investigation)
    challenge_tools = _challenge_tools(investigation)
    ranked_first = _ranked_first(investigation)

    if expectations.termination_reason_in:
        _check(
            AgencyProperty.terminates_for_the_right_reason,
            termination in expectations.termination_reason_in,
            f"terminated {termination!r}, expected one of {expectations.termination_reason_in}",
            outcomes,
        )

    _check_disposition(expectations, disposition, outcomes)
    _check_statuses(expectations, statuses, outcomes)

    if expectations.require_contradicting_evidence:
        directions = {e.direction.value for e in state.evidence}
        _check(
            AgencyProperty.preserves_contradicting_evidence,
            "refutes" in directions,
            f"no refuting evidence retained; directions present: {sorted(directions)}",
            outcomes,
        )

    if expectations.require_challenge:
        _check(
            AgencyProperty.challenges_before_concluding,
            bool(challenge_tools),
            "concluded without testing the claim: "
            + (
                f"{len(state.critiques)} critique(s) raised but none of their falsification "
                f"tools ran (ran {tools})"
                if state.critiques
                else f"no critique was raised at all (ran {tools})"
            ),
            outcomes,
        )

    if expectations.expect_any_tool:
        _check(
            AgencyProperty.path_adapts_to_goal,
            any(t in expectations.expect_any_tool for t in tools),
            f"ran {tools}, expected at least one of {expectations.expect_any_tool}",
            outcomes,
        )

    if expectations.forbid_tools:
        used = [t for t in tools if t in expectations.forbid_tools]
        _check(
            AgencyProperty.path_adapts_to_goal,
            not used,
            f"ran {used}, which do not answer this goal",
            outcomes,
        )

    if expectations.expect_ranked_first is not None:
        _check(
            AgencyProperty.path_adapts_to_goal,
            ranked_first == expectations.expect_ranked_first,
            f"ranking led with {ranked_first!r}, expected {expectations.expect_ranked_first!r}"
            if ranked_first is not None
            else f"no ranking experiment ran (ran {tools}), so the goal's question was never asked",
            outcomes,
        )

    if expectations.no_repeated_tools:
        units = _work_units(investigation)
        repeated = sorted({u for u in units if units.count(u) > 1})
        _check(
            AgencyProperty.avoids_redundant_experiments,
            not repeated,
            f"repeated experiments: {[_describe_unit(u) for u in repeated]}",
            outcomes,
        )

    if expectations.max_experiments is not None:
        _check(
            AgencyProperty.respects_budget,
            len(tools) <= expectations.max_experiments,
            f"ran {len(tools)} experiments, budget allowed {expectations.max_experiments}",
            outcomes,
        )

    _check_max_confidence(expectations, confidence, outcomes)
    _check_rivals(expectations, statuses, outcomes)
    _check_min_confidence(expectations, confidence, outcomes)

    return AgencyCaseResult(
        case_id=case_id,
        description=description,
        passed=all(o.passed for o in outcomes),
        outcomes=outcomes,
        observed_termination=termination,
        observed_disposition=disposition,
        observed_tools=tools,
        observed_hypothesis_statuses=statuses,
        observed_confidence=round(confidence, 6),
        observed_challenge_tools=challenge_tools,
        observed_ranked_first=ranked_first,
        observed_policy_failure=state.termination.policy_failure if state.termination is not None else None,
    )
