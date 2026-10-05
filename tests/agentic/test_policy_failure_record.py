"""
A failed policy decision is recorded as *which* decision failed and *how*.

Before this, every policy failure ended as ``TerminationReason.error`` with nothing else, and a
provider outage reached the loop as an empty reply. So "the model cannot write JSON", "the
model wrote JSON of the wrong shape" and "the laptop timed out" were one record. Comparing
models of different sizes on that record would charge outages to the model and read output
failures as reasoning failures.

One failure is subtler. A selector that names a candidate past the end of the list is treated
as declining to choose, and the run stops as if it had nothing left to try. The stop itself is
deliberately unchanged here, but it must not read as a principled one.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from agentic.agent import FixtureAgentPolicy, ModelAgentPolicy, PolicyTransportError
from agentic.agent.policy import ExperimentChoice
from agentic.domain import PolicyDecisionKind, PolicyFailureKind, TerminationDecision, TerminationReason
from agentic.evaluation.cases import AGENCY_CASES
from agentic.evaluation.runner import run_case

# Concludes `supported`, so every one of the four decisions is reached, the critic included.
_CASE = next(c for c in AGENCY_CASES if c.case_id == "clear_rising_is_concluded")


class _FaultAt(FixtureAgentPolicy):
    """The deterministic policy, except one decision goes through the real model path."""

    def __init__(self, decision: PolicyDecisionKind, respond: Callable[[str, str], str]) -> None:
        super().__init__()
        self._decision = decision
        self._model = ModelAgentPolicy(respond)

    def __getattribute__(self, name: str):
        if name in PolicyDecisionKind.__members__ and name == object.__getattribute__(self, "_decision").value:
            return getattr(object.__getattribute__(self, "_model"), name)
        return super().__getattribute__(name)


def _raises_transport(_system: str, _user: str) -> str:
    raise PolicyTransportError("provider error: connection refused")


@pytest.mark.parametrize("decision", list(PolicyDecisionKind))
@pytest.mark.parametrize(
    ("respond", "kind"),
    [
        (lambda _s, _u: "", PolicyFailureKind.invalid_json),
        (lambda _s, _u: "Sure! Here is the JSON you asked for:", PolicyFailureKind.invalid_json),
        (lambda _s, _u: '{"not_a_field": 1}', PolicyFailureKind.schema),
        (_raises_transport, PolicyFailureKind.transport),
    ],
    ids=["empty", "prose", "schema", "transport"],
)
def test_failure_names_the_decision_and_the_kind(
    decision: PolicyDecisionKind, respond: Callable[[str, str], str], kind: PolicyFailureKind
) -> None:
    result = run_case(_CASE, policy=_FaultAt(decision, respond))

    assert result.observed_termination == TerminationReason.error.value
    failure = result.observed_policy_failure
    assert failure is not None
    assert (failure.decision, failure.kind) == (decision, kind)
    assert failure.detail


def test_a_clean_run_records_no_failure() -> None:
    result = run_case(_CASE, policy=FixtureAgentPolicy())

    assert result.passed
    assert result.observed_policy_failure is None


class _OutOfRangeSelector(FixtureAgentPolicy):
    def select_experiment(self, *, goal_summary: dict, candidates: list[dict]) -> ExperimentChoice:
        return ExperimentChoice(request_index=len(candidates) + 5)


class _DecliningSelector(FixtureAgentPolicy):
    def select_experiment(self, *, goal_summary: dict, candidates: list[dict]) -> ExperimentChoice:
        return ExperimentChoice(request_index=None)


def test_an_out_of_range_choice_is_recorded_without_changing_the_stop() -> None:
    ungrounded = run_case(_CASE, policy=_OutOfRangeSelector())
    declined = run_case(_CASE, policy=_DecliningSelector())

    # Same stop as an explicit decline: the loop's behavior is not what changed.
    assert ungrounded.observed_termination == declined.observed_termination
    assert ungrounded.observed_termination != TerminationReason.error.value
    # The record is what changed.
    assert declined.observed_policy_failure is None
    failure = ungrounded.observed_policy_failure
    assert failure is not None
    assert failure.decision is PolicyDecisionKind.select_experiment
    assert failure.kind is PolicyFailureKind.ungrounded


def test_the_failure_survives_serialization() -> None:
    """It is persisted through ``termination_json``, so it must round-trip with the state."""
    result = run_case(_CASE, policy=_FaultAt(PolicyDecisionKind.critique, lambda _s, _u: "{}{"))
    failure = result.observed_policy_failure
    assert failure is not None
    decision = TerminationDecision(
        should_stop=True, reason=TerminationReason.error, rationale="x", at_iteration=0,
        policy_failure=failure)

    assert TerminationDecision.model_validate_json(decision.model_dump_json()).policy_failure == failure
