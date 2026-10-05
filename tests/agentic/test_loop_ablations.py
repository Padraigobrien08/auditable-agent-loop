"""
Scaffold components can be switched off one at a time, and the defaults are the product.

Condition B of docs/decisions/2026-10-05-scaffold-vs-model.md: to measure what the critic,
typed termination and the mutual-exclusivity check each contribute, each has to come out
cleanly, without dragging anything else with it. The first test here is the one that guards
everything else in the repo: with no ablation requested, the loop must be exactly the loop
every published number was measured on.
"""

from __future__ import annotations

import pytest

from agentic.agent import FixtureAgentPolicy, LoopAblations
from agentic.agent.budget import BudgetTracker, LoopBudget, SafetyLimits
from agentic.agent.components import NaiveTerminationPolicy, TerminationPolicy, enforce_mutual_exclusivity
from agentic.agent.ids import DeterministicIds
from agentic.domain import Hypothesis, InvestigationGoal, InvestigationState, TerminationReason
from agentic.domain.enums import CritiqueType, HypothesisStatus, ProvenanceSource
from agentic.domain.provenance import Provenance
from agentic.evaluation.cases import AGENCY_CASES
from agentic.evaluation.runner import run_agency_suite, run_case

_PROV = Provenance(source=ProvenanceSource.agent_llm, agent_id="test")
_CONCLUDING = next(c for c in AGENCY_CASES if c.case_id == "clear_rising_is_concluded")


# -- the defaults are the product -----------------------------------------------


def test_explicit_defaults_reproduce_the_full_suite_exactly() -> None:
    baseline = run_agency_suite(policy=FixtureAgentPolicy())
    explicit = run_agency_suite(policy=FixtureAgentPolicy(), ablations=LoopAblations())

    assert explicit.model_dump() == baseline.model_dump()


def test_the_default_is_the_full_loop() -> None:
    assert LoopAblations().is_full
    assert LoopAblations().label == ""


def test_without_names_what_was_removed() -> None:
    ablated = LoopAblations.without("critic", "mutual_exclusivity")

    assert ablated.removed == ["critic", "mutual_exclusivity"]
    assert ablated.typed_termination
    assert ablated.label == " -critic -mutual_exclusivity"


def test_an_unknown_component_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown ablation"):
        LoopAblations.without("criticc")


# -- critic ------------------------------------------------------------------------


class _CountingPolicy(FixtureAgentPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.critiques = 0

    def critique(self, **kwargs):  # noqa: ANN003 - pass-through test double
        self.critiques += 1
        return super().critique(**kwargs)


def test_critic_off_asks_for_no_critique() -> None:
    on, off = _CountingPolicy(), _CountingPolicy()
    run_case(_CONCLUDING, policy=on)
    result = run_case(_CONCLUDING, policy=off, ablations=LoopAblations.without("critic"))

    assert on.critiques > 0, "the case must reach the critic, or this test proves nothing"
    assert off.critiques == 0
    assert result.observed_challenge_tools == []


def test_critic_off_under_typed_termination_runs_the_remaining_tools() -> None:
    """
    The two are not independent, and this pins how. Typed termination accepts a supported
    claim only once it has been challenged or no intent tool is left, so without a critic the
    loop does not stop sooner: it runs everything left before concluding.
    """
    full = run_case(_CONCLUDING, policy=FixtureAgentPolicy())
    no_critic = run_case(_CONCLUDING, policy=FixtureAgentPolicy(), ablations=LoopAblations.without("critic"))

    assert len(no_critic.observed_tools) > len(full.observed_tools)
    assert no_critic.observed_termination == TerminationReason.sufficient_evidence.value


# -- mutual exclusivity -------------------------------------------------------------


@pytest.mark.parametrize("enabled", [True, False])
def test_mutual_exclusivity_check_runs_only_when_enabled(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    import agentic.agent.loop as loop_module

    calls: list[int] = []

    def spy(state, idgen):  # noqa: ANN001 - pass-through test double
        calls.append(1)
        return enforce_mutual_exclusivity(state, idgen)

    monkeypatch.setattr(loop_module, "enforce_mutual_exclusivity", spy)
    ablations = LoopAblations() if enabled else LoopAblations.without("mutual_exclusivity")
    run_case(_CONCLUDING, policy=FixtureAgentPolicy(), ablations=ablations)

    assert bool(calls) is enabled


# -- typed termination ----------------------------------------------------------------


def _claim(cid: str, status: HypothesisStatus, confidence: float = 0.9) -> Hypothesis:
    claim = Hypothesis(id=cid, statement=cid, provenance=_PROV)
    claim.set_confidence(confidence)
    if status is not HypothesisStatus.proposed:
        claim.set_status(HypothesisStatus.active)
        claim.set_status(status)
    return claim


def _state(*claims: Hypothesis) -> InvestigationState:
    state = InvestigationState(objective=InvestigationGoal(objective="goal"))
    for claim in claims:
        state.add_hypothesis(claim)
    return state


def _tracker(**budget: float) -> BudgetTracker:
    return BudgetTracker(budget=LoopBudget(**budget), safety=SafetyLimits())  # type: ignore[arg-type]


def _decide(policy: TerminationPolicy, state: InvestigationState, tracker: BudgetTracker | None = None):
    return policy.decide(state, tracker or _tracker(), 1, executed_tools={"trend"},
                         intent_tools=["trend", "changepoint"], user_stop=False)


def test_naive_termination_stops_with_a_rival_claim_still_untested() -> None:
    state = _state(_claim("a", HypothesisStatus.supported), _claim("b", HypothesisStatus.proposed))

    assert _decide(TerminationPolicy(), state) == (False, None)
    assert _decide(NaiveTerminationPolicy(), state) == (True, TerminationReason.sufficient_evidence)


def test_naive_termination_concludes_over_an_open_contradiction() -> None:
    first, second = _claim("a", HypothesisStatus.supported), _claim("b", HypothesisStatus.supported)
    first.mutually_exclusive_with, second.mutually_exclusive_with = ["b"], ["a"]
    state = _state(first, second)
    recorded = enforce_mutual_exclusivity(state, DeterministicIds("inv"))
    assert [c.critique_type for c in recorded] == [CritiqueType.contradiction]
    # Weakened by the conflict; restore one above the bar so only the contradiction differs.
    first.set_status(HypothesisStatus.supported)
    first.set_confidence(0.9)

    typed = _decide(TerminationPolicy(), state)
    naive = _decide(NaiveTerminationPolicy(), state)

    assert typed[1] is not TerminationReason.sufficient_evidence
    assert naive == (True, TerminationReason.sufficient_evidence)


def test_naive_termination_keeps_the_hard_stops() -> None:
    """Budget and safety bound cost, not reasoning, so they are never ablated."""
    state = _state(_claim("a", HypothesisStatus.supported))
    exhausted = _tracker(max_experiments=1)
    exhausted.record_experiment("trend")

    assert _decide(NaiveTerminationPolicy(), state, exhausted) == (True, TerminationReason.budget_exhausted)


def test_naive_finalize_ignores_untested_claims() -> None:
    state = _state(_claim("a", HypothesisStatus.supported), _claim("b", HypothesisStatus.proposed))

    assert TerminationPolicy().finalize_no_candidates(state, ran_any=True) is TerminationReason.insufficient_evidence
    assert NaiveTerminationPolicy().finalize_no_candidates(state, ran_any=True) is TerminationReason.sufficient_evidence


def test_typed_termination_off_stops_at_the_first_supported_claim() -> None:
    full = run_case(_CONCLUDING, policy=FixtureAgentPolicy())
    naive = run_case(_CONCLUDING, policy=FixtureAgentPolicy(), ablations=LoopAblations.without("typed_termination"))

    assert len(naive.observed_tools) < len(full.observed_tools)
    assert naive.observed_termination == TerminationReason.sufficient_evidence.value
