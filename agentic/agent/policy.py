"""
AgentPolicy — the model-backed decision surface.

Deterministic computation never goes through the policy; only *interpretation*,
*hypothesis generation*, *experiment selection*, and *critique* do. Each policy
method is one "model call" with a defined input and a typed, validated output.
Model-backed policies validate the raw response into these types and fail safely
(:class:`MalformedPolicyResponse`) on malformed output.

Two implementations satisfy the protocol: :class:`FixtureAgentPolicy`
(deterministic, for tests) and :class:`ModelAgentPolicy` (wraps a JSON responder).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError

from agentic.domain.common import DomainModel
from agentic.domain.enums import PolicyFailureKind


class AnalysisIntent(str, Enum):
    """What the goal is primarily asking for (drives which experiments are candidates)."""

    trend = "trend"
    comparison = "comparison"
    correlation = "correlation"
    anomaly = "anomaly"
    distribution = "distribution"
    ranking = "ranking"
    association = "association"
    profile = "profile"
    general = "general"


class ChallengeReason(str, Enum):
    """
    Why the critic is being consulted about a claim.

    The critic used to be asked without being told, and its prompt opened "after evidence has
    moved a claim to `supported`" — so on the one recorded occasion it was handed a claim that
    had reached it any other way, it declined, correctly, for want of a mandate. The shape of
    the evidence decides what a useful challenge even *is*, so the reason travels with the ask.
    """

    false_confidence = "false_confidence"
    """The claim reached ``supported``. Guard against stopping at the first agreeable result."""

    conflicting_evidence = "conflicting_evidence"
    """Both supporting and refuting evidence. Name the alternative that tells them apart."""

    undiscriminating_evidence = "undiscriminating_evidence"
    """Every measurement came back neutral. The method is not separating anything; ask for one
    that would, or say the data cannot answer this."""

    unexplained_refutation = "unexplained_refutation"
    """Evidence refutes the claim and nothing supports it, but it has not been dropped yet.
    Ask for an independent method before it is — one refuting measurement is a reason to look
    again, not a verdict.

    Deliberately *not* "propose a different explanation", which is what this shape most wants
    and what a critique cannot do: the mechanism produces a falsification experiment aimed at
    an existing hypothesis, so it can only test the claims the run already holds."""


# -- typed policy I/O --------------------------------------------------------


class GoalInterpretation(DomainModel):
    intent: AnalysisIntent
    metric_hint: str | None = None
    group_hint: str | None = None
    direction: Literal["up", "down"] | None = None
    rationale: str = ""
    #: False when the goal asks about something the dataset does not measure.
    #:
    #: Deciding that "customer loyalty" is not answerable from delivery times is a judgement
    #: about meaning, so it belongs to the policy — but the policy only *reports* it here.
    #: What follows is computed: the loop stops, proposes nothing, runs nothing, and says so.
    #:
    #: This field exists because the alternative is what actually happened. With no typed way
    #: to decline, a model asked an unanswerable question picked the nearest available metric,
    #: and the loop dutifully ranked regions by average delivery days under a claim about
    #: loyalty. It then reported ``insufficient_evidence`` — not because the premise was
    #: broken, but because the proxy's signal was weak. The right answer for the wrong reason
    #: is indistinguishable from luck, and it stops being right the moment the proxy is strong.
    answerable: bool = True
    #: What the goal asks for that the data does not contain. Prose, shown to the user.
    unsupported_concept: str | None = None


class HypothesisProposal(DomainModel):
    statement: str = Field(..., min_length=1)
    metric: str | None = None
    direction: Literal["up", "down", "none"] = "none"
    rationale: str = ""


class HypothesisProposals(DomainModel):
    hypotheses: list[HypothesisProposal] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)


class ExperimentChoice(DomainModel):
    """Selection among candidate experiments; ``request_index`` indexes candidates."""

    request_index: int | None = None
    rationale: str = ""


class CritiqueProposal(DomainModel):
    should_challenge: bool = False
    target_hypothesis_id: str | None = None
    falsification_tool: str | None = None
    message: str = ""
    rationale: str = ""
    #: Another *currently supported* claim that cannot be true at the same time as
    #: ``target_hypothesis_id``. Judging that two natural-language statements are mutually
    #: exclusive is interpretation, so it belongs to the policy — but the model only
    #: *reports* the conflict here. What that does to either claim's status and confidence
    #: is computed by :class:`~agentic.agent.components.Critic`, never by the model.
    contradicts_hypothesis_id: str | None = None


class AnswerNarration(DomainModel):
    """
    The run's findings, written as prose.

    Interpretation, so it belongs to the policy — a list of joined hypothesis statements is
    the truth but not an answer. The model supplies wording only: every figure it states is
    checked against recorded values by :func:`agentic.agent.narrative.verify_narrative`
    before it is kept, and prose that cites a number the run never produced is discarded
    whole. That check, not the prompt, is what upholds the rule that no number in a trace
    originates from a language model.
    """

    answer: str = Field(default="")


# -- errors ------------------------------------------------------------------


class AgentPolicyError(RuntimeError):
    """Base for policy failures (the loop terminates safely on these).

    ``kind`` is recorded on the termination, so the persisted run says *how* the decision
    failed and not merely that it did.
    """

    kind: PolicyFailureKind = PolicyFailureKind.schema


class MalformedPolicyResponse(AgentPolicyError):
    """The model returned output that failed typed validation."""

    def __init__(self, message: str, *, kind: PolicyFailureKind = PolicyFailureKind.schema) -> None:
        super().__init__(message)
        self.kind = kind


class PolicyTransportError(AgentPolicyError):
    """The provider never produced a reply. A failure of the machine, not of the model."""

    kind = PolicyFailureKind.transport


# -- protocol ----------------------------------------------------------------


@runtime_checkable
class AgentPolicy(Protocol):
    """The set of model-backed decisions the loop delegates."""

    def interpret_goal(self, goal_text: str, *, capability_summary: dict) -> GoalInterpretation: ...

    def generate_hypotheses(
        self, interpretation: GoalInterpretation, *, metric_names: list[str],
        dimension_names: list[str], goal_text: str = "",
    ) -> HypothesisProposals: ...

    def select_experiment(self, *, goal_summary: dict, candidates: list[dict]) -> ExperimentChoice: ...

    def critique(
        self,
        *,
        strongest_claim: dict | None,
        available_tools: list[str],
        supported_claims: list[dict] | None = None,
    ) -> CritiqueProposal: ...


@runtime_checkable
class CostAwarePolicy(Protocol):
    """
    Optional policy extension: report and reset the cost accrued since the last drain.

    Policies that know their token usage (model-backed ones) implement this so the
    loop's ``max_cost_usd`` budget is enforceable. Policies that don't are unaffected
    — :func:`drain_policy_cost` reports zero for them.
    """

    def drain_cost_usd(self) -> float: ...


@runtime_checkable
class NarratingPolicy(Protocol):
    """
    Optional policy extension: write the run's findings as prose.

    Separate from :class:`AgentPolicy` for the same reason as :class:`CostAwarePolicy` — the
    four-method contract is deliberate, and a policy that cannot write is not a broken
    policy. The loop asks, and settles for its deterministic statement when the answer is
    absent, unusable, or cites a figure the run never recorded.
    """

    def write_answer(self, *, question: str, findings: dict) -> AnswerNarration: ...


def narrate_answer(policy: object, *, question: str, findings: dict) -> str | None:
    """
    Prose from ``policy``, or ``None`` when it cannot or will not write one.

    Duck-typed like :func:`drain_policy_cost` so :class:`AgentPolicy` stays a four-method
    contract and existing implementations need no change. Every failure mode collapses to
    ``None`` — the caller always has a true statement to fall back to, so there is nothing
    here worth raising over.
    """
    write = getattr(policy, "write_answer", None)
    if not callable(write):
        return None
    try:
        narration = write(question=question, findings=findings)
    except Exception:  # noqa: BLE001 - a policy that fails to write must not fail the run
        return None
    answer = getattr(narration, "answer", None)
    return answer.strip() if isinstance(answer, str) and answer.strip() else None


def drain_policy_cost(policy: object) -> float:
    """
    Cost accrued by ``policy`` since the last drain, or ``0.0`` when it doesn't
    track cost. Kept duck-typed so :class:`AgentPolicy` stays a four-method contract
    and existing implementations need no change.
    """
    drain = getattr(policy, "drain_cost_usd", None)
    if not callable(drain):
        return 0.0
    try:
        return max(0.0, float(drain()))
    except (TypeError, ValueError):
        return 0.0


# -- model-backed implementation --------------------------------------------

Responder = Callable[[str, str], str]
"""A function (system_prompt, user_prompt) -> raw JSON string.

A responder whose provider never answered raises :class:`PolicyTransportError` rather than
returning ``""``: an empty string is something a model can genuinely emit, and conflating the
two would charge an outage to the model."""


@dataclass(frozen=True)
class PolicyPrompts:
    """
    The system prompt for each of the four model-backed decisions.

    Defaults are terse but sufficient: they keep this package standalone, so
    :class:`ModelAgentPolicy` works with no external prompt files and the loop stays
    runnable outside this repository. Richer, versioned bodies live under
    ``backend/agents/prompts/`` and are *injected* — this module never loads them, which
    is what keeps ``agentic/`` free of a dependency on the backend.
    """

    interpret_goal: str = (
        "Interpret the analytical goal. The capability summary lists every column that "
        "exists; a concept the goal names that is not measured by any of them cannot be "
        "answered here, however close another column sounds. In that case set "
        "answerable=false and put the missing concept in unsupported_concept — do not "
        "substitute the nearest metric, because a confident answer about something the user "
        "did not ask is worse than no answer. Reply as GoalInterpretation JSON."
    )
    generate_hypotheses: str = (
        "Propose falsifiable hypotheses and open questions. Reply as HypothesisProposals JSON."
    )
    select_experiment: str = "Choose the most informative next experiment. Reply as ExperimentChoice JSON."
    critique: str = (
        "Challenge the strongest current claim; suggest a falsification tool. If two claims "
        "in supported_claims cannot both be true, report the conflict in "
        "contradicts_hypothesis_id. Reply as CritiqueProposal JSON."
    )
    write_answer: str = (
        "Write the finding as a short, direct answer to the question, in plain prose. "
        "State only figures present in the findings you are given, copied exactly, and name "
        "what each one counts in the same clause ('7 experiments', '95% confidence') — a "
        "bare number with nothing saying what it is will be rejected. The findings contain "
        "no measured values, so never attach a figure to a column or metric: you may say a "
        "claim about revenue was supported, never what revenue was or by how much it moved. "
        "Do not state ratios either ('twice as fast', 'roughly doubled', 'an order of "
        "magnitude') — those are figures too, and the run did not compute them. Comparing "
        "what the run recorded ('more refuting than supporting evidence') is fine. "
        "If you are unsure of a figure, describe the result without it. Say plainly when "
        "nothing was established. Reply as AnswerNarration JSON."
    )


DEFAULT_POLICY_PROMPTS = PolicyPrompts()
"""The standalone defaults, used whenever no prompts are injected."""


class ModelAgentPolicy:
    """
    Policy backed by a JSON responder (e.g. an LLM). Every method validates the
    raw response into a typed model and raises :class:`MalformedPolicyResponse`
    on malformed output, so the loop can fail safely.

    ``prompts`` overrides the system prompt of each decision; omitting it selects
    :data:`DEFAULT_POLICY_PROMPTS`.
    """

    def __init__(self, respond: Responder, *, prompts: PolicyPrompts | None = None) -> None:
        self._respond = respond
        self._prompts = prompts or DEFAULT_POLICY_PROMPTS

    def _call(self, system: str, user: str, model: type[BaseModel]):
        raw = self._respond(system, user)
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise MalformedPolicyResponse(
                f"policy response is not valid JSON: {exc}", kind=PolicyFailureKind.invalid_json) from exc
        try:
            return model.model_validate(data)
        except ValidationError as exc:
            raise MalformedPolicyResponse(
                f"policy response failed {model.__name__} validation: {exc}", kind=PolicyFailureKind.schema) from exc

    def interpret_goal(self, goal_text: str, *, capability_summary: dict) -> GoalInterpretation:
        return self._call(
            self._prompts.interpret_goal,
            json.dumps({"goal": goal_text, "capabilities": capability_summary}),
            GoalInterpretation,
        )

    def generate_hypotheses(
        self, interpretation: GoalInterpretation, *, metric_names: list[str],
        dimension_names: list[str], goal_text: str = "",
    ) -> HypothesisProposals:
        # The goal text goes through as well as the interpretation. The interpretation is a
        # classification — intent, one metric hint, a direction — and it cannot carry an
        # alternative explanation the goal offered ("...or is rising volume the cause?").
        # Without the original wording the generator cannot propose the competing claim, so
        # the run measures the outcome alone and ends inconclusive by construction.
        return self._call(
            self._prompts.generate_hypotheses,
            json.dumps({"goal": goal_text,
                        "interpretation": interpretation.model_dump(mode="json"),
                        "metrics": metric_names, "dimensions": dimension_names}),
            HypothesisProposals,
        )

    def select_experiment(self, *, goal_summary: dict, candidates: list[dict]) -> ExperimentChoice:
        return self._call(
            self._prompts.select_experiment,
            json.dumps({"goal": goal_summary, "candidates": candidates}),
            ExperimentChoice,
        )

    def critique(
        self,
        *,
        strongest_claim: dict | None,
        available_tools: list[str],
        supported_claims: list[dict] | None = None,
    ) -> CritiqueProposal:
        return self._call(
            self._prompts.critique,
            json.dumps({
                "claim": strongest_claim,
                "tools": available_tools,
                "supported_claims": supported_claims or [],
            }),
            CritiqueProposal,
        )

    def write_answer(self, *, question: str, findings: dict) -> AnswerNarration:
        return self._call(
            self._prompts.write_answer,
            json.dumps({"question": question, "findings": findings}),
            AnswerNarration,
        )
