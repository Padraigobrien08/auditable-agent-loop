"""
Build the adaptive loop's :class:`~agentic.agent.policy.AgentPolicy` from backend settings.

The agentic investigation loop delegates only its *model-backed* decisions
(goal interpretation, hypothesis generation, experiment selection, critique) to an
``AgentPolicy``. Deterministic computation never goes through the policy.

This module bridges the backend's configured :class:`ChatCompletionProvider` to the
loop's ``Responder`` contract (``(system_prompt, user_prompt) -> raw JSON string``),
returning a :class:`ModelAgentPolicy` when an LLM is configured and falling back to the
deterministic :class:`FixtureAgentPolicy` otherwise. The loop stays offline-safe: with no
provider it still runs to a typed termination, it just makes deterministic decisions.
"""

from __future__ import annotations

from uuid import UUID

import structlog
from sqlalchemy.orm import Session

from agentic.agent.fixture_policy import FixtureAgentPolicy
from agentic.agent.policy import (
    DEFAULT_POLICY_PROMPTS,
    AgentPolicy,
    ModelAgentPolicy,
    PolicyPrompts,
    PolicyTransportError,
    Responder,
)
from backend.agents.prompt_registry import AGENTIC_POLICY_ROLES, load_registered_prompt
from backend.config.settings import Settings, get_settings
from backend.llm.exceptions import ChatCompletionProviderError, LLMProviderConfigurationError
from backend.llm.factory import get_chat_completion_provider
from backend.llm.pricing import ModelPrice, estimate_cost_usd, parse_model_prices
from backend.llm.protocol import ChatCompletionProvider
from backend.llm.types import ChatCompletionRequest
from backend.services.recorded_chat_completion_service import RecordedChatCompletionService

#: Audit identity for policy completions. The loop's decisions are one logical phase from the
#: run's point of view, so they bucket together in ``GET /v1/runs/{id}/llm-usage`` rather than
#: splitting across the ten components (whose names would blow up the phase cardinality).
AGENTIC_AGENT_ROLE = "agentic_policy"
AGENTIC_PHASE = "agentic"
AGENTIC_PROMPT_ID = "agentic.policy"

log = structlog.get_logger(__name__)


def _agentic_model(settings: Settings) -> str:
    """Model id for policy calls; reuse the shared agent completion model."""
    return settings.agent_completion_model or "gpt-5.4-mini"


class CostTrackingResponder:
    """
    Adapt a :class:`ChatCompletionProvider` to the loop's ``Responder`` contract while
    accumulating the USD cost of each completion.

    Cost is held here (rather than inside the policy) because usage is only visible at
    the provider boundary. The loop drains it after every policy decision via
    :class:`~agentic.agent.policy.CostAwarePolicy`, so ``LoopBudget.max_cost_usd`` binds
    on real token usage. Unpriced models accrue ``0.0`` — see :mod:`backend.llm.pricing`.
    """

    def __init__(
        self,
        provider: ChatCompletionProvider,
        *,
        model: str,
        prices: dict[str, ModelPrice] | None = None,
        recorder: RecordedChatCompletionService | None = None,
        analysis_run_id: UUID | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._prices = prices or {}
        self._pending_cost_usd = 0.0
        # When supplied, every completion is also persisted as a ``ModelCall`` row. Without
        # it the loop's spend is tracked only in memory: the budget still binds during the
        # run, but nothing durable records what was spent — which leaves
        # ``GET /v1/runs/{id}/llm-usage`` empty for agentic runs and makes the per-account
        # USD ceiling in ``backend.services.spend_guard`` read $0.00 forever.
        self._recorder = recorder
        self._analysis_run_id = analysis_run_id

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        request = ChatCompletionRequest(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        try:
            if self._recorder is not None:
                _row, result = self._recorder.complete_and_persist(
                    request,
                    analysis_run_id=self._analysis_run_id,
                    request_metadata={"role": AGENTIC_AGENT_ROLE, "phase": AGENTIC_PHASE},
                    prompt_id=AGENTIC_PROMPT_ID,
                    prompt_version=AGENTIC_PROMPT_VERSION,
                )
            else:
                result = self._provider.complete(request)
        except ChatCompletionProviderError as exc:
            # Boundary: a provider failure becomes a typed transport failure, which the loop
            # treats as a safe termination rather than an unhandled crash. It used to return
            # ``""`` and surface as malformed output — indistinguishable from a model that
            # replied with nothing, which charged every outage to the model. The recorder has
            # already left an ``error`` row, so the attempt is still audited.
            log.warning("agentic.policy.provider_error", error=str(exc))
            raise PolicyTransportError(f"provider error: {exc}") from exc
        self._pending_cost_usd += estimate_cost_usd(
            self._prices,
            model=result.model or self._model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )
        return result.assistant_text or ""

    def drain_cost_usd(self) -> float:
        """Cost accrued since the last drain (the loop charges it to the run budget)."""
        pending, self._pending_cost_usd = self._pending_cost_usd, 0.0
        return pending


#: Prompt file version loaded for every agentic policy role. Bump when a new prompt
#: version ships; the agency benchmark records this so results stay comparable.
#:
#: 1.0.1 — spell out which fields accept ``null``. Under 1.0.0 the critic answered a
#: decline with ``"message": null``, which fails the non-nullable ``str`` and terminated
#: the whole investigation with ``reason=error``. Every converging case was lost that way.
#:
#: 1.0.2 — teach the goal interpreter how to read causal and temporal questions. Under 1.0.1
#: "is quality degrading, or is rising volume the cause?" was classified ``comparison``,
#: producing the hypothesis "avg_delivery_days differs from the other available metrics" — a
#: category error. Intent hard-gates the candidate tools, so that one call left the run with a
#: single experiment and a premature ``insufficient_evidence``. Only the goal-interpreter file
#: changed; the other three are copies of 1.0.1, since one constant versions all four.
#: 1.0.3 — teach the critic to report a contradiction between two supported claims. Nothing
#: else compares claims to each other: the hypothesis updater scores each against its own
#: evidence, so a claim and its negation both reached `supported` at 0.95 in a real recording
#: and the run still reported `sufficient_evidence`. The critic is the only component that
#: sees the supported set together. Only the critic file changed; the other three are copies
#: of 1.0.2, since one constant versions all four.
AGENTIC_PROMPT_VERSION = "1.0.6"


class CostAwareModelPolicy(ModelAgentPolicy):
    """:class:`ModelAgentPolicy` that reports spend, satisfying ``CostAwarePolicy``."""

    def __init__(
        self,
        responder: CostTrackingResponder,
        *,
        prompts: PolicyPrompts | None = None,
        prompt_identity: dict[str, str] | None = None,
    ) -> None:
        super().__init__(responder, prompts=prompts)
        self._cost_source = responder
        # Read by the agency benchmark so a scoreboard row names the exact prompts that
        # produced it. Empty when the policy fell back to the domain defaults.
        self.prompt_identity: dict[str, str] = dict(prompt_identity or {})

    def drain_cost_usd(self) -> float:
        return self._cost_source.drain_cost_usd()


def _load_policy_prompts(
    version: str = AGENTIC_PROMPT_VERSION,
) -> tuple[PolicyPrompts, dict[str, str]]:
    """
    Load the registered policy prompts, plus a ``prompt_id -> version`` identity map.

    Never raises. A missing or malformed prompt file degrades to
    :data:`~agentic.agent.policy.DEFAULT_POLICY_PROMPTS` rather than making the loop
    unrunnable — the same never-raise-on-misconfiguration contract
    :func:`build_agent_policy` already honours for an absent LLM provider.
    """
    try:
        loaded = [load_registered_prompt(role, version) for role in AGENTIC_POLICY_ROLES]
    except (FileNotFoundError, ValueError, OSError) as exc:
        log.warning("agentic.policy.prompts_unavailable", version=version, error=str(exc))
        return DEFAULT_POLICY_PROMPTS, {}
    # Positional by AGENTIC_POLICY_ROLES order, which is documented to match the four
    # AgentPolicy methods; named explicitly here so a reordering cannot silently swap them.
    interpreter, generator, selector, critic, writer = loaded
    prompts = PolicyPrompts(
        interpret_goal=interpreter.template.system_body,
        generate_hypotheses=generator.template.system_body,
        select_experiment=selector.template.system_body,
        critique=critic.template.system_body,
        write_answer=writer.template.system_body,
    )
    identity = {p.prompt_id: p.prompt_version for p in loaded}
    return prompts, identity


def build_provider_responder(provider: ChatCompletionProvider, *, model: str) -> Responder:
    """Adapt a :class:`ChatCompletionProvider` to the loop's ``Responder`` contract.

    Retained as the plain (cost-unaware) adapter; :class:`CostTrackingResponder` is the
    one wired into :func:`build_agent_policy`.
    """
    return CostTrackingResponder(provider, model=model)


def build_agent_policy(
    settings: Settings | None = None,
    *,
    session: Session | None = None,
    analysis_run_id: UUID | None = None,
) -> AgentPolicy:
    """Return the model-backed policy when an LLM is configured, else the fixture policy.

    Never raises on a missing/misconfigured provider: the loop must always be able to run,
    deterministically, without an LLM.

    Pass ``session`` and ``analysis_run_id`` to persist every policy completion as a
    ``ModelCall`` linked to the run. Callers that omit them still get a working policy with an
    in-memory cost budget, but nothing durable records the spend — see
    :class:`CostTrackingResponder`.
    """
    s = settings if settings is not None else get_settings()
    try:
        provider = get_chat_completion_provider(s)
    except LLMProviderConfigurationError:
        log.info("agentic.policy.fixture", reason="llm_provider_unavailable")
        return FixtureAgentPolicy()
    model = _agentic_model(s)
    prices = parse_model_prices(s.llm_model_prices)
    prompts, identity = _load_policy_prompts()
    recorder = (
        RecordedChatCompletionService(session, provider) if session is not None else None
    )
    log.info(
        "agentic.policy.model_backed",
        model=model,
        priced=model in prices,
        recorded=recorder is not None,
        prompts=sorted(identity),
        prompt_version=AGENTIC_PROMPT_VERSION if identity else "defaults",
    )
    return CostAwareModelPolicy(
        CostTrackingResponder(
            provider,
            model=model,
            prices=prices,
            recorder=recorder,
            analysis_run_id=analysis_run_id,
        ),
        prompts=prompts,
        prompt_identity=identity,
    )
