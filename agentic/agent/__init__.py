"""
Adaptive investigation loop.

Ten explicit components (`components.py`) driven by an :class:`AgentPolicy`
(`policy.py`) over the deterministic experiment registry, orchestrated by
:class:`InvestigationLoop` (`loop.py`). Execution paths differ by goal,
intermediate results steer selection, hypotheses can be supported/weakened/
rejected/unresolved, budgets and safety limits bound the run, every decision is
persisted, and the loop is resumable.

See ``docs/agent/{investigation-loop,decision-policy,termination-policy}.md``.
Not yet wired into production orchestration.
"""

from __future__ import annotations

from .budget import BudgetTracker, LoopBudget, SafetyLimits
from .clock import Clock, ManualClock, MonotonicClock
from .components import (
    ConclusionSynthesizer,
    Critic,
    EvidenceUpdater,
    ExperimentExecutor,
    ExperimentSelector,
    GoalInterpreter,
    HypothesisGenerator,
    HypothesisUpdater,
    InvestigationPlanner,
    TerminationPolicy,
)
from .diff import (
    ConclusionSnapshot,
    DiffVerdict,
    HypothesisDelta,
    InvestigationDiff,
    diff_investigations,
)
from .fixture_policy import FixtureAgentPolicy
from .loop import InvestigationLoop, run_investigation
from .observer import (
    NULL_OBSERVER,
    AgentObserver,
    ComponentCompleted,
    ExperimentObserved,
    HypothesisTransitioned,
    InvestigationEnded,
    InvestigationStarted,
    IterationEnded,
    IterationStarted,
    LoopComponent,
    ModelCallObserved,
    RecordingObserver,
    TerminationObserved,
)
from .policy import (
    DEFAULT_POLICY_PROMPTS,
    AgentPolicy,
    AgentPolicyError,
    AnalysisIntent,
    CostAwarePolicy,
    ExperimentChoice,
    GoalInterpretation,
    MalformedPolicyResponse,
    ModelAgentPolicy,
    PolicyPrompts,
    PolicyTransportError,
)
from .replay import (
    ReplayNotPossible,
    ReplayResult,
    baseline_manifest,
    replay_investigation,
)
from .store import InMemoryInvestigationStore, InvestigationStore, NullInvestigationStore

__all__ = [
    "InvestigationLoop",
    "run_investigation",
    # policy
    "AgentPolicy",
    "FixtureAgentPolicy",
    "ModelAgentPolicy",
    "PolicyPrompts",
    "DEFAULT_POLICY_PROMPTS",
    "AgentPolicyError",
    "MalformedPolicyResponse",
    "PolicyTransportError",
    "AnalysisIntent",
    "GoalInterpretation",
    "ExperimentChoice",
    "CostAwarePolicy",
    # observability
    "AgentObserver",
    "NULL_OBSERVER",
    "RecordingObserver",
    "LoopComponent",
    "InvestigationStarted",
    "InvestigationEnded",
    "IterationStarted",
    "IterationEnded",
    "ComponentCompleted",
    "ExperimentObserved",
    "HypothesisTransitioned",
    "TerminationObserved",
    "ModelCallObserved",
    # replay / diff
    "replay_investigation",
    "ReplayResult",
    "ReplayNotPossible",
    "baseline_manifest",
    "diff_investigations",
    "InvestigationDiff",
    "DiffVerdict",
    "HypothesisDelta",
    "ConclusionSnapshot",
    "Clock",
    "MonotonicClock",
    "ManualClock",
    # components
    "GoalInterpreter",
    "HypothesisGenerator",
    "InvestigationPlanner",
    "ExperimentSelector",
    "ExperimentExecutor",
    "EvidenceUpdater",
    "HypothesisUpdater",
    "Critic",
    "TerminationPolicy",
    "ConclusionSynthesizer",
    # budgets / store
    "LoopBudget",
    "SafetyLimits",
    "BudgetTracker",
    "InvestigationStore",
    "InMemoryInvestigationStore",
    "NullInvestigationStore",
]
