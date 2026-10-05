"""
Switches that remove parts of the scaffold, for measuring what each part contributes.

This is an evaluation instrument, not a product setting. The defaults *are* the product, and
nothing outside ``agentic/evaluation`` and the benchmark harness should construct anything
else: an ablated loop reaches conclusions the full loop exists to prevent, so it never belongs
on a path a user's run takes.

Three components, each switched off independently (condition B of
``docs/decisions/2026-10-05-scaffold-vs-model.md``):

- ``critic``: the per-iteration challenge of the strongest claim. Off means no critique is
  requested, so no falsification experiment is proposed and no model-reported contradiction
  is recorded. This is *not* independent of termination: typed termination accepts a
  supported claim only once it has been challenged or no intent tool is left to run, so with
  the critic off and termination on, the loop runs every remaining tool before concluding.
- ``typed_termination``: the judgement half of :class:`~agentic.agent.components.TerminationPolicy`
  (nothing left ``proposed``, no live contradiction, every supported claim challenged). Off
  replaces it with :class:`~agentic.agent.components.NaiveTerminationPolicy`, which stops at
  the first claim that clears the confidence bar. Budget, safety and user-stop limits are
  kept: they bound cost, not reasoning, and removing them would measure runaway runs.
- ``mutual_exclusivity``: the deterministic check that records a contradiction when two
  claims the goal posed as alternatives are both standing.
"""

from __future__ import annotations

from agentic.domain.common import DomainModel


class LoopAblations(DomainModel):
    """Which scaffold components are switched on. All on is the product."""

    critic: bool = True
    typed_termination: bool = True
    mutual_exclusivity: bool = True

    @classmethod
    def without(cls, *names: str) -> LoopAblations:
        """The full loop minus the named components, e.g. ``without("critic")``."""
        unknown = set(names) - set(cls.model_fields)
        if unknown:
            raise ValueError(f"unknown ablation(s): {sorted(unknown)}; choose from {sorted(cls.model_fields)}")
        return cls(**dict.fromkeys(names, False))

    @property
    def removed(self) -> list[str]:
        """Names of the components switched off, in field order."""
        return [name for name in type(self).model_fields if not getattr(self, name)]

    @property
    def is_full(self) -> bool:
        return not self.removed

    @property
    def label(self) -> str:
        """Suffix for a scoreboard row, so an ablated row can never pass for the full loop."""
        return "" if self.is_full else " -" + " -".join(self.removed)
