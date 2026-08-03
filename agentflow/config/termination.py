"""Centralized termination policy for agent workflow loops.

All loop-guard limits that stop infinite planner/reflector cycles live here:

  - ``max_planner_cycles``: how many times the Planner may run before the
    workflow forces completion when no progress is being made.
  - ``max_replan_count``: how many Re-Plan attempts the Reflector may request.
  - ``max_stuck_rounds``: consecutive Reflector rounds with no TODO tasks
    before the workflow gives up.
  - ``reflector_planner_cycle_cap``: planner cycle count at which the
    Reflector stops asking the Planner for more tasks.

Values are configurable through environment variables (see ``.env.example``)
and unit-testable in one place instead of being spread across routing
functions with inconsistent literals.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentflow.config.settings import settings


@dataclass(frozen=True)
class TerminationPolicy:
    """Limits that stop the planner/reflector loop when no progress is made."""

    max_planner_cycles: int = 5
    max_replan_count: int = 3
    max_stuck_rounds: int = 3
    reflector_planner_cycle_cap: int = 4

    @classmethod
    def from_settings(cls) -> "TerminationPolicy":
        """Build the policy from the current application settings."""
        return cls(
            max_planner_cycles=settings.max_planner_cycles,
            max_replan_count=settings.max_replan_count,
            max_stuck_rounds=settings.max_stuck_rounds,
            reflector_planner_cycle_cap=settings.reflector_planner_cycle_cap,
        )

    # ------------------------------------------------------------------
    # Planner node guard
    # ------------------------------------------------------------------

    def planner_should_force_complete(
        self,
        cycle_count: int,
        has_todo: bool,
        plan_completed: bool,
    ) -> bool:
        """Force ``goal_completed`` when the planner keeps producing nothing."""
        return not plan_completed and not has_todo and cycle_count >= self.max_planner_cycles

    # ------------------------------------------------------------------
    # Reflector routing guards
    # ------------------------------------------------------------------

    def reflector_should_force_answer_on_replan(self, replan_count: int) -> bool:
        """Stop re-planning after too many attempts."""
        return replan_count >= self.max_replan_count

    def reflector_should_force_answer_when_stuck(
        self,
        stuck_rounds: int,
        planner_cycles: int,
    ) -> bool:
        """Give up when the reflector keeps finding no work and no completion."""
        return (
            stuck_rounds >= self.max_stuck_rounds
            or planner_cycles >= self.reflector_planner_cycle_cap
        )


# Lazily-built shared instance (settings are loaded once at import time).
_policy_cache: TerminationPolicy | None = None


def get_termination_policy() -> TerminationPolicy:
    """Return the shared termination policy."""
    global _policy_cache
    if _policy_cache is None:
        _policy_cache = TerminationPolicy.from_settings()
    return _policy_cache
