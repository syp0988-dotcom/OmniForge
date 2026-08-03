"""Tests for the centralized workflow termination policy."""

from agentflow.config.termination import TerminationPolicy


def _policy(**overrides) -> TerminationPolicy:
    defaults = {
        "max_planner_cycles": 5,
        "max_replan_count": 3,
        "max_stuck_rounds": 3,
        "reflector_planner_cycle_cap": 4,
    }
    defaults.update(overrides)
    return TerminationPolicy(**defaults)


# -- Planner node guard ------------------------------------------------------


def test_planner_force_complete_at_boundary():
    p = _policy()
    # At cycle 5 with no TODO and no completion -> force.
    assert p.planner_should_force_complete(5, has_todo=False, plan_completed=False)
    # One before the cap -> keep going.
    assert not p.planner_should_force_complete(4, has_todo=False, plan_completed=False)
    # TODO work remains -> never force.
    assert not p.planner_should_force_complete(10, has_todo=True, plan_completed=False)
    # Plan already completed -> never force.
    assert not p.planner_should_force_complete(10, has_todo=False, plan_completed=True)


def test_planner_force_complete_custom_limit():
    p = _policy(max_planner_cycles=2)
    assert not p.planner_should_force_complete(1, has_todo=False, plan_completed=False)
    assert p.planner_should_force_complete(2, has_todo=False, plan_completed=False)


# -- Replan guard ------------------------------------------------------------


def test_replan_cap_boundary():
    p = _policy()
    assert not p.reflector_should_force_answer_on_replan(0)
    assert not p.reflector_should_force_answer_on_replan(2)
    assert p.reflector_should_force_answer_on_replan(3)
    assert p.reflector_should_force_answer_on_replan(4)


# -- Stuck / planner-cycle guard ---------------------------------------------


def test_stuck_rounds_cap():
    p = _policy()
    assert not p.reflector_should_force_answer_when_stuck(2, 0)
    assert p.reflector_should_force_answer_when_stuck(3, 0)


def test_planner_cycle_cap():
    p = _policy()
    assert not p.reflector_should_force_answer_when_stuck(1, 3)
    assert p.reflector_should_force_answer_when_stuck(1, 4)


def test_stuck_or_cycle_triggers():
    p = _policy(reflector_planner_cycle_cap=10, max_stuck_rounds=5)
    # Neither exceeded.
    assert not p.reflector_should_force_answer_when_stuck(4, 9)
    # Stuck exceeded.
    assert p.reflector_should_force_answer_when_stuck(5, 1)
    # Cycle cap exceeded.
    assert p.reflector_should_force_answer_when_stuck(1, 10)


def test_from_settings_loads_values():
    from agentflow.config.settings import settings

    p = TerminationPolicy.from_settings()
    assert p.max_planner_cycles == settings.max_planner_cycles
    assert p.max_replan_count == settings.max_replan_count
    assert p.max_stuck_rounds == settings.max_stuck_rounds
    assert p.reflector_planner_cycle_cap == settings.reflector_planner_cycle_cap
