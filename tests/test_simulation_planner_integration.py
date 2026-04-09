"""
Integration-style simulation tests for planner backends and fallback behavior.
"""

import os
import sys
from unittest.mock import patch
import numpy as np


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents import RobotAgent
from src.map_generator import WarehouseMap
from src.navigation.planners import PLANNER_STATUS_FAILED_NO_SOLUTION, PlannerResult
from src.simulation.game_engine import GameEngine


def _empty_grid(width: int, height: int):
    return [['.' for _ in range(width)] for _ in range(height)]


def _build_engine_with_simple_layout(planner_mode: str = "astar", disable_llm: bool = False) -> GameEngine:
    prev_mode = os.environ.get("PATH_PLANNER_MODE")
    prev_disable_llm = os.environ.get("DISABLE_LLM_NEGOTIATION")
    os.environ["PATH_PLANNER_MODE"] = planner_mode
    os.environ["DISABLE_LLM_NEGOTIATION"] = "true" if disable_llm else "false"
    try:
        engine = GameEngine(width=6, height=6, num_agents=2)
    finally:
        if prev_mode is None:
            os.environ.pop("PATH_PLANNER_MODE", None)
        else:
            os.environ["PATH_PLANNER_MODE"] = prev_mode
        if prev_disable_llm is None:
            os.environ.pop("DISABLE_LLM_NEGOTIATION", None)
        else:
            os.environ["DISABLE_LLM_NEGOTIATION"] = prev_disable_llm

    warehouse = WarehouseMap(width=6, height=6)
    warehouse.grid = np.array(_empty_grid(6, 6), dtype=str)
    warehouse.agents = {1: (1, 1), 2: (4, 1)}
    warehouse.boxes = {1: (1, 2), 2: (4, 2)}
    warehouse.targets = {1: (1, 4), 2: (4, 4)}
    warehouse.agent_goals = {1: 1, 2: 2}
    engine.warehouse_map = warehouse
    engine.agents = {
        1: RobotAgent(agent_id=1, initial_position=(1, 1)),
        2: RobotAgent(agent_id=2, initial_position=(4, 1)),
    }
    engine.simulation_mode = "turn_based"
    engine.log_enabled = False
    engine.logger = None
    engine.silent_mode = True
    return engine


def test_initialize_simulation_assigns_initial_paths_with_astar_backend():
    engine = _build_engine_with_simple_layout("astar")
    engine.initialize_simulation()

    assert engine.multi_agent_planner.get_backend_name() == "astar"
    assert engine.agents[1].planned_path
    assert engine.agents[2].planned_path
    assert engine.agents[1].planned_path[0] == engine.agents[1].position
    assert engine.agents[2].planned_path[0] == engine.agents[2].position


def test_initialize_simulation_assigns_initial_paths_with_lns2_backend():
    engine = _build_engine_with_simple_layout("LNS2")
    engine.initialize_simulation()

    assert engine.multi_agent_planner.get_backend_name() == "LNS2"
    assert engine.agents[1].planned_path
    assert engine.agents[2].planned_path
    assert engine.agents[1].planned_path[-1] == engine.agents[1].target_position
    assert engine.agents[2].planned_path[-1] == engine.agents[2].target_position


def test_initialize_simulation_accumulates_initial_min_required_steps():
    engine = _build_engine_with_simple_layout("astar")
    engine.initialize_simulation()

    # Each agent starts exactly one step from its assigned box.
    assert engine.total_min_required_steps == 2
    assert engine.total_actual_steps == 0


def test_path_efficiency_counts_failed_attempts_in_denominator():
    engine = _build_engine_with_simple_layout("astar")
    engine.initialize_simulation()

    # Use deterministic counters for this unit test.
    engine.total_min_required_steps = 1
    engine.total_actual_steps = 0

    agent = engine.agents[1]
    map_state = engine.warehouse_map.get_state_dict()

    # One successful move attempt.
    success, _ = agent.move_to((1, 2), map_state)
    assert success is True

    # One failed move attempt (non-adjacent).
    success, _ = agent.move_to((1, 4), map_state)
    assert success is False

    metrics = engine.calculate_performance_metrics()
    assert engine.total_actual_steps == 2
    assert metrics['path_efficiency'] == 50.0


def test_path_efficiency_is_not_capped_at_100_percent():
    engine = _build_engine_with_simple_layout("astar")
    engine.initialize_simulation()

    engine.total_min_required_steps = 6
    engine.total_actual_steps = 3

    metrics = engine.calculate_performance_metrics()
    assert metrics['path_efficiency'] == 200.0


def test_stagnation_uses_llm_only_when_planner_cannot_find_path():
    engine = _build_engine_with_simple_layout("LNS2")
    engine.initialize_simulation()

    # Force stagnation detection on both agents.
    engine.stagnation_turns = 1
    engine.agent_failed_move_history = {
        1: [{'turn': 0, 'attempted_move': (1, 1), 'from_position': (1, 1), 'failure_reason': 'forced'}],
        2: [{'turn': 0, 'attempted_move': (4, 1), 'from_position': (4, 1), 'failure_reason': 'forced'}],
    }

    # Make planner fail for one specific agent only.
    original_replan_subset = engine.multi_agent_planner.replan_subset

    def controlled_replan_subset(agents, map_state, agent_ids):
        if 2 in agent_ids:
            return PlannerResult(solutions={}, status=PLANNER_STATUS_FAILED_NO_SOLUTION)
        return original_replan_subset(agents, map_state, agent_ids)

    with patch.object(engine.multi_agent_planner, "replan_subset", side_effect=controlled_replan_subset):
        conflict = engine.detect_stagnation_conflicts()

    assert conflict['has_conflicts'] is True
    assert 2 in conflict['conflicting_agents']
    assert 1 not in conflict['conflicting_agents']


def test_disable_llm_toggle_is_loaded_from_environment():
    engine = _build_engine_with_simple_layout("astar", disable_llm=True)
    assert engine.disable_llm_negotiation is True


def test_turn_based_planner_no_solution_fails_immediately():
    engine = _build_engine_with_simple_layout("astar", disable_llm=True)
    engine.initialize_simulation()

    with patch.object(
        engine,
        "_get_sequential_plan_result",
        return_value=PlannerResult(solutions={}, status=PLANNER_STATUS_FAILED_NO_SOLUTION),
    ):
        step_ok = engine.run_simulation_step()

    assert step_ok is False
    assert engine.simulation_failed is True
    assert engine.failure_reason == "mapf_failed_no_solution"


def test_turn_based_disables_llm_and_fails_on_deadlock_trigger():
    engine = _build_engine_with_simple_layout("astar", disable_llm=True)
    engine.initialize_simulation()

    fake_path = {1: [(1, 1), (1, 2)]}
    fake_conflict = {
        'has_conflicts': True,
        'conflicting_agents': [1],
        'conflict_points': [(1, 1)],
    }
    fake_deadlock = {
        'has_conflicts': True,
        'conflicting_agents': [1],
        'conflict_points': [(1, 1)],
        'agents': [],
    }

    with patch.object(
        engine,
        "_get_sequential_plan_result",
        return_value=PlannerResult(solutions=fake_path, status="success"),
    ), patch.object(
        engine.conflict_detector,
        "detect_path_conflicts",
        return_value=fake_conflict,
    ), patch.object(
        engine,
        "detect_move_failure_deadlocks",
        return_value=fake_deadlock,
    ), patch.object(engine, "_negotiate_conflicts") as mocked_negotiate:
        step_ok = engine.run_simulation_step()

    assert step_ok is False
    assert engine.simulation_failed is True
    assert engine.failure_reason == "deadlock_triggered_llm_disabled"
    mocked_negotiate.assert_not_called()


def test_async_disables_background_negotiation_and_fails_on_deadlock_trigger():
    engine = _build_engine_with_simple_layout("astar", disable_llm=True)
    engine.initialize_simulation()
    engine.simulation_mode = "async"

    fake_path = {1: [(1, 1), (1, 2)]}
    fake_conflict = {
        'has_conflicts': True,
        'conflicting_agents': [1],
        'conflict_points': [(1, 1)],
    }
    fake_deadlock = {
        'has_conflicts': True,
        'conflicting_agents': [1],
        'conflict_points': [(1, 1)],
        'agents': [],
    }

    with patch.object(
        engine,
        "_get_sequential_plan_result",
        return_value=PlannerResult(solutions=fake_path, status="success"),
    ), patch.object(
        engine.conflict_detector,
        "detect_path_conflicts",
        return_value=fake_conflict,
    ), patch.object(
        engine,
        "detect_move_failure_deadlocks",
        return_value=fake_deadlock,
    ), patch.object(engine, "_start_background_negotiation") as mocked_bg:
        step_ok = engine.run_simulation_step()

    assert step_ok is False
    assert engine.simulation_failed is True
    assert engine.failure_reason == "deadlock_triggered_llm_disabled"
    mocked_bg.assert_not_called()
