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
from src.simulation.game_engine import GameEngine


def _empty_grid(width: int, height: int):
    return [['.' for _ in range(width)] for _ in range(height)]


def _build_engine_with_simple_layout(planner_mode: str = "astar") -> GameEngine:
    prev_mode = os.environ.get("PATH_PLANNER_MODE")
    os.environ["PATH_PLANNER_MODE"] = planner_mode
    try:
        engine = GameEngine(width=6, height=6, num_agents=2)
    finally:
        if prev_mode is None:
            os.environ.pop("PATH_PLANNER_MODE", None)
        else:
            os.environ["PATH_PLANNER_MODE"] = prev_mode

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
            return {}
        return original_replan_subset(agents, map_state, agent_ids)

    with patch.object(engine.multi_agent_planner, "replan_subset", side_effect=controlled_replan_subset):
        conflict = engine.detect_stagnation_conflicts()

    assert conflict['has_conflicts'] is True
    assert 2 in conflict['conflicting_agents']
    assert 1 not in conflict['conflicting_agents']
