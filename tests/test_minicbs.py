"""Unit tests for bounded mini-CBS subset repair."""

import os
import sys
from unittest.mock import patch


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents import RobotAgent
from src.navigation import ConflictDetector
from src.navigation.minicbs import MiniCBSRepair


def _open_map_state(width: int = 6, height: int = 6) -> dict:
    grid = [['.' for _ in range(width)] for _ in range(height)]
    return {
        'grid': grid,
        'agents': {},
        'boxes': {},
        'targets': {},
        'agent_goals': {},
    }


def test_minicbs_repairs_swap_conflict_without_internal_collisions():
    repair = MiniCBSRepair(
        width=6,
        height=6,
        min_time_horizon=16,
        grid_horizon_multiplier=2,
        max_nodes=128,
    )
    agents = {
        1: RobotAgent(agent_id=1, initial_position=(1, 2), target_position=(3, 2)),
        2: RobotAgent(agent_id=2, initial_position=(3, 2), target_position=(1, 2)),
    }
    map_state = _open_map_state()

    replanned = repair.replan_subset(agents, map_state, {1, 2})

    assert set(replanned.keys()) == {1, 2}
    assert replanned[1][-1] == (3, 2)
    assert replanned[2][-1] == (1, 2)

    conflicts = ConflictDetector(6, 6).detect_path_conflicts(replanned, 0)
    assert not conflicts['has_conflicts']


def test_minicbs_respects_non_replanned_agent_as_static_dynamic_obstacle():
    repair = MiniCBSRepair(
        width=6,
        height=6,
        min_time_horizon=16,
        grid_horizon_multiplier=2,
        max_nodes=128,
    )
    agents = {
        1: RobotAgent(agent_id=1, initial_position=(1, 2), target_position=(4, 2)),
        2: RobotAgent(agent_id=2, initial_position=(4, 2), target_position=(1, 2)),
        3: RobotAgent(agent_id=3, initial_position=(2, 2), target_position=(2, 2)),
    }
    agents[3].planned_path = [(2, 2)]
    map_state = _open_map_state()

    replanned = repair.replan_subset(agents, map_state, {1, 2})

    assert set(replanned.keys()) == {1, 2}
    for path in replanned.values():
        assert (2, 2) not in path


def test_minicbs_returns_empty_when_node_budget_is_exhausted():
    repair = MiniCBSRepair(
        width=6,
        height=6,
        min_time_horizon=16,
        grid_horizon_multiplier=2,
        max_nodes=8,
    )
    agents = {
        1: RobotAgent(agent_id=1, initial_position=(1, 2), target_position=(3, 2)),
        2: RobotAgent(agent_id=2, initial_position=(3, 2), target_position=(1, 2)),
    }
    map_state = _open_map_state()

    def forced_conflict(_paths):
        return {
            "type": "vertex",
            "a1": 1,
            "a2": 2,
            "loc": (2, 2),
            "t": 1,
        }

    def fixed_path(start, goal, walls, static_table, constraints, max_time_steps):
        del walls, static_table, constraints, max_time_steps
        return [start, goal]

    with patch.object(repair, "_first_conflict", forced_conflict), patch.object(
        repair,
        "_find_path",
        fixed_path,
    ):
        replanned = repair.replan_subset(agents, map_state, {1, 2})

    assert replanned == {}