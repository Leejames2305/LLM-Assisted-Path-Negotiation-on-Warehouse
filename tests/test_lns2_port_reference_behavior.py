"""
Behavior tests for the MAPF-LNS2-inspired planner loop.
"""

import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents import RobotAgent
from src.navigation import SimplePathfinder
from src.navigation.planners import AStarReservationPlanner, LNS2Planner


def _build_agents() -> dict:
    return {
        1: RobotAgent(agent_id=1, initial_position=(1, 1), target_position=(4, 1)),
        2: RobotAgent(agent_id=2, initial_position=(4, 1), target_position=(1, 1)),
        3: RobotAgent(agent_id=3, initial_position=(1, 3), target_position=(4, 3)),
    }


def _build_open_map_state(width: int = 6, height: int = 6) -> dict:
    grid = [['.' for _ in range(width)] for _ in range(height)]
    return {
        'grid': grid,
        'agents': {1: (1, 1), 2: (4, 1), 3: (1, 3)},
        'boxes': {},
        'targets': {},
        'agent_goals': {},
    }


def _make_lns2(strategy: str = "randomwalk") -> LNS2Planner:
    base = AStarReservationPlanner(
        pathfinder=SimplePathfinder(6, 6),
        width=6,
        height=6,
        min_time_horizon=16,
        grid_horizon_multiplier=2,
    )
    return LNS2Planner(
        base_planner=base,
        width=6,
        height=6,
        iterations=5,
        destroy_ratio=0.5,
        seed=7,
        destroy_strategy=strategy,
    )


def test_lns2_randomwalk_plan_all_returns_paths_for_active_agents():
    planner = _make_lns2("randomwalk")
    agents = _build_agents()
    map_state = _build_open_map_state()

    planned = planner.plan_all(agents, map_state, None)

    assert set(planned.keys()) == {1, 2, 3}
    for aid, path in planned.items():
        assert path, f"agent {aid} should have non-empty path"
        assert path[0] == agents[aid].position
        assert path[-1] == agents[aid].target_position


def test_lns2_adaptive_updates_destroy_weights_over_iterations():
    planner = _make_lns2("adaptive")
    agents = _build_agents()
    map_state = _build_open_map_state()
    before = planner.destroy_weights.copy()

    planned = planner.plan_all(agents, map_state, None)

    assert planned
    assert any(abs(a - b) > 1e-9 for a, b in zip(before, planner.destroy_weights))


def test_lns2_replan_subset_respects_subset_keys():
    planner = _make_lns2("intersection")
    agents = _build_agents()
    map_state = _build_open_map_state()

    initial = planner.plan_all(agents, map_state, None)
    assert initial

    replanned = planner.replan_subset(agents, map_state, {1, 2})
    assert set(replanned.keys()).issubset({1, 2})
    assert replanned
