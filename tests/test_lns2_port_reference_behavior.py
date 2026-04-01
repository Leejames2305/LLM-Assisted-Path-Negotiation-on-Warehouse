"""
Behavior tests for the MAPF-LNS2-inspired planner loop.
"""

import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents import RobotAgent
from src.navigation import SimplePathfinder
from src.navigation.planners import (
    AStarReservationPlanner,
    LNS2Planner,
    PLANNER_STATUS_FAILED_NO_SOLUTION,
    PLANNER_STATUS_PARTIAL_SUCCESS,
    PLANNER_STATUS_SUCCESS,
    PlannerResult,
)


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


def _build_map_state_with_agents(agent_positions: dict, width: int = 6, height: int = 6) -> dict:
    grid = [['.' for _ in range(width)] for _ in range(height)]
    return {
        'grid': grid,
        'agents': dict(agent_positions),
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

    planned_result = planner.plan_all(agents, map_state, None)
    planned = planned_result.solutions

    assert planned_result.status == PLANNER_STATUS_SUCCESS
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

    planned_result = planner.plan_all(agents, map_state, None)
    planned = planned_result.solutions

    assert planned_result.status == PLANNER_STATUS_SUCCESS
    assert planned
    assert any(abs(a - b) > 1e-9 for a, b in zip(before, planner.destroy_weights))


def test_lns2_replan_subset_respects_subset_keys():
    planner = _make_lns2("intersection")
    agents = _build_agents()
    map_state = _build_open_map_state()

    initial_result = planner.plan_all(agents, map_state, None)
    initial = initial_result.solutions
    assert initial

    replanned_result = planner.replan_subset(agents, map_state, {1, 2})
    replanned = replanned_result.solutions
    assert set(replanned.keys()).issubset({1, 2})
    assert replanned


def test_lns2_returns_partial_success_when_conflicts_remain():
    planner = _make_lns2("randomwalk")
    agents = _build_agents()
    map_state = _build_open_map_state()

    # Force conflict metric to remain non-zero to verify best-effort status.
    original_conflict_count = planner._conflict_count

    def fake_conflict_count(_paths, _current_turn=0, _path_table=None):
        return 1

    planner._conflict_count = fake_conflict_count
    try:
        result = planner.plan_all(agents, map_state, None)
    finally:
        planner._conflict_count = original_conflict_count

    assert result.status == PLANNER_STATUS_PARTIAL_SUCCESS
    assert result.solutions


def test_lns2_stall_increases_destroy_ratio():
    planner = _make_lns2("randomwalk")
    planner.repair_backend = "astar"
    planner.adaptive_stall_iterations = 1
    planner.destroy_ratio_step = 0.1
    planner.iterations = 4
    before = planner.destroy_ratio

    agents = _build_agents()
    map_state = _build_open_map_state()

    original_replan = planner.base_planner.replan_subset

    def always_fail_replan(_agents, _map_state, _agent_ids):
        return PlannerResult(solutions={}, status=PLANNER_STATUS_FAILED_NO_SOLUTION)

    planner.base_planner.replan_subset = always_fail_replan
    try:
        _ = planner.plan_all(agents, map_state, None)
    finally:
        planner.base_planner.replan_subset = original_replan

    assert planner.destroy_ratio > before


def test_lns2_replan_subset_prefers_minicbs_backend_when_available():
    planner = _make_lns2("randomwalk")
    planner.repair_backend = "minicbs"
    agents = _build_agents()
    map_state = _build_open_map_state()

    expected = {
        1: [(1, 1), (2, 1), (3, 1), (4, 1)],
        2: [(4, 1), (3, 1), (2, 1), (1, 1)],
    }

    original_minicbs_replan = planner.minicbs_repair.replan_subset
    original_base_replan = planner.base_planner.replan_subset

    def minicbs_success(_agents, _map_state, _agent_ids):
        return expected

    def base_should_not_run(_agents, _map_state, _agent_ids):
        raise AssertionError("base replanner should not run when mini-CBS succeeds")

    planner.minicbs_repair.replan_subset = minicbs_success
    planner.base_planner.replan_subset = base_should_not_run
    try:
        result = planner.replan_subset(agents, map_state, {1, 2})
    finally:
        planner.minicbs_repair.replan_subset = original_minicbs_replan
        planner.base_planner.replan_subset = original_base_replan

    assert result.status == PLANNER_STATUS_SUCCESS
    assert result.solutions == expected


def test_lns2_replan_subset_falls_back_to_astar_when_minicbs_fails():
    planner = _make_lns2("randomwalk")
    planner.repair_backend = "minicbs"
    agents = _build_agents()
    map_state = _build_open_map_state()

    expected = {
        1: [(1, 1), (1, 2), (2, 2), (3, 2), (4, 2), (4, 1)],
        2: [(4, 1), (4, 0), (3, 0), (2, 0), (1, 0), (1, 1)],
    }

    original_minicbs_replan = planner.minicbs_repair.replan_subset
    original_base_replan = planner.base_planner.replan_subset

    def minicbs_fail(_agents, _map_state, _agent_ids):
        return {}

    def base_success(_agents, _map_state, _agent_ids):
        return PlannerResult(solutions=expected, status=PLANNER_STATUS_SUCCESS)

    planner.minicbs_repair.replan_subset = minicbs_fail
    planner.base_planner.replan_subset = base_success
    try:
        result = planner.replan_subset(agents, map_state, {1, 2})
    finally:
        planner.minicbs_repair.replan_subset = original_minicbs_replan
        planner.base_planner.replan_subset = original_base_replan

    assert result.status == PLANNER_STATUS_SUCCESS
    assert result.solutions == expected


def test_lns2_randomwalk_chain_expands_beyond_seed_collisions():
    planner = _make_lns2("randomwalk")
    planner.destroy_ratio = 1.0
    planner.randomwalk_samples_per_agent = 10
    planner.randomwalk_max_expansions = 10

    agents = {
        1: RobotAgent(agent_id=1, initial_position=(0, 0), target_position=(2, 0)),
        2: RobotAgent(agent_id=2, initial_position=(1, 1), target_position=(2, 1)),
        3: RobotAgent(agent_id=3, initial_position=(2, 2), target_position=(2, 1)),
    }
    solution = {
        1: [(0, 0), (1, 0), (2, 0)],
        2: [(1, 1), (1, 0), (1, 1), (2, 1)],
        3: [(2, 2), (2, 1), (1, 1), (2, 1)],
    }
    table, _ = planner._build_solution_path_table(solution)

    original_seed_picker = planner._find_most_delayed_agent
    planner._find_most_delayed_agent = lambda _sol, _agents: 1
    try:
        subset = planner._select_randomwalk_subset(solution, agents, table)
    finally:
        planner._find_most_delayed_agent = original_seed_picker

    # Agent 1 conflicts with 2, and 2 further conflicts with 3 in later timesteps.
    assert 1 in subset
    assert 2 in subset
    assert 3 in subset


def test_lns2_plan_all_treats_finished_agents_as_static_obstacles():
    planner = _make_lns2("randomwalk")
    agents = {
        1: RobotAgent(agent_id=1, initial_position=(2, 1), target_position=None),
        2: RobotAgent(agent_id=2, initial_position=(1, 1), target_position=(3, 1)),
    }
    map_state = _build_map_state_with_agents({1: (2, 1), 2: (1, 1)})

    result = planner.plan_all(agents, map_state, None)
    path = result.solutions.get(2, [])

    assert result.status == PLANNER_STATUS_SUCCESS
    assert path
    assert path[-1] == (3, 1)
    assert (2, 1) not in path[1:]


def test_lns2_subset_replan_treats_finished_agents_as_static_obstacles():
    planner = _make_lns2("randomwalk")
    planner.repair_backend = "minicbs"
    agents = {
        1: RobotAgent(agent_id=1, initial_position=(2, 1), target_position=None),
        2: RobotAgent(agent_id=2, initial_position=(1, 1), target_position=(3, 1)),
    }
    map_state = _build_map_state_with_agents({1: (2, 1), 2: (1, 1)})

    result = planner.replan_subset(agents, map_state, {2})
    path = result.solutions.get(2, [])

    assert result.status == PLANNER_STATUS_SUCCESS
    assert path
    assert path[-1] == (3, 1)
    assert (2, 1) not in path[1:]
