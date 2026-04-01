"""Stress tests for dense LNS2 + mini-CBS scenarios."""

import os
import random
import sys
from unittest.mock import patch

import pytest


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents import RobotAgent
from src.navigation import ConflictDetector, SimplePathfinder
from src.navigation.planners import (
    AStarReservationPlanner,
    LNS2Planner,
    PLANNER_STATUS_FAILED_NO_SOLUTION,
    PLANNER_STATUS_PARTIAL_SUCCESS,
    PLANNER_STATUS_SUCCESS,
)


def _dense_slots(width: int, height: int):
    return [(x, y) for y in range(height) for x in range(width)]


def _build_dense_agents(num_agents: int, width: int, height: int) -> dict:
    slots = _dense_slots(width, height)
    assert num_agents <= len(slots)

    starts = slots[:num_agents]
    shift = max(1, num_agents // 2)
    goals = starts[shift:] + starts[:shift]

    agents = {}
    for idx, (start, goal) in enumerate(zip(starts, goals), start=1):
        agents[idx] = RobotAgent(agent_id=idx, initial_position=start, target_position=goal)
    return agents


def _map_state_from_agents(agents: dict, width: int, height: int) -> dict:
    return {
        'grid': [['.' for _ in range(width)] for _ in range(height)],
        'agents': {aid: agent.position for aid, agent in agents.items()},
        'boxes': {},
        'targets': {},
        'agent_goals': {},
    }


def _make_dense_lns2(width: int, height: int, iterations: int = 6, seed: int = 13) -> LNS2Planner:
    base = AStarReservationPlanner(
        pathfinder=SimplePathfinder(width, height),
        width=width,
        height=height,
        min_time_horizon=32,
        grid_horizon_multiplier=1,
    )
    return LNS2Planner(
        base_planner=base,
        width=width,
        height=height,
        iterations=iterations,
        destroy_ratio=0.45,
        seed=seed,
        destroy_strategy="randomwalk",
        phase1_ratio=0.7,
        max_init_retries=3,
        adaptive_stall_iterations=3,
        destroy_ratio_step=0.1,
        repair_backend="minicbs",
        minicbs_max_nodes=256,
        low_level_backend="astar",
    )


@pytest.mark.parametrize("num_agents", [8, 16, 24, 32])
def test_lns2_minicbs_dense_plan_all_stress(num_agents):
    width, height = 10, 10
    planner = _make_dense_lns2(width, height, iterations=6, seed=17 + num_agents)
    agents = _build_dense_agents(num_agents, width, height)
    map_state = _map_state_from_agents(agents, width, height)

    original_replan = planner.minicbs_repair.replan_subset
    minicbs_calls = {"count": 0}

    def counting_replan(*args, **kwargs):
        minicbs_calls["count"] += 1
        return original_replan(*args, **kwargs)

    with patch.object(planner.minicbs_repair, "replan_subset", side_effect=counting_replan):
        result = planner.plan_all(agents, map_state, None)

    assert minicbs_calls["count"] > 0
    assert result.status in {PLANNER_STATUS_SUCCESS, PLANNER_STATUS_PARTIAL_SUCCESS}
    assert len(result.solutions) >= max(1, num_agents // 2)

    for aid, path in result.solutions.items():
        assert path
        assert path[0] == agents[aid].position
        assert path[-1] == agents[aid].target_position

    if result.status == PLANNER_STATUS_SUCCESS:
        conflicts = ConflictDetector(width, height).detect_path_conflicts(result.solutions, 0)
        assert not conflicts['has_conflicts']


def test_lns2_minicbs_dense_replan_loop_stress_32_agents():
    width, height = 10, 10
    planner = _make_dense_lns2(width, height, iterations=4, seed=31)
    agents = _build_dense_agents(32, width, height)
    map_state = _map_state_from_agents(agents, width, height)

    initial = planner.plan_all(agents, map_state, None)
    assert initial.solutions

    for aid, path in initial.solutions.items():
        agents[aid].planned_path = path.copy()

    original_replan = planner.minicbs_repair.replan_subset
    minicbs_calls = {"count": 0}

    def counting_replan(*args, **kwargs):
        minicbs_calls["count"] += 1
        return original_replan(*args, **kwargs)

    rng = random.Random(23)
    non_empty_rounds = 0

    with patch.object(planner.minicbs_repair, "replan_subset", side_effect=counting_replan):
        for _ in range(8):
            subset = set(rng.sample(sorted(agents.keys()), 8))
            result = planner.replan_subset(agents, map_state, subset)

            assert result.status in {
                PLANNER_STATUS_SUCCESS,
                PLANNER_STATUS_PARTIAL_SUCCESS,
                PLANNER_STATUS_FAILED_NO_SOLUTION,
            }
            assert set(result.solutions.keys()).issubset(subset)

            if result.solutions:
                non_empty_rounds += 1
            for aid, path in result.solutions.items():
                assert path
                assert path[0] == agents[aid].position
                agents[aid].planned_path = path.copy()

    assert minicbs_calls["count"] > 0
    assert non_empty_rounds >= 1