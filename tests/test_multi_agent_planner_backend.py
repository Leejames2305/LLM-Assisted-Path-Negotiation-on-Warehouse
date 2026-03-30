"""
Tests for planner backend selection and subset repair behavior.
"""

import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents import RobotAgent
from src.simulation.game_engine import GameEngine


def test_default_planner_mode_is_astar_backend():
    previous = os.environ.pop('PATH_PLANNER_MODE', None)
    try:
        engine = GameEngine(width=6, height=6, num_agents=2)
        assert engine.multi_agent_planner.get_backend_name() == 'astar'
    finally:
        if previous is not None:
            os.environ['PATH_PLANNER_MODE'] = previous


def test_lns2_mode_uses_lns2_backend():
    previous = os.environ.get('PATH_PLANNER_MODE')
    os.environ['PATH_PLANNER_MODE'] = 'LNS2'
    try:
        engine = GameEngine(width=6, height=6, num_agents=2)
        assert engine.multi_agent_planner.get_backend_name() == 'LNS2'
    finally:
        if previous is None:
            os.environ.pop('PATH_PLANNER_MODE', None)
        else:
            os.environ['PATH_PLANNER_MODE'] = previous


def test_replan_subset_updates_subset_only():
    engine = GameEngine(width=6, height=6, num_agents=2)
    engine.agents = {
        1: RobotAgent(agent_id=1, initial_position=(1, 1), target_position=(1, 4)),
        2: RobotAgent(agent_id=2, initial_position=(2, 1), target_position=(2, 4)),
    }

    engine.agents[1].planned_path = [(1, 1)]
    engine.agents[2].planned_path = [(2, 1)]

    map_state = engine.warehouse_map.get_state_dict()
    planned = engine.multi_agent_planner.replan_subset(engine.agents, map_state, {1})
    for agent_id, path in planned.items():
        engine.agents[agent_id].planned_path = path.copy()

    assert 1 in planned
    assert 2 not in planned
    assert engine.agents[1].planned_path == planned[1]
    assert engine.agents[2].planned_path == [(2, 1)]
