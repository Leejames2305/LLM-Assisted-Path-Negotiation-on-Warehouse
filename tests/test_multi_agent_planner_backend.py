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


def test_lns2_repair_backend_env_selection():
    previous_mode = os.environ.get('PATH_PLANNER_MODE')
    previous_repair = os.environ.get('LNS2_REPAIR_BACKEND')
    os.environ['PATH_PLANNER_MODE'] = 'LNS2'
    os.environ['LNS2_REPAIR_BACKEND'] = 'minicbs'
    try:
        engine = GameEngine(width=6, height=6, num_agents=2)
        assert engine.multi_agent_planner.get_backend_name() == 'LNS2'
        assert hasattr(engine.multi_agent_planner, 'repair_backend')
        assert engine.multi_agent_planner.repair_backend == 'minicbs'
    finally:
        if previous_mode is None:
            os.environ.pop('PATH_PLANNER_MODE', None)
        else:
            os.environ['PATH_PLANNER_MODE'] = previous_mode
        if previous_repair is None:
            os.environ.pop('LNS2_REPAIR_BACKEND', None)
        else:
            os.environ['LNS2_REPAIR_BACKEND'] = previous_repair


def test_lns2_low_level_backend_env_selection():
    previous_mode = os.environ.get('PATH_PLANNER_MODE')
    previous_low_level = os.environ.get('LNS2_LOW_LEVEL_BACKEND')
    os.environ['PATH_PLANNER_MODE'] = 'LNS2'
    os.environ['LNS2_LOW_LEVEL_BACKEND'] = 'sipp'
    try:
        engine = GameEngine(width=6, height=6, num_agents=2)
        assert engine.multi_agent_planner.get_backend_name() == 'LNS2'
        assert hasattr(engine.multi_agent_planner, 'low_level_backend')
        assert engine.multi_agent_planner.low_level_backend == 'sipp'
        assert engine.multi_agent_planner.base_planner.low_level_backend == 'sipp'
    finally:
        if previous_mode is None:
            os.environ.pop('PATH_PLANNER_MODE', None)
        else:
            os.environ['PATH_PLANNER_MODE'] = previous_mode
        if previous_low_level is None:
            os.environ.pop('LNS2_LOW_LEVEL_BACKEND', None)
        else:
            os.environ['LNS2_LOW_LEVEL_BACKEND'] = previous_low_level


def test_lns2_minicbs_max_nodes_env_selection():
    previous_mode = os.environ.get('PATH_PLANNER_MODE')
    previous_nodes = os.environ.get('LNS2_MINICBS_MAX_NODES')
    os.environ['PATH_PLANNER_MODE'] = 'LNS2'
    os.environ['LNS2_MINICBS_MAX_NODES'] = '256'
    try:
        engine = GameEngine(width=6, height=6, num_agents=2)
        assert engine.multi_agent_planner.get_backend_name() == 'LNS2'
        assert hasattr(engine.multi_agent_planner, 'minicbs_repair')
        assert engine.multi_agent_planner.minicbs_repair.max_nodes == 256
    finally:
        if previous_mode is None:
            os.environ.pop('PATH_PLANNER_MODE', None)
        else:
            os.environ['PATH_PLANNER_MODE'] = previous_mode
        if previous_nodes is None:
            os.environ.pop('LNS2_MINICBS_MAX_NODES', None)
        else:
            os.environ['LNS2_MINICBS_MAX_NODES'] = previous_nodes


def test_replan_subset_updates_subset_only():
    engine = GameEngine(width=6, height=6, num_agents=2)
    engine.agents = {
        1: RobotAgent(agent_id=1, initial_position=(1, 1), target_position=(1, 4)),
        2: RobotAgent(agent_id=2, initial_position=(2, 1), target_position=(2, 4)),
    }

    engine.agents[1].planned_path = [(1, 1)]
    engine.agents[2].planned_path = [(2, 1)]

    map_state = engine.warehouse_map.get_state_dict()
    planned_result = engine.multi_agent_planner.replan_subset(engine.agents, map_state, {1})
    planned = planned_result.solutions
    for agent_id, path in planned.items():
        engine.agents[agent_id].planned_path = path.copy()

    assert 1 in planned
    assert 2 not in planned
    assert engine.agents[1].planned_path == planned[1]
    assert engine.agents[2].planned_path == [(2, 1)]
