"""
Regression test for deadlock A* path conflict checks against stationary agents.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents import RobotAgent
from src.simulation.game_engine import GameEngine


def test_check_path_conflicts_includes_stationary_agents():
    engine = GameEngine(width=5, height=5, num_agents=2)
    engine.silent_mode = True

    moving_agent = RobotAgent(agent_id=1, initial_position=(1, 1), target_position=(3, 1))
    stationary_agent = RobotAgent(agent_id=2, initial_position=(2, 1), target_position=None)
    stationary_agent.planned_path = []

    engine.agents = {
        1: moving_agent,
        2: stationary_agent,
    }

    candidate_path = [(1, 1), (2, 1), (3, 1)]

    assert engine._check_path_conflicts_with_others(1, candidate_path) is True

