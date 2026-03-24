"""
Tests for fallback handling when negotiated responses miss agent actions.
"""

import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents import RobotAgent
from src.simulation.game_engine import GameEngine


def _build_engine_with_agents() -> GameEngine:
    engine = GameEngine(width=5, height=5, num_agents=2)
    engine.agents = {
        1: RobotAgent(agent_id=1, initial_position=(1, 1), target_position=(1, 3)),
        2: RobotAgent(agent_id=2, initial_position=(2, 2), target_position=(2, 4)),
    }
    return engine


def test_missing_action_gets_sequential_astar_retry():
    engine = _build_engine_with_agents()

    def fake_sequential_planner(agent_ids):
        assert 2 in agent_ids
        return {2: [(2, 2), (2, 3), (2, 4)]}

    engine._get_sequential_planned_moves = fake_sequential_planner  # type: ignore

    resolution = {
        'resolution': 'reroute',
        'agent_actions': {
            '1': {'action': 'move', 'path': [[1, 1], [1, 2]]}
        },
        'reasoning': 'initial'
    }

    updated = engine._apply_sequential_astar_fallback_for_missing_actions(resolution, [1, 2])

    assert '2' in updated['agent_actions']
    assert updated['agent_actions']['2']['action'] == 'move'
    assert updated['agent_actions']['2']['path'] == [[2, 2], [2, 3], [2, 4]]
    assert 'sequential A* retry' in updated['reasoning']


def test_top_level_action_format_is_normalized_before_fallback():
    engine = _build_engine_with_agents()

    engine._get_sequential_planned_moves = lambda agent_ids: {2: [(2, 2)]}  # type: ignore

    resolution = {
        '1': {'action': 'wait', 'path': [[1, 1], [1, 1]]},
        'reasoning': 'top_level_format'
    }

    updated = engine._apply_sequential_astar_fallback_for_missing_actions(resolution, [1, 2])

    assert 'agent_actions' in updated
    assert '1' in updated['agent_actions']
    assert '2' in updated['agent_actions']
    assert updated['agent_actions']['2']['action'] == 'wait'
    assert updated['agent_actions']['2']['path'] == [[2, 2]]
