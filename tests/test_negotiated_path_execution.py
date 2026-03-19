#!/usr/bin/env python3
"""
Test for verifying negotiated paths execute fully in a single turn
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.simulation.game_engine import GameEngine
from src.map_generator import WarehouseMap
from src.agents import RobotAgent


def test_negotiated_path_full_execution():
    """Test that negotiated paths execute all steps in the same turn"""

    # Create a simple 10x10 map
    game_engine = GameEngine(width=10, height=10, num_agents=1)

    # Create a simple warehouse with no obstacles in the center
    warehouse = game_engine.warehouse_map

    # Place agent at (2, 2)
    agent = RobotAgent(0, (2, 2))
    agent.set_target((5, 5))
    game_engine.agents[0] = agent

    # Create a mock negotiation resolution with a 3-step path
    resolution = {
        'agent_actions': {
            '0': {
                'action': 'move',
                'path': [(3, 2), (4, 2), (5, 2)]  # 3 steps to the right
            }
        }
    }

    # Mark agent as HMAS-2 validated
    agent._hmas2_validated = True

    # Record initial position
    initial_position = agent.position
    print(f"Initial position: {initial_position}")

    # Execute negotiated actions
    game_engine._execute_negotiated_actions(resolution)

    # Verify agent moved all 3 steps
    final_position = agent.position
    print(f"Final position: {final_position}")

    # Agent should be at (5, 2) after executing all 3 steps
    assert final_position == (5, 2), f"Expected (5, 2), but got {final_position}"

    # Negotiated path flag should be cleared
    assert not getattr(agent, '_has_negotiated_path', False), "Negotiated path flag should be cleared"

    # Planned path should be empty
    assert len(agent.planned_path) == 0, f"Planned path should be empty, but has {len(agent.planned_path)} steps"

    print("✅ Test passed: Negotiated path executed fully in single turn")


def test_negotiated_path_with_wait():
    """Test that wait actions are handled correctly"""

    # Create a simple 10x10 map
    game_engine = GameEngine(width=10, height=10, num_agents=1)

    # Place agent at (2, 2)
    agent = RobotAgent(0, (2, 2))
    game_engine.agents[0] = agent

    # Create a mock negotiation resolution with wait action
    resolution = {
        'agent_actions': {
            '0': {
                'action': 'wait',
                'wait_turns': 1
            }
        }
    }

    # Record initial position
    initial_position = agent.position
    print(f"Initial position: {initial_position}")

    # Execute negotiated actions
    game_engine._execute_negotiated_actions(resolution)

    # Verify agent stayed in place
    final_position = agent.position
    print(f"Final position: {final_position}")

    assert final_position == initial_position, f"Agent should stay at {initial_position}, but moved to {final_position}"

    print("✅ Test passed: Wait action handled correctly")


def test_negotiated_path_with_current_position():
    """Test that paths starting with current position are handled correctly"""

    # Create a simple 10x10 map
    game_engine = GameEngine(width=10, height=10, num_agents=1)

    # Place agent at (2, 2)
    agent = RobotAgent(0, (2, 2))
    game_engine.agents[0] = agent

    # Create a mock negotiation resolution with path starting at current position
    resolution = {
        'agent_actions': {
            '0': {
                'action': 'move',
                'path': [(2, 2), (3, 2), (4, 2), (5, 2)]  # Includes current position
            }
        }
    }

    # Mark agent as HMAS-2 validated
    agent._hmas2_validated = True

    # Record initial position
    initial_position = agent.position
    print(f"Initial position: {initial_position}")

    # Execute negotiated actions
    game_engine._execute_negotiated_actions(resolution)

    # Verify agent moved to the end of the path
    final_position = agent.position
    print(f"Final position: {final_position}")

    # Agent should be at (5, 2)
    assert final_position == (5, 2), f"Expected (5, 2), but got {final_position}"

    print("✅ Test passed: Path with current position handled correctly")


if __name__ == "__main__":
    print("Running negotiated path execution tests...\n")

    try:
        test_negotiated_path_full_execution()
        print()
        test_negotiated_path_with_wait()
        print()
        test_negotiated_path_with_current_position()
        print()
        print("="*60)
        print("✅ All tests passed!")
        print("="*60)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
