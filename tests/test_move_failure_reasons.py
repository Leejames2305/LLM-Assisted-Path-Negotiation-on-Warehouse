"""
Test move failure reason tracking
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents import RobotAgent


def test_move_failure_out_of_bounds():
    """Test that out of bounds moves return proper failure reason"""
    agent = RobotAgent(agent_id=1, initial_position=(2, 2))
    
    # Create a simple 5x5 map state
    map_state = {
        'grid': [
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.']
        ],
        'agents': {1: (2, 2)},
        'boxes': {}
    }
    
    # Try to move out of bounds (to negative coordinates)
    success, reason = agent.move_to((-1, 2), map_state)
    
    assert success == False, "Move should fail for out of bounds position"
    assert reason is not None, "Failure reason should not be None"
    assert "not_adjacent" in reason, f"Expected 'not_adjacent' in reason, got: {reason}"
    print(f"✅ Out of bounds test passed. Reason: {reason}")


def test_move_failure_wall_collision():
    """Test that wall collision returns proper failure reason"""
    agent = RobotAgent(agent_id=1, initial_position=(2, 2))
    
    # Create a map with a wall at (2, 3)
    map_state = {
        'grid': [
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '.', '#', '.', '.'],  # Wall at (2, 3)
            ['.', '.', '.', '.', '.']
        ],
        'agents': {1: (2, 2)},
        'boxes': {}
    }
    
    # Try to move into a wall
    success, reason = agent.move_to((2, 3), map_state)
    
    assert success == False, "Move should fail for wall collision"
    assert reason is not None, "Failure reason should not be None"
    assert "wall_collision" in reason, f"Expected 'wall_collision' in reason, got: {reason}"
    print(f"✅ Wall collision test passed. Reason: {reason}")


def test_move_failure_agent_collision():
    """Test that agent collision returns proper failure reason"""
    agent = RobotAgent(agent_id=1, initial_position=(2, 2))
    
    # Create a map with another agent at (2, 3)
    map_state = {
        'grid': [
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.']
        ],
        'agents': {1: (2, 2), 2: (2, 3)},  # Agent 2 at (2, 3)
        'boxes': {}
    }
    
    # Try to move into another agent
    success, reason = agent.move_to((2, 3), map_state)
    
    assert success == False, "Move should fail for agent collision"
    assert reason is not None, "Failure reason should not be None"
    assert "agent_collision" in reason, f"Expected 'agent_collision' in reason, got: {reason}"
    assert "Agent 2" in reason, f"Expected 'Agent 2' in reason, got: {reason}"
    print(f"✅ Agent collision test passed. Reason: {reason}")


def test_successful_move():
    """Test that successful moves return None for failure reason"""
    agent = RobotAgent(agent_id=1, initial_position=(2, 2))
    
    # Create a simple map
    map_state = {
        'grid': [
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.']
        ],
        'agents': {1: (2, 2)},
        'boxes': {}
    }
    
    # Make a valid move
    success, reason = agent.move_to((2, 3), map_state)
    
    assert success == True, "Move should succeed"
    assert reason is None, f"Failure reason should be None for successful move, got: {reason}"
    assert agent.position == (2, 3), "Agent position should be updated"
    print(f"✅ Successful move test passed. Position: {agent.position}")


def test_stay_in_place():
    """Test that staying in place returns success with None reason"""
    agent = RobotAgent(agent_id=1, initial_position=(2, 2))
    
    # Create a simple map
    map_state = {
        'grid': [
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.']
        ],
        'agents': {1: (2, 2)},
        'boxes': {}
    }
    
    # Stay in place (wait)
    success, reason = agent.move_to((2, 2), map_state)
    
    assert success == True, "Staying in place should succeed"
    assert reason is None, f"Failure reason should be None for wait, got: {reason}"
    assert agent.position == (2, 2), "Agent position should remain the same"
    print(f"✅ Stay in place test passed.")


if __name__ == "__main__":
    print("Running move failure reason tests...\n")
    
    test_move_failure_out_of_bounds()
    test_move_failure_wall_collision()
    test_move_failure_agent_collision()
    test_successful_move()
    test_stay_in_place()
    
    print("\n✅ All tests passed!")
