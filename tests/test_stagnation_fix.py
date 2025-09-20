#!/usr/bin/env python3
"""
Test script to verify that stagnation detection now works correctly:
- Should NOT trigger for intentional waiting (negotiated paths)
- Should ONLY trigger for actual failed moves
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.simulation.game_engine import GameEngine

def test_stagnation_detection():
    """Test the improved stagnation detection logic"""
    
    print("🧪 TESTING STAGNATION DETECTION IMPROVEMENTS")
    print("=" * 60)
    
    # Create a simple game engine
    engine = GameEngine(width=8, height=6, num_agents=2)
    engine.initialize_simulation()
    
    print("\n📋 TESTING SCENARIOS:")
    print("1. Intentional waiting (negotiated path) - should NOT trigger stagnation")
    print("2. Actual failed moves - should trigger stagnation after 3 failures")
    
    # Test 1: Simulate intentional waiting
    print(f"\n🔬 TEST 1: Intentional Waiting")
    print("-" * 30)
    
    agent_0 = engine.agents[0]
    agent_1 = engine.agents[1]
    
    # Simulate agent staying in same position for multiple turns (intentional)
    initial_pos = agent_0.position
    print(f"Agent 0 initial position: {initial_pos}")
    
    # Simulate 3 turns of staying in same position without failed moves
    for turn in range(3):
        # Agent successfully "waits" in place (no failed move recorded)
        print(f"Turn {turn + 1}: Agent 0 waiting at {initial_pos} (intentional)")
        # Don't add to failed_move_history - this simulates successful waiting
    
    # Check stagnation detection
    stagnation_result = engine.detect_stagnation_conflicts()
    if stagnation_result['has_conflicts']:
        print("❌ FAILED: Stagnation incorrectly detected for intentional waiting")
    else:
        print("✅ PASSED: Stagnation correctly NOT detected for intentional waiting")
    
    # Test 2: Simulate actual failed moves
    print(f"\n🔬 TEST 2: Actual Failed Moves")
    print("-" * 30)
    
    # Simulate 3 consecutive failed moves for agent 1
    for turn in range(3):
        engine.current_turn = turn
        agent_id = 1
        
        # Simulate a failed move attempt
        if agent_id not in engine.agent_failed_move_history:
            engine.agent_failed_move_history[agent_id] = []
        
        engine.agent_failed_move_history[agent_id].append({
            'turn': turn,
            'attempted_move': (999, 999),  # Invalid position
            'from_position': agent_1.position
        })
        
        print(f"Turn {turn + 1}: Agent 1 failed to move from {agent_1.position}")
    
    # Check stagnation detection
    stagnation_result = engine.detect_stagnation_conflicts()
    if stagnation_result['has_conflicts'] and 1 in stagnation_result['conflicting_agents']:
        print("✅ PASSED: Stagnation correctly detected for actual failed moves")
        print(f"   Detected agents: {stagnation_result['conflicting_agents']}")
        agent_data = next(a for a in stagnation_result['agents'] if a['id'] == 1)
        print(f"   Failed move count: {agent_data['failed_move_count']}")
    else:
        print("❌ FAILED: Stagnation not detected for actual failed moves")
    
    print(f"\n🏁 STAGNATION DETECTION TEST COMPLETE")

if __name__ == "__main__":
    test_stagnation_detection()