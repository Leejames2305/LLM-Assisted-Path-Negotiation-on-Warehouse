#!/usr/bin/env python3
"""
Test script to verify main.py produces visualization-compatible logs
"""

import os
import sys
import json
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.simulation.game_engine import GameEngine

def test_main_logging_format():
    """Test that main.py GameEngine produces logs compatible with visualization"""
    print("🧪 Testing main.py logging format compatibility...")
    
    # Create a simple GameEngine instance
    game_engine = GameEngine(width=6, height=4, num_agents=2)
    
    # Initialize simulation (this should set up scenario data)
    game_engine.initialize_simulation()
    
    # Log a few turns
    game_engine.current_turn = 1
    game_engine._log_turn_state('routine')
    
    game_engine.current_turn = 2
    game_engine._log_turn_state('conflict')
    
    game_engine.current_turn = 3
    game_engine._log_turn_state('negotiation')
    
    # Mark simulation as complete
    game_engine.simulation_complete = True
    
    # Save the log
    log_path = game_engine.save_simulation_log("test_main_format.json")
    
    # Verify the log structure
    print(f"\n📁 Log saved to: {log_path}")
    
    if log_path is None:
        print("❌ Error: No log file was created")
        return None
    
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    
    # Check required fields for visualization compatibility
    required_fields = ['scenario', 'turns', 'summary']
    print(f"\n✅ Checking required top-level fields:")
    for field in required_fields:
        if field in log_data:
            print(f"  ✅ {field}: ✓")
        else:
            print(f"  ❌ {field}: MISSING")
    
    # Check scenario structure
    if 'scenario' in log_data:
        scenario = log_data['scenario']
        scenario_fields = ['type', 'map_size', 'initial_agents', 'initial_targets', 'grid', 'timestamp']
        print(f"\n✅ Checking scenario fields:")
        for field in scenario_fields:
            if field in scenario:
                print(f"  ✅ {field}: ✓")
            else:
                print(f"  ❌ {field}: MISSING")
        
        # Check grid data
        if 'grid' in scenario and scenario['grid']:
            print(f"  📊 Grid data: {len(scenario['grid'])}x{len(scenario['grid'][0])} cells")
    
    # Check turns structure
    if 'turns' in log_data and log_data['turns']:
        turn = log_data['turns'][0]
        turn_fields = ['turn', 'timestamp', 'type', 'agent_states', 'map_state', 'conflicts_detected', 'negotiation_occurred', 'results']
        print(f"\n✅ Checking turn structure (turn 1):")
        for field in turn_fields:
            if field in turn:
                print(f"  ✅ {field}: ✓")
            else:
                print(f"  ❌ {field}: MISSING")
    
    # Check summary structure
    if 'summary' in log_data:
        summary = log_data['summary']
        summary_fields = ['total_turns', 'total_conflicts', 'total_negotiations', 'completion_timestamp', 'negotiation_turns', 'routine_turns']
        print(f"\n✅ Checking summary fields:")
        for field in summary_fields:
            if field in summary:
                print(f"  ✅ {field}: ✓")
            else:
                print(f"  ❌ {field}: MISSING")
    
    print(f"\n📊 Test Summary:")
    print(f"  Total turns logged: {len(log_data.get('turns', []))}")
    print(f"  Scenario type: {log_data.get('scenario', {}).get('type', 'N/A')}")
    print(f"  Grid data available: {'✅' if log_data.get('scenario', {}).get('grid') else '❌'}")
    
    print(f"\n🎯 Result: Main.py logging format is {'COMPATIBLE' if all(field in log_data for field in required_fields) else 'INCOMPATIBLE'} with visualization script!")
    
    return log_path

if __name__ == "__main__":
    test_main_logging_format()