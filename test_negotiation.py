#!/usr/bin/env python3
"""
LLM Negotiation Test - Forces guaranteed conflicts and logs all negotiations
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.simulation.game_engine import GameEngine
from src.map_generator import WarehouseMap

class NegotiationTester:
    """Test class for forcing conflicts and logging LLM negotiations"""
    
    def __init__(self):
        self.negotiation_log = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'test_type': 'forced_conflict_negotiation',
                'description': 'Artificial scenarios designed to force 100% conflict rate'
            },
            'negotiations': [],
            'conflict_scenarios': []
        }
    
    def create_forced_conflict_map(self, scenario_type: str = "single_corridor") -> WarehouseMap:
        """Create maps designed to guarantee conflicts"""
        
        warehouse = None
        
        if scenario_type == "single_corridor":
            # Extreme single corridor that forces all agents through same path
            width, height = 8, 3
            warehouse = WarehouseMap(width, height)
            
            # Fill with walls
            warehouse.grid.fill('#')
            
            # Create single corridor down the middle
            for x in range(1, width - 1):
                warehouse.grid[1, x] = '.'
            
            # Place agents at opposite ends
            warehouse.agents = {0: (1, 1), 1: (width-2, 1)}
            
            # Place boxes and targets to force crossing paths
            warehouse.boxes = {0: (2, 1), 1: (width-3, 1)}
            warehouse.targets = {0: (width-3, 1), 1: (2, 1)}  # Crossed targets!
            warehouse.agent_goals = {0: 0, 1: 1}
            
        elif scenario_type == "bottleneck_chamber":
            # Two chambers connected by single cell bottleneck
            width, height = 7, 5
            warehouse = WarehouseMap(width, height)
            
            # Fill with walls
            warehouse.grid.fill('#')
            
            # Left chamber
            for y in range(1, 4):
                for x in range(1, 3):
                    warehouse.grid[y, x] = '.'
            
            # Right chamber  
            for y in range(1, 4):
                for x in range(4, 6):
                    warehouse.grid[y, x] = '.'
            
            # Single bottleneck connection
            warehouse.grid[2, 3] = '.'
            
            # Place agents in opposite chambers
            warehouse.agents = {0: (1, 1), 1: (5, 1), 2: (1, 3)}
            
            # Force them to cross through bottleneck
            warehouse.boxes = {0: (2, 2), 1: (4, 2), 2: (5, 3)}
            warehouse.targets = {0: (5, 2), 1: (1, 2), 2: (2, 1)}  # All must cross!
            warehouse.agent_goals = {0: 0, 1: 1, 2: 2}
            
        elif scenario_type == "triple_intersection":
            # Three paths converging at single intersection
            width, height = 5, 5
            warehouse = WarehouseMap(width, height)
            
            # Fill with walls
            warehouse.grid.fill('#')
            
            # Create three paths converging at center
            # Horizontal path
            for x in range(5):
                warehouse.grid[2, x] = '.'
            # Vertical path
            for y in range(5):
                warehouse.grid[y, 2] = '.'
            
            # Place agents at ends of each path
            warehouse.agents = {0: (0, 2), 1: (4, 2), 2: (2, 0)}
            
            # All must pass through center (2,2)
            warehouse.boxes = {0: (1, 2), 1: (3, 2), 2: (2, 1)}
            warehouse.targets = {0: (4, 2), 1: (0, 2), 2: (2, 4)}
            warehouse.agent_goals = {0: 0, 1: 1, 2: 2}
        else:
            # Default fallback
            warehouse = WarehouseMap(8, 3)
            warehouse.generate_map(num_agents=2, layout_type="extreme")
        
        return warehouse
    
    def run_forced_conflict_test(self, scenario_type: str, max_turns: int = 20) -> Dict[str, Any]:
        """Run a single conflict scenario and capture all negotiations"""
        
        print(f"\n🎯 TESTING SCENARIO: {scenario_type.upper()}")
        print("=" * 60)
        
        # Create the forced conflict map
        warehouse = self.create_forced_conflict_map(scenario_type)
        
        # Create game engine with this specific map
        game_engine = GameEngine(width=warehouse.width, height=warehouse.height, num_agents=len(warehouse.agents))
        game_engine.warehouse_map = warehouse
        game_engine.agents = {}
        
        # Initialize agents with the pre-set positions
        from src.agents import RobotAgent  # Use RobotAgent instead of Agent
        for agent_id, position in warehouse.agents.items():
            agent = RobotAgent(agent_id, position)
            # Set target based on assigned goals
            if agent_id in warehouse.agent_goals:
                goal_id = warehouse.agent_goals[agent_id]
                if goal_id in warehouse.targets:
                    target_pos = warehouse.targets[goal_id]
                    agent.set_target(target_pos)
            game_engine.agents[agent_id] = agent
        
        # Display initial scenario
        print("🗺️  FORCED CONFLICT SCENARIO:")
        warehouse.display()
        
        print(f"\n📋 Scenario Details:")
        for agent_id, agent in game_engine.agents.items():
            start_pos = warehouse.agents[agent_id] 
            target_pos = agent.target_position
            print(f"   Agent {agent_id}: {start_pos} → {target_pos}")
        
        print(f"\n⚔️  Expected Conflicts: GUARANTEED (all agents must use same path)")
        
        # Store scenario info
        scenario_data = {
            'type': scenario_type,
            'map_size': (warehouse.width, warehouse.height),
            'agents': dict(warehouse.agents),
            'targets': dict(warehouse.targets),
            'expected_conflicts': 'GUARANTEED',
            'negotiations': []
        }
        
        # Monkey-patch the negotiation method to capture data
        original_negotiate = game_engine.central_negotiator.negotiate_path_conflict
        original_system_prompt = game_engine.central_negotiator._create_negotiation_system_prompt
        original_conflict_desc = game_engine.central_negotiator._create_conflict_description
        
        def capture_negotiate(conflict_data):
            print(f"\n🤖 CONFLICT DETECTED! Initiating LLM Negotiation...")
            print(f"   Conflicting agents: {[a.get('id') for a in conflict_data.get('agents', [])]}")
            print(f"   Conflict points: {conflict_data.get('conflict_points', [])}")
            
            # Capture the input prompt
            negotiation_entry = {
                'turn': conflict_data.get('turn', 0),
                'timestamp': datetime.now().isoformat(),
                'system_prompt': None,
                'user_prompt': None,
                'llm_response': None,
                'conflict_data': conflict_data
            }
            
            # Capture prompts
            def capture_system_prompt():
                prompt = original_system_prompt()
                negotiation_entry['system_prompt'] = prompt
                print(f"\n📋 SYSTEM PROMPT:")
                print("=" * 50)
                print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
                print("=" * 50)
                return prompt
            
            def capture_user_prompt(conflict_data):
                prompt = original_conflict_desc(conflict_data)
                negotiation_entry['user_prompt'] = prompt
                print(f"\n📝 USER PROMPT TO LLM:")
                print("=" * 50)
                print(prompt)
                print("=" * 50)
                return prompt
            
            # Temporarily replace methods to capture prompts
            game_engine.central_negotiator._create_negotiation_system_prompt = capture_system_prompt
            game_engine.central_negotiator._create_conflict_description = capture_user_prompt
            
            # Call original method and capture response
            try:
                response = original_negotiate(conflict_data)
                negotiation_entry['llm_response'] = response
                
                print(f"\n💬 LLM RESPONSE:")
                print("-" * 40)
                if isinstance(response, dict):
                    print(json.dumps(response, indent=2))
                else:
                    print(str(response))
                print("-" * 40)
                
            except Exception as e:
                negotiation_entry['error'] = str(e)
                print(f"❌ LLM Negotiation Error: {e}")
                response = {'error': str(e), 'fallback': 'wait_action'}
            
            # Restore original methods
            game_engine.central_negotiator._create_negotiation_system_prompt = original_system_prompt
            game_engine.central_negotiator._create_conflict_description = original_conflict_desc
            
            scenario_data['negotiations'].append(negotiation_entry)
            self.negotiation_log['negotiations'].append(negotiation_entry)
            
            return response
        
        game_engine.central_negotiator.negotiate_path_conflict = capture_negotiate
        
        # Run simulation for specified turns
        print(f"\n🚀 Running simulation for up to {max_turns} turns...")
        print("Press Enter to start, then Enter for each turn...")
        input()
        
        conflicts_detected = 0
        turn_completed = 0
        for turn_completed in range(max_turns):
            print(f"\n=== TURN {turn_completed + 1} ===")
            
            try:
                # Run one simulation step
                continue_sim = game_engine.run_simulation_step()
                
                # Count conflicts in this turn
                turn_conflicts = len([n for n in scenario_data['negotiations'] 
                                    if n.get('turn') == game_engine.current_turn])
                if turn_conflicts > 0:
                    conflicts_detected += turn_conflicts
                    print(f"🔥 Conflicts this turn: {turn_conflicts}")
                
                if not continue_sim:
                    print("🏁 Simulation completed!")
                    break
                    
            except Exception as e:
                print(f"❌ Simulation error: {e}")
                break
            
            input("Press Enter for next turn...")
        
        scenario_data['total_conflicts'] = conflicts_detected
        scenario_data['turns_completed'] = turn_completed + 1
        
        self.negotiation_log['conflict_scenarios'].append(scenario_data)
        
        print(f"\n📊 SCENARIO RESULTS:")
        print(f"   Total conflicts detected: {conflicts_detected}")
        print(f"   Total negotiations: {len(scenario_data['negotiations'])}")
        print(f"   Turns completed: {scenario_data['turns_completed']}")
        
        return scenario_data
    
    def save_negotiation_log(self, filename: str | None = None) -> str:
        """Save all negotiation data to logs folder"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"negotiation_test_{timestamp}.json"
        
        # Ensure logs directory exists
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        filepath = os.path.join(logs_dir, filename)
        
        # Add summary statistics
        self.negotiation_log['summary'] = {
            'total_scenarios': len(self.negotiation_log['conflict_scenarios']),
            'total_negotiations': len(self.negotiation_log['negotiations']),
            'total_conflicts': sum(s.get('total_conflicts', 0) for s in self.negotiation_log['conflict_scenarios']),
            'success_rate': len([n for n in self.negotiation_log['negotiations'] if 'error' not in n]) / max(len(self.negotiation_log['negotiations']), 1)
        }
        
        with open(filepath, 'w') as f:
            json.dump(self.negotiation_log, f, indent=2)
        
        print(f"\n💾 Negotiation log saved to: {filepath}")
        print(f"📊 Summary: {self.negotiation_log['summary']}")
        
        return filepath

def main():
    """Run comprehensive negotiation tests"""
    print("🤖 LLM NEGOTIATION TEST SUITE")
    print("=" * 60)
    print("This test creates scenarios guaranteed to cause conflicts")
    print("and logs all LLM negotiations for analysis.")
    print()
    
    tester = NegotiationTester()
    
    # Test scenarios in order of complexity
    scenarios = [
        ("single_corridor", "Two agents must cross paths in narrow corridor"),
        ("bottleneck_chamber", "Three agents must pass through single bottleneck"),
        ("triple_intersection", "Three agents converge at single intersection")
    ]
    
    print("Available test scenarios:")
    for i, (scenario, description) in enumerate(scenarios, 1):
        print(f"  {i}. {scenario}: {description}")
    print()
    
    choice = input("Enter scenario number (1-3) or 'all' for all scenarios: ").strip()
    
    if choice.lower() == 'all':
        print("🚀 Running ALL negotiation scenarios...")
        for scenario, _ in scenarios:
            tester.run_forced_conflict_test(scenario)
    else:
        try:
            scenario_idx = int(choice) - 1
            if 0 <= scenario_idx < len(scenarios):
                scenario, _ = scenarios[scenario_idx]
                print(f"🚀 Running {scenario} negotiation test...")
                tester.run_forced_conflict_test(scenario)
            else:
                print("❌ Invalid choice!")
                return
        except ValueError:
            print("❌ Invalid choice!")
            return
    
    # Save all negotiation data
    log_file = tester.save_negotiation_log()
    
    print(f"\n🎉 NEGOTIATION TESTING COMPLETED!")
    print(f"📁 All data saved to: {log_file}")
    print("🔍 Analyze the logs to see how LLMs handle conflict resolution!")

if __name__ == "__main__":
    main()
