#!/usr/bin/env python3
"""
LLM Negotiation Test - Forces guaranteed conflicts and uses unified logging

This simplified version relies on GameEngine's unified logging system
to capture all simulation and HMAS-2 negotiation data in a single JSON file.
"""

import sys
import os
from datetime import datetime
from typing import Dict, Any
from colorama import init, Fore, Style

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.simulation.game_engine import GameEngine
from src.map_generator import WarehouseMap
from src.map_generator.layout_selector import get_layout_for_game

# Initialize colorama
init(autoreset=True)


class NegotiationTester:
    """Test class for forcing conflicts and running negotiation simulations"""
    
    def __init__(self):
        print("🤖 Negotiation Tester initialized")
        print("📝 Using unified logging system (single JSON per run)")
    
    def load_layout_for_negotiation_test(self) -> WarehouseMap | None:
        """Load a layout for testing (same as main.py)"""
        print(f"\n{Fore.CYAN}Loading Warehouse Layout...{Style.RESET_ALL}")
        layout = get_layout_for_game(allow_selection=True)
        
        if layout is None:
            print(f"{Fore.RED}No layout selected. Test cancelled.{Style.RESET_ALL}")
            return None
        
        try:
            # Load the layout into the warehouse map
            warehouse = WarehouseMap.from_layout(layout)
            
            print(f"✅ Layout loaded: {layout.get('name', 'Untitled')}")
            print(f"   Dimensions: {layout['dimensions']['width']}x{layout['dimensions']['height']}")
            print(f"   Agents: {len(layout['agents'])}, Boxes: {len(layout['boxes'])}, Targets: {len(layout['targets'])}")
            
            return warehouse
            
        except Exception as e:
            print(f"{Fore.RED}Error loading layout: {e}{Style.RESET_ALL}")
            return None
    
    def run_negotiation_test(self, max_turns: int = 100, enable_spatial_hints: bool = True) -> Dict[str, Any]:
        """Run a negotiation test with user-selected layout"""
        
        hints_status = "WITH SPATIAL HINTS" if enable_spatial_hints else "BASELINE (NO HINTS)"
        print("=" * 60)
        print(f"📋 NEGOTIATION TEST MODE: {hints_status}")
        print("=" * 60)
        
        # Load layout from user selection
        warehouse = self.load_layout_for_negotiation_test()
        
        if warehouse is None:
            return {'cancelled': True}
        
        # Display the scenario
        print("\n🗺️  LOADED LAYOUT MAP:")
        print("=" * 50)
        print(warehouse.display())
        
        print(f"\n📋 Initial Scenario Setup:")
        print(f"   Map size: {warehouse.width} x {warehouse.height}")
        print(f"   Agent positions: {dict(warehouse.agents)}")
        print(f"   Box positions: {dict(warehouse.boxes)}")
        print(f"   Target positions: {dict(warehouse.targets)}")
        print(f"   Agent goals: {dict(warehouse.agent_goals)}")
        
        # Show expected agent paths
        print(f"\n📍 Agent Movement Plan:")
        for agent_id, start_pos in warehouse.agents.items():
            goal_id = warehouse.agent_goals.get(agent_id)
            if goal_id is not None:
                # Show the correct two-phase plan
                if goal_id in warehouse.boxes and goal_id in warehouse.targets:
                    box_pos = warehouse.boxes[goal_id]
                    target_pos = warehouse.targets[goal_id]
                    print(f"   Agent {agent_id}: {start_pos} → 📦{box_pos} → 🎯{target_pos}")
                else:
                    print(f"   ⚠️  Agent {agent_id}: Missing box or target for goal {goal_id}")
            else:
                print(f"   ⚠️  Agent {agent_id}: No goal assigned")
        
        print("=" * 50)
        
        # Confirm before proceeding
        proceed = input("\n🤔 Does this scenario look good? Press Enter to proceed or 'q' to quit: ").strip().lower()
        if proceed == 'q':
            print("Test cancelled by user.")
            return {'cancelled': True}
        
        # Create game engine with this specific map
        game_engine = GameEngine(width=warehouse.width, height=warehouse.height, num_agents=len(warehouse.agents))
        game_engine.warehouse_map = warehouse
        game_engine.agents = {}
        
        # Configure spatial hints on the central negotiator
        game_engine.central_negotiator.set_spatial_hints(enable_spatial_hints)
        
        # Initialize agents with the pre-set positions
        from src.agents import RobotAgent
        for agent_id, position in warehouse.agents.items():
            agent = RobotAgent(agent_id, position)
            
            # CRITICAL FIX: Set the box as the initial target, not the delivery target!
            if agent_id in warehouse.agent_goals:
                goal_id = warehouse.agent_goals[agent_id]
                
                # Phase 1: Agent should first go to their assigned box
                if goal_id in warehouse.boxes:
                    box_pos = warehouse.boxes[goal_id]
                    agent.set_target(box_pos)  # Go to box first!
                    
                    # Store the final delivery target for later (after box pickup)
                    if goal_id in warehouse.targets:
                        delivery_target = warehouse.targets[goal_id]
                        setattr(agent, 'delivery_target', delivery_target)
                else:
                    print(f"   ⚠️  WARNING: Agent {agent_id} assigned to non-existent box {goal_id}")
            else:
                print(f"   ⚠️  WARNING: Agent {agent_id} has no assigned goal!")
                
            game_engine.agents[agent_id] = agent
        
        # CRITICAL: Plan initial paths for all agents (ignoring other agents for initial planning)
        map_state = warehouse.get_state_dict()
        
        # Create a temporary pathfinder for initial planning
        from src.navigation import SimplePathfinder
        temp_pathfinder = SimplePathfinder(warehouse.width, warehouse.height)
        
        # Get wall positions
        wall_positions = set()
        for y, row in enumerate(map_state['grid']):
            for x, cell in enumerate(row):
                if cell == '#':
                    wall_positions.add((x, y))
        
        for agent_id, agent in game_engine.agents.items():
            if agent.target_position:
                # Plan path WITHOUT avoiding other agents (for initial planning)
                path = temp_pathfinder.find_path_with_obstacles(
                    start=agent.position,
                    goal=agent.target_position,
                    walls=wall_positions,
                    agent_positions={},  # Empty - don't avoid other agents initially
                    exclude_agent=agent_id
                )
                
                # Set the planned path directly
                agent.planned_path = path
                
                if len(path) == 0:
                    print(f"   ⚠️  WARNING: Agent {agent_id} has NO PATH to target!")
            else:
                print(f"   ⚠️  WARNING: Agent {agent_id} has NO TARGET!")
        print("🧠 Path planning completed.")
        
        # Initialize simulation (this also initializes unified logging)
        game_engine.initialize_simulation()
        
        # Run simulation for specified turns
        print(f"\n🚀 Running simulation...")
        print("Press Enter to start, then Enter for each turn...")
        input()
        
        conflicts_detected = 0
        turn_completed = 0
        
        # Run simulation until completion OR max turns
        while turn_completed < max_turns:
            print(f"\n{'='*60}")
            print(f"=== TURN {turn_completed + 1} ===")
            print(f"{'='*60}")
            
            try:
                # Run one simulation step (logging handled by GameEngine)
                continue_sim = game_engine.run_simulation_step()
                
                # IMPORTANT: Break if simulation signals completion
                if not continue_sim:
                    print("🏁 Simulation completed!")
                    break
                    
            except Exception as e:
                print(f"❌ Simulation error: {e}")
                import traceback
                traceback.print_exc()
                break
            
            turn_completed += 1
            input("Press Enter for next turn...")
        
        # Save unified log
        log_path = game_engine.save_simulation_log()
        
        # Return summary
        return {
            'cancelled': False,
            'turns_completed': turn_completed,
            'log_file': log_path,
            'spatial_hints_enabled': enable_spatial_hints
        }
    
    def preview_layouts(self):
        """Show a quick preview of available layouts"""
        from src.map_generator.layout_manager import LayoutManager
        
        print("\n🔍 AVAILABLE LAYOUTS:")
        print("=" * 70)
        
        manager = LayoutManager()
        layouts_info = manager.list_layout_details()
        print(layouts_info)
        print("=" * 70)


def main():
    """Run negotiation tests with user-selected layouts"""
    print("🤖 LLM NEGOTIATION TEST SUITE")
    print("=" * 60)
    print("Test LLM negotiations with your chosen warehouse layout.")
    print("📝 All data logged to a single unified JSON file per run.")
    print()
    
    tester = NegotiationTester()
    
    # Test mode selection
    print("📊 TEST MODES:")
    print("1. Run negotiation test (with spatial hints)")
    print("2. Run negotiation test (baseline - no spatial hints)")
    print("3. View available layouts")
    
    try:
        mode = input("\nSelect mode (1-3): ").strip()
        
        if mode == "3":
            tester.preview_layouts()
            return
        
        # Run negotiation test
        if mode == "1":
            # With spatial hints
            print("\n🎯 Starting negotiation test WITH SPATIAL HINTS...")
            result = tester.run_negotiation_test(enable_spatial_hints=True)
        elif mode == "2": 
            # Baseline without hints
            print("\n🎯 Starting negotiation test BASELINE (NO SPATIAL HINTS)...")
            result = tester.run_negotiation_test(enable_spatial_hints=False)
        else:
            print("❌ Invalid mode choice")
            return
        
        # Show results summary
        if not result.get('cancelled'):
            print(f"\n{'='*60}")
            print(f"📊 TEST COMPLETE")
            print(f"{'='*60}")
            print(f"   Turns completed: {result.get('turns_completed', 0)}")
            print(f"   Log file: {result.get('log_file', 'N/A')}")
            print(f"   Spatial hints: {'Enabled' if result.get('spatial_hints_enabled') else 'Disabled'}")
        else:
            print("Test was cancelled - no results to save.")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Test interrupted!")
        print("💾 Data should be auto-saved by unified logger...")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
