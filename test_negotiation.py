#!/usr/bin/env python3
"""
LLM Negotiation Test - Forces guaranteed conflicts and logs all negotiations
"""

import sys
import os
import json
import time
import signal
import atexit
from datetime import datetime
from typing import Dict, List, Any

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.simulation.game_engine import GameEngine
from src.map_generator import WarehouseMap

# Global variable to store the tester instance for signal handling
_global_tester = None

def signal_handler(signum, frame):
    """Handle Ctrl+C and other interruption signals"""
    print(f"\n\n🛑 PROCESS INTERRUPTED! (Signal {signum})")
    print("💾 Saving negotiation data before exit...")
    
    if _global_tester is not None:
        try:
            log_file = _global_tester.save_negotiation_log()
            print(f"✅ Emergency save completed: {log_file}")
        except Exception as e:
            print(f"❌ Error during emergency save: {e}")
    else:
        print("❌ No negotiation data to save")
    
    print("👋 Exiting gracefully...")
    sys.exit(0)

def cleanup_on_exit():
    """Cleanup function called on normal exit"""
    if _global_tester is not None and hasattr(_global_tester, '_unsaved_data'):
        if _global_tester._unsaved_data:
            print("💾 Auto-saving negotiation data on exit...")
            _global_tester.save_negotiation_log()

# Register signal handlers and exit cleanup
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination request
atexit.register(cleanup_on_exit)               # Normal exit cleanup

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
        self._unsaved_data = False  # Track if we have unsaved data
        self._auto_save_count = 0   # Counter for auto-saves
        
        # Set global reference for signal handling
        global _global_tester
        _global_tester = self
        
        print("🛡️  Emergency save protection enabled (Ctrl+C safe)")
    
    def _mark_data_changed(self):
        """Mark that we have unsaved data"""
        self._unsaved_data = True
    
    def _auto_save_check(self):
        """Auto-save every few negotiations to prevent data loss"""
        self._auto_save_count += 1
        if self._auto_save_count % 3 == 0:  # Auto-save every 3 negotiations
            print("💾 Auto-saving progress...")
            self.save_negotiation_log(auto_save=True)
            self._unsaved_data = False
    
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
            
        elif scenario_type == "s_shaped_corridor":
            # Create S-shaped corridor exactly as user specified with strategic wiggle room
            width, height = 10, 8
            warehouse = WarehouseMap(width, height)
            
            # Start with all walls
            warehouse.grid.fill('#')
            
            # Implement user's exact design pattern:
            # Row 0: # # # # # # # # # #
            # Row 1: # . . . . . . . # #
            # Row 2: # # # # # # # . . #  
            # Row 3: # # # # # # # . . #
            # Row 4: # # . . . . . . # #
            # Row 5: # . . # # # # # # #
            # Row 6: # # . . . . . . . #
            # Row 7: # # # # # # # # # #
            
            # Row 1: Top horizontal corridor
            for x in range(1, 8):
                warehouse.grid[1, x] = '.'
            
            # Row 2: Right area with wiggle space
            warehouse.grid[2, 7] = '.'
            warehouse.grid[2, 8] = '.'
            
            # Row 3: Right area with wiggle space  
            warehouse.grid[3, 7] = '.'
            warehouse.grid[3, 8] = '.'
            
            # Row 4: Middle horizontal corridor with wiggle space at end
            for x in range(2, 8):
                warehouse.grid[4, x] = '.'
            # Note: Row 4 ends at column 7 to match user's pattern (# #)
            
            # Row 5: Left wiggle space and connection
            warehouse.grid[5, 1] = '.'
            warehouse.grid[5, 2] = '.'
            
            # Row 6: Bottom horizontal corridor  
            for x in range(2, 9):
                warehouse.grid[6, x] = '.'
            
            # Place agents to GUARANTEE conflict requiring negotiation
            # Agent 0: Top area, needs to traverse S to bottom-right  
            # Agent 1: Bottom area, needs to traverse S to top-left
            # Their paths WILL block each other, requiring LLM negotiation to resolve
            warehouse.agents = {0: (4, 1), 1: (5, 6)}
            
            # Place boxes near agents
            warehouse.boxes = {0: (3, 1), 1: (6, 6)}
            
            # Targets require full S-traversal creating GUARANTEED negotiation scenario
            warehouse.targets = {0: (8, 6), 1: (2, 1)}
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
            
        elif scenario_type == "validation_stress_test":
            # New scenario designed to trigger agent validation rejections
            width, height = 6, 4
            warehouse = WarehouseMap(width, height)
            
            # Fill with walls
            warehouse.grid.fill('#')
            
            # Create a narrow L-shaped corridor with hazards
            # Horizontal part
            for x in range(1, 5):
                warehouse.grid[1, x] = '.'
            # Vertical part (short)
            warehouse.grid[2, 4] = '.'
            warehouse.grid[3, 4] = '.'  # Dead end to trigger validation issues
            
            # Place agents in challenging positions
            warehouse.agents = {0: (1, 1), 1: (4, 1)}
            
            # Place boxes in positions that might trigger validation failures
            warehouse.boxes = {0: (2, 1), 1: (3, 1)}
            warehouse.targets = {0: (4, 3), 1: (1, 1)}  # Crossed, one unreachable
            warehouse.agent_goals = {0: 0, 1: 1}
            
        else:
            # Default fallback
            warehouse = WarehouseMap(8, 3)
            warehouse.generate_map(num_agents=2, layout_type="extreme")
        
        return warehouse
    
    def run_forced_conflict_test(self, scenario_type: str, max_turns: int = 20, enable_spatial_hints: bool = True) -> Dict[str, Any]:
        """Run a single conflict scenario and capture all negotiations"""
        
        print(f"\n🎯 TESTING SCENARIO: {scenario_type.upper()}")
        hints_status = "WITH SPATIAL HINTS" if enable_spatial_hints else "BASELINE (NO HINTS)"
        print(f"📊 Mode: {hints_status}")
        print("=" * 60)
        
        # Create the forced conflict map
        warehouse = self.create_forced_conflict_map(scenario_type)
        
        # Display the scenario IMMEDIATELY after creation
        print("🗺️  GENERATED SCENARIO MAP:")
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
        
        print(f"\n⚔️  Expected Conflicts: GUARANTEED (agents must navigate through same bottlenecks)")
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
        from src.agents import RobotAgent  # Use RobotAgent instead of Agent
        for agent_id, position in warehouse.agents.items():
            agent = RobotAgent(agent_id, position)
            
            # CRITICAL FIX: Set the box as the initial target, not the delivery target!
            if agent_id in warehouse.agent_goals:
                goal_id = warehouse.agent_goals[agent_id]
                
                # Phase 1: Agent should first go to their assigned box
                if goal_id in warehouse.boxes:
                    box_pos = warehouse.boxes[goal_id]
                    agent.set_target(box_pos)  # Go to box first!
                    print(f"   Agent {agent_id}: Initial target set to BOX at {box_pos}")
                    
                    # Store the final delivery target for later (after box pickup)
                    if goal_id in warehouse.targets:
                        delivery_target = warehouse.targets[goal_id]
                        setattr(agent, 'delivery_target', delivery_target)
                        print(f"   Agent {agent_id}: Delivery target will be {delivery_target}")
                else:
                    print(f"   ⚠️  WARNING: Agent {agent_id} assigned to non-existent box {goal_id}")
            else:
                print(f"   ⚠️  WARNING: Agent {agent_id} has no assigned goal!")
                
            game_engine.agents[agent_id] = agent
        
        # CRITICAL: Plan initial paths for all agents (ignoring other agents for initial planning)
        print("🧠 Planning initial paths for all agents...")
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
                
                # Show what the agent is planning to do
                if hasattr(agent, 'delivery_target'):
                    print(f"   Agent {agent_id}: Phase 1 - Go to BOX at {agent.target_position} ({len(path)} steps)")
                    if len(path) <= 10:  # Show short paths
                        print(f"      Path to box: {path}")
                else:
                    print(f"   Agent {agent_id}: Planned path with {len(path)} steps")
                    if len(path) <= 10:  # Show short paths
                        print(f"      Path: {path}")
                
                if len(path) == 0:
                    print(f"   ⚠️  WARNING: Agent {agent_id} has NO PATH to target!")
            else:
                print(f"   ⚠️  WARNING: Agent {agent_id} has NO TARGET!")
        print("🧠 Path planning completed.")
        
        # Store scenario info
        scenario_data = {
            'type': scenario_type,
            'map_size': (warehouse.width, warehouse.height),
            'agents': dict(warehouse.agents),
            'targets': dict(warehouse.targets),
            'expected_conflicts': 'GUARANTEED',
            'negotiations': []
        }
        
        # Monkey-patch the negotiation method to capture HMAS-2 data
        original_negotiate = game_engine.central_negotiator.negotiate_path_conflict
        original_system_prompt = game_engine.central_negotiator._create_negotiation_system_prompt
        original_conflict_desc = game_engine.central_negotiator._create_conflict_description
        
        # Also patch agent validation methods
        agent_original_execute = {}
        for agent_id, agent in game_engine.agents.items():
            agent_original_execute[agent_id] = agent.execute_negotiated_action
        
        # Initialize negotiation capture flag using setattr to avoid linter warnings
        setattr(game_engine, '_negotiation_captured', False)
        
        def capture_negotiate(conflict_data):
            # Prevent double-negotiation with flag check
            if getattr(game_engine, '_negotiation_captured', False):
                print("🔄 Skipping duplicate negotiation call - already captured this turn")
                return {
                    'resolution': 'already_handled',
                    'agent_actions': {},
                    'reasoning': 'Negotiation already captured by HMAS-2 test'
                }
            
            # Set flag to prevent re-negotiation
            setattr(game_engine, '_negotiation_captured', True)
            
            print(f"\n🤖 CONFLICT DETECTED! Initiating HMAS-2 Negotiation...")
            print(f"   🏢 CENTRAL LLM: Analyzing conflict between agents {[a.get('id') for a in conflict_data.get('agents', [])]}")
            print(f"   📍 Conflict points: {conflict_data.get('conflict_points', [])}")
            
            # Get the actual models being used (no more hardcoding!)
            central_model = getattr(game_engine.central_negotiator, 'model', 'unknown')
            
            print(f"🔍 DEBUG: Central Negotiator using model: {central_model}")
            
            # Enhanced HMAS-2 negotiation entry structure
            negotiation_entry = {
                'turn': conflict_data.get('turn', 0),
                'timestamp': datetime.now().isoformat(),
                'hmas2_stages': {
                    'central_negotiation': {
                        'system_prompt': None,
                        'user_prompt': None,
                        'llm_response': None,
                        'model_used': central_model  # Dynamic model detection!
                    },
                    'agent_validations': {},
                    'final_actions': {},
                    'validation_overrides': {}
                },
                'conflict_data': conflict_data
            }
            
            # Capture prompts
            def capture_system_prompt():
                prompt = original_system_prompt()
                negotiation_entry['hmas2_stages']['central_negotiation']['system_prompt'] = prompt
                print(f"\n📋 CENTRAL LLM SYSTEM PROMPT:")
                print("=" * 50)
                print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
                print("=" * 50)
                return prompt
            
            def capture_user_prompt(conflict_data):
                prompt = original_conflict_desc(conflict_data)
                negotiation_entry['hmas2_stages']['central_negotiation']['user_prompt'] = prompt
                print(f"\n📝 CENTRAL LLM USER PROMPT:")
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
                negotiation_entry['hmas2_stages']['central_negotiation']['llm_response'] = response
                
                print(f"\n💬 CENTRAL LLM RESPONSE:")
                print("-" * 40)
                if isinstance(response, dict):
                    print(json.dumps(response, indent=2))
                else:
                    print(str(response))
                print("-" * 40)
                
            except Exception as e:
                negotiation_entry['hmas2_stages']['central_negotiation']['error'] = str(e)
                print(f"❌ Central LLM Negotiation Error: {e}")
                response = {'error': str(e), 'fallback': 'wait_action'}
            
            # Restore original methods
            game_engine.central_negotiator._create_negotiation_system_prompt = original_system_prompt
            game_engine.central_negotiator._create_conflict_description = original_conflict_desc
            
            # Debug output to see what keys exist in the response
            print(f"\n🔍 DEBUG: Response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
            print(f"🔍 DEBUG: Response type: {type(response)}")
            
            # STAGE 2: Capture Agent LLM Validations (if response contains agent actions)
            if isinstance(response, dict) and ('actions' in response or 'agent_actions' in response):
                print(f"\n🤖 STAGE 2: AGENT LLM VALIDATIONS")
                print("=" * 60)
                
                # Handle both 'actions' and 'agent_actions' keys
                actions = response.get('actions') or response.get('agent_actions', {})
                print(f"🔍 DEBUG: Actions found: {actions}")
                print(f"🔍 DEBUG: Actions type: {type(actions)}")
                
                if isinstance(actions, dict):
                    for agent_id, action_data in actions.items():
                        # Convert string agent_id to int if needed for lookup - FIXED TYPE HANDLING
                        agent_id_key = int(agent_id) if isinstance(agent_id, str) and agent_id.isdigit() else agent_id
                        
                        print(f"🔍 DEBUG: Processing agent_id: {agent_id} (type: {type(agent_id)}) -> {agent_id_key} (type: {type(agent_id_key)})")
                        print(f"🔍 DEBUG: Looking for agent {agent_id} (converted to {agent_id_key}) in agents: {list(game_engine.agents.keys())}")
                        
                        if agent_id_key in game_engine.agents:
                            agent = game_engine.agents[agent_id_key]
                            
                            print(f"\n🔍 Agent {agent_id_key}: Validating action {action_data}")
                            
                            # Get the actual agent model being used (no hardcoding!)
                            agent_model = getattr(agent.validator, 'model', 'unknown') if hasattr(agent, 'validator') else 'unknown'
                            print(f"🔍 DEBUG: Agent {agent_id_key} validator using model: {agent_model}")
                            
                            # Capture the validation process - USE THE CONVERTED agent_id_key
                            validation_entry = {
                                'agent_id': agent_id_key,  # Use converted ID
                                'proposed_action': action_data,
                                'validation_model': agent_model,  # Dynamic model detection!
                                'validation_result': None,
                                'alternative_suggested': None,
                                'final_action_executed': None
                            }
                            
                            try:
                                # Get the map state for validation
                                map_state = game_engine.warehouse_map.get_state_dict()
                                
                                # Debug the map state before passing to validator
                                print(f"🔍 DEBUG: Map state before validation: {map_state.get('agents', {})}")
                                print(f"🔍 DEBUG: Map state agent types: {[(k, type(k)) for k in map_state.get('agents', {}).keys()]}")
                                
                                # Call the agent's validator - USE THE CONVERTED agent_id_key
                                validation_result = agent.validator.validate_negotiated_action(
                                    agent_id_key, action_data, map_state  # Use converted ID here too
                                )
                                
                                validation_entry['validation_result'] = validation_result
                                
                                print(f"   📋 Validation Result: {validation_result.get('valid', 'Unknown')}")
                                if 'reason' in validation_result:
                                    print(f"   💭 Reason: {validation_result['reason']}")
                                
                                if not validation_result.get('valid', False):
                                    print(f"   ❌ Agent {agent_id} REJECTED the central negotiation!")
                                    
                                    # COUNT ALL REJECTIONS, not just alternatives
                                    negotiation_entry['hmas2_stages']['validation_overrides'][agent_id] = {
                                        'rejected_central_action': action_data,
                                        'rejection_reason': validation_result.get('reason', 'unknown'),
                                        'agent_alternative': None
                                    }
                                    
                                    if 'alternative' in validation_result:
                                        alternative = validation_result['alternative']
                                        validation_entry['alternative_suggested'] = alternative
                                        print(f"   🔄 Agent {agent_id} suggests alternative: {alternative}")
                                        
                                        # Update the override entry with the alternative
                                        negotiation_entry['hmas2_stages']['validation_overrides'][agent_id]['agent_alternative'] = alternative
                                        final_action = alternative
                                    else:
                                        print(f"   ⏸️  Agent {agent_id} suggests WAIT/NO_ACTION")
                                        final_action = {'action': 'wait', 'reason': 'validation_failed'}
                                        
                                else:
                                    print(f"   ✅ Agent {agent_id} APPROVED the central negotiation!")
                                    final_action = action_data
                                
                                validation_entry['final_action_executed'] = final_action
                                negotiation_entry['hmas2_stages']['agent_validations'][agent_id] = validation_entry
                                negotiation_entry['hmas2_stages']['final_actions'][agent_id] = final_action
                                
                            except Exception as validation_error:
                                validation_entry['error'] = str(validation_error)
                                print(f"   ❌ Validation Error for Agent {agent_id}: {validation_error}")
                                
                                # Fallback to wait action
                                fallback_action = {'action': 'wait', 'reason': 'validation_error'}
                                validation_entry['final_action_executed'] = fallback_action
                                negotiation_entry['hmas2_stages']['agent_validations'][agent_id] = validation_entry
                                negotiation_entry['hmas2_stages']['final_actions'][agent_id] = fallback_action
                        else:
                            print(f"⚠️  Agent {agent_id} (key: {agent_id_key}) not found in game_engine.agents: {list(game_engine.agents.keys())}")
                
                print("\n🏁 HMAS-2 NEGOTIATION COMPLETE:")
                print(f"   Central LLM: {'✅' if 'error' not in negotiation_entry['hmas2_stages']['central_negotiation'] else '❌'}")
                print(f"   Agent Validations: {len(negotiation_entry['hmas2_stages']['agent_validations'])} agents processed")
                print(f"   Validation Overrides: {len(negotiation_entry['hmas2_stages']['validation_overrides'])} rejections")
                
                # Flag all agents as HMAS-2 validated to skip redundant game engine validation
                print(f"\n🏷️  Flagging agents as HMAS-2 validated:")
                for agent_id_str, validation_data in negotiation_entry['hmas2_stages']['agent_validations'].items():
                    agent_id_int = int(agent_id_str) if isinstance(agent_id_str, str) and agent_id_str.isdigit() else agent_id_str
                    if agent_id_int in game_engine.agents:
                        setattr(game_engine.agents[agent_id_int], '_hmas2_validated', True)
                        print(f"   🏷️  Agent {agent_id_int}: Flagged as HMAS-2 validated")
            
            else:
                print(f"\n⚠️  Central LLM response format doesn't contain agent actions")
                print(f"🔧 FALLBACK: Creating mock actions for HMAS-2 validation demo")
                
                # Create mock actions based on conflict data for validation testing
                if 'agents' in conflict_data and len(conflict_data['agents']) > 0:
                    print(f"\n🤖 STAGE 2: AGENT LLM VALIDATIONS (FALLBACK MODE)")
                    print("=" * 60)
                    
                    actions = {}
                    for agent_info in conflict_data['agents']:
                        agent_id = agent_info.get('id')
                        if agent_id in game_engine.agents:
                            # Create a mock action based on the agent's planned path
                            agent = game_engine.agents[agent_id]
                            if hasattr(agent, 'planned_path') and len(agent.planned_path) > 1:
                                next_position = agent.planned_path[1]  # Next step in path
                                actions[agent_id] = {
                                    'action': 'move',
                                    'target': next_position,
                                    'reason': 'mock_action_for_validation_test'
                                }
                            else:
                                actions[agent_id] = {
                                    'action': 'wait',
                                    'reason': 'no_valid_path_available'
                                }
                    
                    print(f"🔧 Created mock actions for validation: {actions}")
                else:
                    actions = {}
                
                # Process the actions (either from LLM or mock) for validation
                if isinstance(actions, dict) and len(actions) > 0:
                    for agent_id, action_data in actions.items():
                        # Ensure type consistency for mock actions too
                        agent_id_key = int(agent_id) if isinstance(agent_id, str) and agent_id.isdigit() else agent_id
                        
                        if agent_id_key in game_engine.agents:
                            agent = game_engine.agents[agent_id_key]
                            
                            print(f"\n🔍 Agent {agent_id_key}: Validating action {action_data}")
                            
                            # Get the actual agent model being used (no hardcoding!)
                            agent_model = getattr(agent.validator, 'model', 'unknown') if hasattr(agent, 'validator') else 'unknown'
                            print(f"🔍 DEBUG: Agent {agent_id_key} validator using model: {agent_model}")
                            
                            # Capture the validation process
                            validation_entry = {
                                'agent_id': agent_id_key,
                                'proposed_action': action_data,
                                'validation_model': agent_model,  # Dynamic model detection!
                                'validation_result': None,
                                'alternative_suggested': None,
                                'final_action_executed': None
                            }
                            
                            try:
                                # Get the map state for validation
                                map_state = game_engine.warehouse_map.get_state_dict()
                                
                                # Call the agent's validator - use converted agent_id_key
                                validation_result = agent.validator.validate_negotiated_action(
                                    agent_id_key, action_data, map_state
                                )
                                
                                validation_entry['validation_result'] = validation_result
                                
                                print(f"   📋 Validation Result: {validation_result.get('valid', 'Unknown')}")
                                if 'reason' in validation_result:
                                    print(f"   💭 Reason: {validation_result['reason']}")
                                
                                if not validation_result.get('valid', False):
                                    print(f"   ❌ Agent {agent_id_key} REJECTED the central negotiation!")
                                    
                                    # COUNT ALL REJECTIONS, not just alternatives
                                    negotiation_entry['hmas2_stages']['validation_overrides'][agent_id_key] = {
                                        'rejected_central_action': action_data,
                                        'rejection_reason': validation_result.get('reason', 'unknown'),
                                        'agent_alternative': None
                                    }
                                    
                                    if 'alternative' in validation_result:
                                        alternative = validation_result['alternative']
                                        validation_entry['alternative_suggested'] = alternative
                                        print(f"   🔄 Agent {agent_id_key} suggests alternative: {alternative}")
                                        
                                        # Update the override entry with the alternative
                                        negotiation_entry['hmas2_stages']['validation_overrides'][agent_id_key]['agent_alternative'] = alternative
                                        final_action = alternative
                                    else:
                                        print(f"   ⏸️  Agent {agent_id_key} suggests WAIT/NO_ACTION")
                                        final_action = {'action': 'wait', 'reason': 'validation_failed'}
                                        
                                else:
                                    print(f"   ✅ Agent {agent_id_key} APPROVED the central negotiation!")
                                    final_action = action_data
                                
                                validation_entry['final_action_executed'] = final_action
                                negotiation_entry['hmas2_stages']['agent_validations'][agent_id_key] = validation_entry
                                negotiation_entry['hmas2_stages']['final_actions'][agent_id_key] = final_action
                                
                            except Exception as validation_error:
                                validation_entry['error'] = str(validation_error)
                                print(f"   ❌ Validation Error for Agent {agent_id_key}: {validation_error}")
                                
                                # Fallback to wait action
                                fallback_action = {'action': 'wait', 'reason': 'validation_error'}
                                validation_entry['final_action_executed'] = fallback_action
                                negotiation_entry['hmas2_stages']['agent_validations'][agent_id_key] = validation_entry
                                negotiation_entry['hmas2_stages']['final_actions'][agent_id_key] = fallback_action
                
                    print("\n🏁 HMAS-2 NEGOTIATION COMPLETE:")
                    print(f"   Central LLM: {'✅' if 'error' not in negotiation_entry['hmas2_stages']['central_negotiation'] else '❌'}")
                    print(f"   Agent Validations: {len(negotiation_entry['hmas2_stages']['agent_validations'])} agents processed")
                    print(f"   Validation Overrides: {len(negotiation_entry['hmas2_stages']['validation_overrides'])} rejections")
                else:
                    print(f"\n⚠️  No actions available for validation")
            
            scenario_data['negotiations'].append(negotiation_entry)
            self.negotiation_log['negotiations'].append(negotiation_entry)
            
            # Mark data as changed and check for auto-save
            self._mark_data_changed()
            self._auto_save_check()
            
            print(f"\n🏁 HMAS-2 CAPTURE COMPLETE - Returning to game engine...")
            
            # Reset flag after processing to allow next turn's negotiation
            setattr(game_engine, '_negotiation_captured', False)
            
            # CRITICAL FIX: Ensure response has correct format for game engine
            if isinstance(response, dict):
                # If response has 'actions' but not 'agent_actions', fix it
                if 'actions' in response and 'agent_actions' not in response:
                    response['agent_actions'] = response['actions']
                    print(f"🔧 Fixed response format: moved 'actions' to 'agent_actions'")
                
                # Ensure agent_actions exists (fallback)
                if 'agent_actions' not in response:
                    response['agent_actions'] = {}
                    print(f"🔧 Added missing 'agent_actions' to response")
            
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
            
            # Reset negotiation flag at start of each turn for clean state
            if hasattr(game_engine, '_negotiation_captured'):
                setattr(game_engine, '_negotiation_captured', False)
            
            # Reset all agent HMAS-2 validation flags for new turn
            for agent_id, agent in game_engine.agents.items():
                if hasattr(agent, '_hmas2_validated'):
                    setattr(agent, '_hmas2_validated', False)
                    print(f"🔄 Agent {agent_id}: Reset HMAS-2 validation flag for new turn")
            
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
        self._mark_data_changed()  # Mark that we have new scenario data
        
        print(f"\n📊 SCENARIO RESULTS:")
        print(f"   Total conflicts detected: {conflicts_detected}")
        print(f"   Total negotiations: {len(scenario_data['negotiations'])}")
        print(f"   Turns completed: {scenario_data['turns_completed']}")
        
        return scenario_data
    
    def preview_all_scenarios(self):
        """Show a quick preview of all available scenarios"""
        print("🔍 SCENARIO PREVIEW - Quick look at all available maps:")
        print("=" * 70)
        
        scenarios = [
            ("single_corridor", "Two agents must cross paths in narrow corridor"),
            ("s_shaped_corridor", "S-shaped path like user diagram - solvable with negotiations"),
            ("bottleneck_chamber", "Three agents must pass through single bottleneck"),
            ("triple_intersection", "Three agents converge at single intersection"),
            ("validation_stress_test", "HMAS-2 validation test - agents reject central decisions")
        ]
        
        for i, (scenario_type, description) in enumerate(scenarios, 1):
            print(f"\n{i}. {scenario_type.upper()}: {description}")
            print("-" * 40)
            
            # Create and display the scenario map
            try:
                warehouse = self.create_forced_conflict_map(scenario_type)
                
                # Show compact map
                for y in range(warehouse.height):
                    row = f"{y}: "
                    for x in range(warehouse.width):
                        cell = warehouse.grid[y, x]
                        
                        # Show agents and targets on map
                        if (x, y) in warehouse.agents.values():
                            agent_id = [k for k, v in warehouse.agents.items() if v == (x, y)][0]
                            row += f"A{agent_id}"
                        elif (x, y) in warehouse.targets.values():
                            target_id = [k for k, v in warehouse.targets.items() if v == (x, y)][0]
                            row += f"T{target_id}"
                        elif (x, y) in warehouse.boxes.values():
                            box_id = [k for k, v in warehouse.boxes.items() if v == (x, y)][0]
                            row += f"B{box_id}"
                        elif cell == '#':
                            row += "##"
                        else:
                            row += " ."
                    print(row)
                
                print(f"   Agents: {dict(warehouse.agents)}")
                print(f"   Targets: {dict(warehouse.targets)}")
                
            except Exception as e:
                print(f"   Error creating scenario: {e}")
        
        print("\n" + "=" * 70)
    
    def run_benchmark_comparison(self, scenario_type: str, max_turns: int = 20):
        """Run the same scenario with and without spatial hints for comparison"""
        print(f"\n🔬 BENCHMARK COMPARISON: {scenario_type.upper()}")
        print("=" * 70)
        
        # Run without spatial hints (baseline)
        print("\n📊 PHASE 1: Running BASELINE (no spatial hints)...")
        baseline_results = self.run_forced_conflict_test(scenario_type, max_turns, enable_spatial_hints=False)
        
        if baseline_results.get('cancelled'):
            print("Benchmark cancelled during baseline phase.")
            return baseline_results
        
        print("\n⏱️  Waiting 3 seconds before next phase...")
        import time
        time.sleep(3)
        
        # Run with spatial hints (enhanced)
        print("\n📊 PHASE 2: Running ENHANCED (with spatial hints)...")
        enhanced_results = self.run_forced_conflict_test(scenario_type, max_turns, enable_spatial_hints=True)
        
        if enhanced_results.get('cancelled'):
            print("Benchmark cancelled during enhanced phase.")
            return enhanced_results
        
        # Compare results
        print(f"\n📈 COMPARISON RESULTS:")
        print("=" * 50)
        print(f"BASELINE (no hints):")
        print(f"  - Conflicts: {baseline_results.get('total_conflicts', 0)}")
        print(f"  - Negotiations: {len(baseline_results.get('negotiations', []))}")
        print(f"  - Turns: {baseline_results.get('turns_completed', 0)}")
        
        print(f"\nENHANCED (with hints):")
        print(f"  - Conflicts: {enhanced_results.get('total_conflicts', 0)}")
        print(f"  - Negotiations: {len(enhanced_results.get('negotiations', []))}")
        print(f"  - Turns: {enhanced_results.get('turns_completed', 0)}")
        
        # Analyze resolution strategies
        baseline_strategies = self._count_resolution_strategies(baseline_results.get('negotiations', []))
        enhanced_strategies = self._count_resolution_strategies(enhanced_results.get('negotiations', []))
        
        print(f"\nSTRATEGY USAGE:")
        print(f"BASELINE: {baseline_strategies}")
        print(f"ENHANCED: {enhanced_strategies}")
        
        return {
            'baseline': baseline_results,
            'enhanced': enhanced_results,
            'comparison': {
                'baseline_strategies': baseline_strategies,
                'enhanced_strategies': enhanced_strategies
            }
        }
    
    def _count_resolution_strategies(self, negotiations: List[Dict]) -> Dict[str, int]:
        """Count how often each resolution strategy was used"""
        strategy_counts = {'priority': 0, 'reroute': 0, 'wait': 0, 'other': 0}
        
        for neg in negotiations:
            resolution = neg.get('llm_response', {}).get('resolution', 'other')
            if resolution in strategy_counts:
                strategy_counts[resolution] += 1
            else:
                strategy_counts['other'] += 1
        
        return strategy_counts
    
    def save_negotiation_log(self, filename: str | None = None, auto_save: bool = False) -> str:
        """Save all negotiation data to logs folder"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if auto_save:
                filename = f"negotiation_test_{timestamp}_autosave.json"
            else:
                filename = f"negotiation_test_{timestamp}.json"
        
        # Ensure logs directory exists
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        filepath = os.path.join(logs_dir, filename)
        
        # Add summary statistics
        hmas2_validations = []
        validation_overrides = []
        for negotiation in self.negotiation_log['negotiations']:
            if 'hmas2_stages' in negotiation:
                hmas2_validations.extend(negotiation['hmas2_stages'].get('agent_validations', {}).values())
                validation_overrides.extend(negotiation['hmas2_stages'].get('validation_overrides', {}).values())
        
        self.negotiation_log['summary'] = {
            'total_scenarios': len(self.negotiation_log['conflict_scenarios']),
            'total_negotiations': len(self.negotiation_log['negotiations']),
            'total_conflicts': sum(s.get('total_conflicts', 0) for s in self.negotiation_log['conflict_scenarios']),
            'success_rate': len([n for n in self.negotiation_log['negotiations'] if 'error' not in n]) / max(len(self.negotiation_log['negotiations']), 1),
            'hmas2_metrics': {
                'total_agent_validations': len(hmas2_validations),
                'validation_approvals': len([v for v in hmas2_validations if v.get('validation_result', {}).get('valid', False)]),
                'validation_rejections': len([v for v in hmas2_validations if not v.get('validation_result', {}).get('valid', False)]),
                'agent_alternatives_suggested': len(validation_overrides),
                'central_vs_agent_disagreement_rate': len(validation_overrides) / max(len(hmas2_validations), 1)
            },
            'saved_at': datetime.now().isoformat(),
            'auto_save': auto_save
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.negotiation_log, f, indent=2)
            
            if not auto_save:  # Only print detailed info for manual saves
                print(f"\n💾 Negotiation log saved to: {filepath}")
                print(f"📊 Summary: {self.negotiation_log['summary']}")
            else:
                print(f"   ✅ Auto-saved to: {os.path.basename(filepath)}")
            
            self._unsaved_data = False  # Mark as saved
            
        except Exception as e:
            print(f"❌ Error saving negotiation log: {e}")
            raise
        
        return filepath

def main():
    """Run comprehensive negotiation tests"""
    print("🤖 LLM NEGOTIATION TEST SUITE")
    print("=" * 60)
    print("This test creates scenarios guaranteed to cause conflicts")
    print("and logs all LLM negotiations for analysis.")
    print()
    
    tester = NegotiationTester()
    
    # Test mode selection
    print("📊 TEST MODES:")
    print("1. Standard test (with spatial hints)")
    print("2. Baseline test (no spatial hints)")  
    print("3. Benchmark comparison (both modes)")
    print("4. Preview all scenarios")
    
    try:
        mode = input("\nSelect mode (1-4): ").strip()
        
        if mode == "4":
            tester.preview_all_scenarios()
            return
        
        # Scenario selection
        scenarios = [
            ("single_corridor", "Two agents must cross paths in narrow corridor"),
            ("s_shaped_corridor", "S-shaped path like user diagram - solvable with negotiations"),
            ("bottleneck_chamber", "Three agents must pass through single bottleneck"),
            ("triple_intersection", "Three agents converge at single intersection"),
            ("validation_stress_test", "HMAS-2 validation test - agents reject central decisions")
        ]
        
        print("\nAvailable test scenarios:")
        for i, (scenario, description) in enumerate(scenarios, 1):
            print(f"{i}. {scenario}: {description}")
        
        scenario_choice = input("\nSelect scenario (1-5): ").strip()
        try:
            scenario_idx = int(scenario_choice) - 1
            scenario_type = scenarios[scenario_idx][0]
        except (ValueError, IndexError):
            print("❌ Invalid scenario choice")
            return
        
        # Run based on selected mode
        if mode == "1":
            # Standard with hints
            result = tester.run_forced_conflict_test(scenario_type, enable_spatial_hints=True)
        elif mode == "2": 
            # Baseline without hints
            result = tester.run_forced_conflict_test(scenario_type, enable_spatial_hints=False)
        elif mode == "3":
            # Benchmark comparison
            result = tester.run_benchmark_comparison(scenario_type)
        else:
            print("❌ Invalid mode choice")
            return
        
        # Save results (skip if cancelled)
        if not result.get('cancelled'):
            filename = tester.save_negotiation_log()
            print(f"\n💾 Results saved to: {filename}")
        else:
            print("Test was cancelled - no results to save.")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Test interrupted!")
        filename = tester.save_negotiation_log(auto_save=True)
        print(f"💾 Partial results saved to: {filename}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        filename = tester.save_negotiation_log(auto_save=True)
        print(f"💾 Emergency save completed: {filename}")


if __name__ == "__main__":
    main()
