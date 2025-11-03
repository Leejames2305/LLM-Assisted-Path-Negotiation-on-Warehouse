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
from colorama import init, Fore, Style

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.simulation.game_engine import GameEngine
from src.map_generator import WarehouseMap
from src.map_generator.layout_selector import get_layout_for_game

# Initialize colorama
init(autoreset=True)

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
        
        # Load layout from user selection
        warehouse = self.load_layout_for_negotiation_test()
        
        if warehouse is None:
            return {'cancelled': True}
        
        # Display the scenario
        print("🗺️  LOADED LAYOUT MAP:")
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
        
        # Store scenario info
        scenario_data = {
            'type': 'user_selected_layout',
            'map_size': (warehouse.width, warehouse.height),
            'agents': dict(warehouse.agents),
            'targets': dict(warehouse.targets),
            'negotiations': []
        }
        
        # Initialize comprehensive simulation log for visualization
        simulation_log = {
            'scenario': {
                'type': 'user_selected_layout',
                'map_size': [warehouse.width, warehouse.height],
                'initial_agents': {str(k): list(v) for k, v in warehouse.agents.items()},
                'initial_targets': {str(k): list(v) for k, v in warehouse.targets.items()},
                'grid': warehouse.grid.tolist(),  # Include actual grid data for wall detection
                'timestamp': datetime.now().isoformat()
            },
            'turns': [],
            'summary': {}
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
        
        def capture_negotiate(conflict_data, agent_validators=None):  # type: ignore
            # Prevent double-negotiation with flag check
            if getattr(game_engine, '_negotiation_captured', False):
                print("🔄 Skipping duplicate negotiation call - already captured this turn")
                return {
                    'resolution': 'already_handled',
                    'agent_actions': {},
                    'reasoning': 'Negotiation already captured by HMAS-2 test'
                }, []  # Return as tuple
            
            # Set flag to prevent re-negotiation
            setattr(game_engine, '_negotiation_captured', True)
            
            # Get the actual models being used
            central_model = getattr(game_engine.central_negotiator, 'model', 'unknown')
            
            # Sanitize conflict_data to avoid circular references in JSON serialization
            def sanitize_for_json(obj, visited=None):
                """Recursively sanitize objects to remove only actual circular references"""
                if visited is None:
                    visited = set()
                
                obj_id = id(obj)
                
                # Prevent circular references
                if obj_id in visited:
                    return None
                
                if isinstance(obj, dict):
                    visited_copy = visited.copy()
                    visited_copy.add(obj_id)
                    result = {}
                    for k, v in obj.items():
                        if isinstance(k, str):  # Only keep string keys
                            result[k] = sanitize_for_json(v, visited_copy)
                    return result
                elif isinstance(obj, (list, tuple)):
                    visited_copy = visited.copy()
                    visited_copy.add(obj_id)
                    return [sanitize_for_json(item, visited_copy) for item in obj]
                elif isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                else:
                    # Skip non-serializable objects
                    return None
            
            sanitized_conflict_data = sanitize_for_json(conflict_data)
            
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
                'conflict_data': sanitized_conflict_data
            }
            
            # Capture prompts
            def capture_system_prompt():
                prompt = original_system_prompt()
                negotiation_entry['hmas2_stages']['central_negotiation']['system_prompt'] = prompt
                return prompt
            
            def capture_user_prompt(conflict_data):
                prompt = original_conflict_desc(conflict_data)
                negotiation_entry['hmas2_stages']['central_negotiation']['user_prompt'] = prompt
                return prompt
            
            # Temporarily replace methods to capture prompts
            game_engine.central_negotiator._create_negotiation_system_prompt = capture_system_prompt
            game_engine.central_negotiator._create_conflict_description = capture_user_prompt
            
            # Call original method and capture response
            try:
                # CRITICAL: Pass agent_validators to enable refinement loop!
                negotiate_result = original_negotiate(conflict_data, agent_validators=agent_validators)
                
                # Unpack tuple if returned (new negotiate_path_conflict signature returns Tuple[Dict, List[Dict]])
                if isinstance(negotiate_result, tuple):
                    response, refinement_history = negotiate_result
                else:
                    response = negotiate_result
                    refinement_history = []
                
                negotiation_entry['hmas2_stages']['central_negotiation']['llm_response'] = response
                
                print(f"\n💬 CENTRAL LLM RESPONSE PREVIEW:")
                print("-" * 40)
                if isinstance(response, dict):
                    # Just print a snippet of the JSON response
                    response_str = json.dumps(response, indent=2, default=str)[:100]
                    print(response_str)
                    if len(json.dumps(response, default=str)) > 100:
                        print("...")
                else:
                    print(str(response)[:100])
                print("-" * 40)
                
            except Exception as e:
                negotiation_entry['hmas2_stages']['central_negotiation']['error'] = str(e)
                print(f"❌ Central LLM Negotiation Error: {e}")
                response = {'error': str(e), 'fallback': 'wait_action'}
                refinement_history = []
            
            # Restore original methods
            game_engine.central_negotiator._create_negotiation_system_prompt = original_system_prompt
            game_engine.central_negotiator._create_conflict_description = original_conflict_desc
            
            # STAGE 2A: Capture Refinement Loop History (already obtained from tuple unpacking above)
            # If not already obtained, try to get from negotiator (fallback)
            if not refinement_history:
                refinement_history = game_engine.central_negotiator.get_refinement_history()
            
            # Initialize safe_iterations for use in STAGE 2 optimization check
            safe_iterations = []
            
            if refinement_history:

                # Extract only the key information from refinement history to avoid circular refs
                safe_iterations = []
                iteration_num = 1
                for iteration_record in refinement_history:
                    iteration = iteration_record.get('iteration', '?')
                    stage = iteration_record.get('stage', '?')
                    status = iteration_record.get('final_status', '')
                    rejected_by = iteration_record.get('rejected_by', [])
                    validation_results = iteration_record.get('validation_results', {})
                    
                    # Only show validation details for validation stage iterations (not initial negotiation)
                    if stage == 'validation' and validation_results and isinstance(validation_results, dict):
                        for agent_id, val_result in validation_results.items():
                            if isinstance(val_result, dict):
                                valid = val_result.get('valid', False)
                                reason = val_result.get('reason', 'unknown')
                                
                                # Try to get agent model info
                                agent_idx = agent_id if isinstance(agent_id, int) else int(agent_id) if str(agent_id).isdigit() else agent_id
                                if agent_idx in game_engine.agents:
                                    agent = game_engine.agents[agent_idx]
                                    agent_model = getattr(agent.validator, 'model', 'openai/gpt-oss-20b:free') if hasattr(agent, 'validator') else 'openai/gpt-oss-20b:free'
                                    
                                    print(f"🔎 Validation Agent {agent_id}: {[valid, reason, 'action']}\n")
                    
                    if rejected_by and stage == 'validation':
                        print(f"   Iteration {iteration}: {len(rejected_by)} agent(s) rejected")
                    
                    # Store only safe data (avoid storing complex objects)
                    safe_iterations.append({
                        'iteration': iteration,
                        'stage': stage,
                        'final_status': status,
                        'rejected_by': rejected_by if isinstance(rejected_by, list) else list(rejected_by) if rejected_by else [],
                        'timestamp': iteration_record.get('timestamp', '')
                    })
                
                # Add to negotiation entry
                negotiation_entry['hmas2_stages']['refinement_loop'] = {
                    'total_iterations': len(safe_iterations),
                    'final_status': safe_iterations[-1].get('final_status', 'unknown') if safe_iterations else 'none',
                    'iterations': safe_iterations
                }
            else:
                negotiation_entry['hmas2_stages']['refinement_loop'] = {
                    'total_iterations': 0,
                    'final_status': 'none',
                    'iterations': []
                }
            
            # Check if response has agent actions (either as 'agent_actions' key or as agent IDs directly)
            has_agent_actions = False
            if isinstance(response, dict):
                if 'agent_actions' in response or 'actions' in response:
                    has_agent_actions = True
                else:
                    # Check if any agent IDs are present as top-level keys
                    for key in response.keys():
                        if str(key).isdigit():
                            has_agent_actions = True
                            break
            
            # STAGE 2: Capture Agent LLM Validations (if response contains agent actions)
            if has_agent_actions:
                print("=" * 60)
                
                # OPTIMIZATION: Check if refinement loop already approved all paths
                # If so, skip redundant agent validation (same criteria already checked)
                skip_agent_validation = False
                if safe_iterations:
                    # Check if final iteration has NO rejections (unanimous approval)
                    final_iteration = safe_iterations[-1] if safe_iterations else {}
                    rejected_by = final_iteration.get('rejected_by', [])
                    final_status = final_iteration.get('final_status', '').lower()
                    
                    # Unanimous approval if: no rejections OR status says "approved"
                    is_unanimous = (not rejected_by) or ('approved' in final_status)
                    
                    if is_unanimous:
                        skip_agent_validation = True
                
                # Handle different response formats:
                # 1. Response with 'agent_actions' key
                # 2. Response with 'actions' key
                # 3. Response with agent IDs directly as top-level keys
                if 'agent_actions' in response:
                    actions = response.get('agent_actions', {})
                elif 'actions' in response:
                    actions = response.get('actions', {})
                else:
                    # Agent IDs are top-level keys, use response directly (minus non-agent keys)
                    actions = {k: v for k, v in response.items() if str(k).isdigit()}
                
                if isinstance(actions, dict):
                    iteration_num = 1
                    for agent_id, action_data in actions.items():
                        # Convert string agent_id to int if needed for lookup - FIXED TYPE HANDLING
                        agent_id_key = int(agent_id) if isinstance(agent_id, str) and agent_id.isdigit() else agent_id
                        
                        if agent_id_key in game_engine.agents:
                            agent = game_engine.agents[agent_id_key]
                            
                            # OPTIMIZATION: Check if validation can be skipped (refinement loop already approved)
                            if skip_agent_validation:
                                
                                # Create validation entry showing it was pre-validated
                                validation_entry = {
                                    'agent_id': agent_id_key,
                                    'proposed_action': action_data,
                                    'validation_model': 'refinement_loop_pre_validated',
                                    'validation_result': {'valid': True, 'reason': 'pre_validated_by_refinement_loop'},
                                    'alternative_suggested': None,
                                    'final_action_executed': action_data
                                }
                                
                                # Record in both places
                                negotiation_entry['hmas2_stages']['agent_validations'][agent_id] = validation_entry
                                negotiation_entry['hmas2_stages']['final_actions'][agent_id] = action_data
                                
                                # Still flag agent as HMAS-2 validated for game engine
                                setattr(agent, '_hmas2_validated', True)
                                continue  # Skip to next agent
                            
                            # Get the actual agent model being used
                            agent_model = getattr(agent.validator, 'model', 'unknown') if hasattr(agent, 'validator') else 'unknown'
                            
                            # Capture the validation process
                            validation_entry = {
                                'agent_id': agent_id_key,
                                'proposed_action': action_data,
                                'validation_model': agent_model,
                                'validation_result': None,
                                'alternative_suggested': None,
                                'final_action_executed': None
                            }
                            
                            try:
                                # Get the map state for validation
                                map_state = game_engine.warehouse_map.get_state_dict()
                                
                                # Get the actual agent model being used
                                agent_model = getattr(agent.validator, 'model', 'unknown') if hasattr(agent, 'validator') else 'unknown'
                                
                                # Show LLM request being sent
                                print(f"\n📡 Iteration {iteration_num}: Sending request to {Fore.YELLOW}{agent_model}{Style.RESET_ALL}...")
                                
                                # Call the agent's validator - USE THE CONVERTED agent_id_key
                                validation_result = agent.validator.validate_negotiated_action(
                                    agent_id_key, action_data, map_state  # Use converted ID here too
                                )
                                
                                valid = validation_result.get('valid', False)
                                reason = validation_result.get('reason', 'unknown')
                                action_type = action_data.get('action', '?') if isinstance(action_data, dict) else '?'
                                print(f"🔎 Validation Agent {agent_id}: {[valid, reason, action_type]}")
                                
                                validation_entry['validation_result'] = validation_result
                                
                                if not validation_result.get('valid', False):
                                    # COUNT ALL REJECTIONS, not just alternatives
                                    negotiation_entry['hmas2_stages']['validation_overrides'][agent_id] = {
                                        'rejected_central_action': action_data,
                                        'rejection_reason': validation_result.get('reason', 'unknown'),
                                        'agent_alternative': None
                                    }
                                    
                                    if 'alternative' in validation_result:
                                        alternative = validation_result['alternative']
                                        validation_entry['alternative_suggested'] = alternative
                                        
                                        # Update the override entry with the alternative
                                        negotiation_entry['hmas2_stages']['validation_overrides'][agent_id]['agent_alternative'] = alternative
                                        final_action = alternative
                                    else:
                                        final_action = {'action': 'wait', 'reason': 'validation_failed'}
                                        
                                else:
                                    final_action = action_data
                                
                                validation_entry['final_action_executed'] = final_action
                                negotiation_entry['hmas2_stages']['agent_validations'][agent_id] = validation_entry
                                negotiation_entry['hmas2_stages']['final_actions'][agent_id] = final_action
                                
                                # Increment iteration counter for next agent
                                iteration_num += 1
                                
                            except Exception as validation_error:
                                validation_entry['error'] = str(validation_error)
                                print(f"   ❌ Validation Error for Agent {agent_id}: {validation_error}")
                                
                                # Fallback to wait action
                                fallback_action = {'action': 'wait', 'reason': 'validation_error'}
                                validation_entry['final_action_executed'] = fallback_action
                                negotiation_entry['hmas2_stages']['agent_validations'][agent_id] = validation_entry
                                negotiation_entry['hmas2_stages']['final_actions'][agent_id] = fallback_action
                                
                                # Increment iteration counter even on error
                                iteration_num += 1
                        else:
                            print(f"⚠️  Agent {agent_id} (key: {agent_id_key}) not found in game_engine.agents: {list(game_engine.agents.keys())}")
                
                # Flag all agents as HMAS-2 validated to skip redundant game engine validation
                for agent_id_str, validation_data in negotiation_entry['hmas2_stages']['agent_validations'].items():
                    agent_id_int = int(agent_id_str) if isinstance(agent_id_str, str) and agent_id_str.isdigit() else agent_id_str
                    if agent_id_int in game_engine.agents:
                        setattr(game_engine.agents[agent_id_int], '_hmas2_validated', True)
            
            else:
                print(f"\n⚠️  Central LLM response format doesn't contain agent actions")
                print(f"🔧 FALLBACK: Creating mock actions for HMAS-2 validation demo")
                
                # Create mock actions based on conflict data for validation testing
                if 'agents' in conflict_data and len(conflict_data['agents']) > 0:
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
                    
                else:
                    actions = {}
                
                # Process the actions (either from LLM or mock) for validation
                if isinstance(actions, dict) and len(actions) > 0:
                    for agent_id, action_data in actions.items():
                        # Ensure type consistency for mock actions too
                        agent_id_key = int(agent_id) if isinstance(agent_id, str) and agent_id.isdigit() else agent_id
                        
                        if agent_id_key in game_engine.agents:
                            agent = game_engine.agents[agent_id_key]
                            
                            # Get the actual agent model being used
                            agent_model = getattr(agent.validator, 'model', 'unknown') if hasattr(agent, 'validator') else 'unknown'
                            
                            # Capture the validation process
                            validation_entry = {
                                'agent_id': agent_id_key,
                                'proposed_action': action_data,
                                'validation_model': agent_model,
                                'validation_result': None,
                                'alternative_suggested': None,
                                'final_action_executed': None
                            }
                            
                            try:
                                # Get the map state for validation
                                map_state = game_engine.warehouse_map.get_state_dict()
                                
                                # Call the agent's validator
                                validation_result = agent.validator.validate_negotiated_action(
                                    agent_id_key, action_data, map_state
                                )
                                
                                validation_entry['validation_result'] = validation_result
                                
                                print(f"   Agent {agent_id}: {'✅ Approved' if validation_result.get('valid', False) else '❌ Rejected'} ({validation_result.get('reason', 'unknown')})")
                                
                                if not validation_result.get('valid', False):
                                    # COUNT ALL REJECTIONS, not just alternatives
                                    negotiation_entry['hmas2_stages']['validation_overrides'][agent_id_key] = {
                                        'rejected_central_action': action_data,
                                        'rejection_reason': validation_result.get('reason', 'unknown'),
                                        'agent_alternative': None
                                    }
                                    
                                    if 'alternative' in validation_result:
                                        alternative = validation_result['alternative']
                                        validation_entry['alternative_suggested'] = alternative
                                        print(f"      → Suggested alternative: {alternative.get('action', '?')}")
                                        
                                        # Update the override entry with the alternative
                                        negotiation_entry['hmas2_stages']['validation_overrides'][agent_id_key]['agent_alternative'] = alternative
                                        final_action = alternative
                                    else:
                                        final_action = {'action': 'wait', 'reason': 'validation_failed'}
                                        
                                else:
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
            print(f"   Agent Validations: {len(negotiation_entry['hmas2_stages']['agent_validations'])} agents")
            print(f"   Validation Rejections: {len(negotiation_entry['hmas2_stages']['validation_overrides'])} agents")
            
            scenario_data['negotiations'].append(negotiation_entry)
            self.negotiation_log['negotiations'].append(negotiation_entry)
            
            # Mark data as changed and check for auto-save
            self._mark_data_changed()
            self._auto_save_check()
            
            # Reset flag after processing to allow next turn's negotiation
            setattr(game_engine, '_negotiation_captured', False)
            
            # CRITICAL FIX: Ensure response has correct format for game engine
            if isinstance(response, dict):
                # Check if agent IDs are directly in response (as top-level keys)
                agent_ids_as_keys = [k for k in response.keys() if str(k).isdigit()]
                
                # If response has agent IDs as top-level keys, wrap them in 'agent_actions'
                if agent_ids_as_keys and 'agent_actions' not in response:
                    # Keep the agent actions at top level OR wrap them - game engine can handle both
                    # For compatibility with both formats, we'll keep it as-is since game engine now handles both
                    pass
                
                # If response has 'actions' but not 'agent_actions', fix it
                elif 'actions' in response and 'agent_actions' not in response:
                    response['agent_actions'] = response['actions']  # type: ignore
            
            # Return as tuple to match new negotiate_path_conflict signature
            return response, refinement_history
        
        game_engine.central_negotiator.negotiate_path_conflict = capture_negotiate
        
        # Run simulation for specified turns
        print(f"\n🚀 Running simulation...")
        print("Press Enter to start, then Enter for each turn...")
        input()
        
        conflicts_detected = 0
        turn_completed = 0
        
        # CRITICAL FIX: Let simulation run until completion OR max turns, whichever comes first
        while turn_completed < max_turns:
            print(f"\n=== TURN {turn_completed + 1} ===")
            
            # Capture turn state BEFORE executing the turn
            turn_entry = {
                'turn': turn_completed + 1,
                'timestamp': datetime.now().isoformat(),
                'type': 'routine',  # Will be updated to 'negotiation' if conflicts occur
                'agent_states': {
                    str(aid): {
                        'position': list(agent.position),
                        'target': list(agent.target_position) if agent.target_position else None,
                        'carrying_box': agent.carrying_box,
                        'box_id': getattr(agent, 'box_id', None),
                        'is_waiting': agent.is_waiting,
                        'wait_turns_remaining': getattr(agent, 'wait_turns_remaining', 0),
                        'planned_path': agent.planned_path[:10] if hasattr(agent, 'planned_path') and agent.planned_path else [],  # First 10 steps
                        'has_negotiated_path': getattr(agent, '_has_negotiated_path', False)
                    } for aid, agent in game_engine.agents.items()
                },
                'map_state': {
                    'boxes': {str(k): list(v) for k, v in game_engine.warehouse_map.get_state_dict().get('boxes', {}).items()},
                    'targets': {str(k): list(v) for k, v in game_engine.warehouse_map.get_state_dict().get('targets', {}).items()},
                    'dimensions': [game_engine.width, game_engine.height]
                },
                'conflicts_detected': False,
                'negotiation_occurred': False
            }
            
            # Reset negotiation flag at start of each turn for clean state
            if hasattr(game_engine, '_negotiation_captured'):
                setattr(game_engine, '_negotiation_captured', False)
            
            # Reset all agent HMAS-2 validation flags for new turn
            for agent_id, agent in game_engine.agents.items():
                if hasattr(agent, '_hmas2_validated'):
                    setattr(agent, '_hmas2_validated', False)
            
            try:
                # Run one simulation step
                continue_sim = game_engine.run_simulation_step()
                
                # Check if conflicts were detected and negotiation occurred
                if hasattr(game_engine, '_negotiation_captured') and getattr(game_engine, '_negotiation_captured'):
                    turn_entry['type'] = 'negotiation'
                    turn_entry['conflicts_detected'] = True
                    turn_entry['negotiation_occurred'] = True
                    print("📋 Negotiation captured for this turn")
                
                # Count conflicts in this turn
                turn_conflicts = len([n for n in scenario_data['negotiations'] 
                                    if n.get('turn') == game_engine.current_turn])
                if turn_conflicts > 0:
                    conflicts_detected += turn_conflicts
                    print(f"🔥 Conflicts this turn: {turn_conflicts}")
                
                # Capture turn state AFTER executing the turn
                turn_entry['results'] = {
                    'agent_states_after': {
                        str(aid): {
                            'position': list(agent.position),
                            'target': list(agent.target_position) if agent.target_position else None,
                            'carrying_box': agent.carrying_box,
                            'box_id': getattr(agent, 'box_id', None),
                            'is_waiting': agent.is_waiting,
                            'wait_turns_remaining': getattr(agent, 'wait_turns_remaining', 0),
                            'planned_path': agent.planned_path[:10] if hasattr(agent, 'planned_path') and agent.planned_path else [],
                            'has_negotiated_path': getattr(agent, '_has_negotiated_path', False)
                        } for aid, agent in game_engine.agents.items()
                    },
                    'map_state_after': {
                        'boxes': {str(k): list(v) for k, v in game_engine.warehouse_map.get_state_dict().get('boxes', {}).items()},
                        'targets': {str(k): list(v) for k, v in game_engine.warehouse_map.get_state_dict().get('targets', {}).items()},
                        'dimensions': [game_engine.width, game_engine.height]
                    },
                    'simulation_continued': continue_sim
                }
                
                # Add turn entry to simulation log
                simulation_log['turns'].append(turn_entry)
                
                # IMPORTANT: Break if simulation signals completion
                if not continue_sim:
                    print("🏁 Simulation completed!")
                    break
                    
            except Exception as e:
                print(f"❌ Simulation error: {e}")
                break
            
            turn_completed += 1
            input("Press Enter for next turn...")
        
        # Complete simulation log summary
        simulation_log['summary'] = {
            'total_turns': turn_completed,
            'total_conflicts': conflicts_detected,
            'total_negotiations': len(scenario_data['negotiations']),
            'completion_timestamp': datetime.now().isoformat(),
            'negotiation_turns': len([t for t in simulation_log['turns'] if t['type'] == 'negotiation']),
            'routine_turns': len([t for t in simulation_log['turns'] if t['type'] == 'routine'])
        }
        
        # Save comprehensive simulation log
        simulation_log_filename = f"negotiation_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Ensure logs directory exists
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        simulation_log_path = os.path.join(logs_dir, simulation_log_filename)
        
        try:
            with open(simulation_log_path, 'w') as f:
                json.dump(simulation_log, f, indent=2)
            print(f"💾 Comprehensive simulation log saved: {simulation_log_filename}")
        except Exception as e:
            print(f"❌ Failed to save simulation log: {e}")
        
        scenario_data['total_conflicts'] = conflicts_detected
        scenario_data['turns_completed'] = turn_completed
        scenario_data['simulation_log_file'] = simulation_log_filename  # Reference to detailed log
        
        self.negotiation_log['conflict_scenarios'].append(scenario_data)
        self._mark_data_changed()  # Mark that we have new scenario data
        
        print(f"\n📊 SCENARIO RESULTS:")
        print(f"   Total conflicts detected: {conflicts_detected}")
        print(f"   Total negotiations: {len(scenario_data['negotiations'])}")
        print(f"   Turns completed: {scenario_data['turns_completed']}")
        print(f"   Detailed log: {simulation_log_filename}")
        
        return scenario_data
    
    def preview_layouts(self):
        """Show a quick preview of available layouts"""
        from src.map_generator.layout_manager import LayoutManager
        
        print("\n🔍 AVAILABLE LAYOUTS:")
        print("=" * 70)
        
        manager = LayoutManager()
        layouts_info = manager.list_layout_details()
        print(layouts_info)
        print("=" * 70)
    
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
        
        # Function to recursively convert to JSON-safe format
        def make_json_safe(obj, visited=None):
            """Recursively convert object to JSON-safe format, avoiding circular references"""
            if visited is None:
                visited = set()
            
            obj_id = id(obj)
            
            # Prevent circular references
            if obj_id in visited:
                return None
            
            if isinstance(obj, dict):
                visited.add(obj_id)
                result = {}
                for k, v in obj.items():
                    if isinstance(k, str):
                        try:
                            result[k] = make_json_safe(v, visited.copy())
                        except Exception:
                            result[k] = None
                return result
            elif isinstance(obj, (list, tuple)):
                visited.add(obj_id)
                try:
                    result = [make_json_safe(item, visited.copy()) for item in obj]
                    return result
                except Exception:
                    return None
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                # For non-serializable objects, skip them
                return None
        
        # Add summary statistics
        hmas2_validations = []
        validation_overrides = []
        for negotiation in self.negotiation_log['negotiations']:
            if 'hmas2_stages' in negotiation:
                hmas2_validations.extend(negotiation['hmas2_stages'].get('agent_validations', {}).values())
                validation_overrides.extend(negotiation['hmas2_stages'].get('validation_overrides', {}).values())
        
        summary = {
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
            # Create a clean copy of the log without circular references
            safe_log = make_json_safe(self.negotiation_log)
            
            # Ensure we have a dict to add summary to
            if not isinstance(safe_log, dict):
                safe_log = {}
            
            safe_log['summary'] = summary
            
            with open(filepath, 'w') as f:
                json.dump(safe_log, f, indent=2)
            
            if not auto_save:  # Only print detailed info for manual saves
                print(f"\n💾 Negotiation log saved to: {filepath}")
                print(f"📊 Summary: {summary}")
            else:
                print(f"   ✅ Auto-saved to: {os.path.basename(filepath)}")
            
            self._unsaved_data = False  # Mark as saved
            
        except Exception as e:
            print(f"❌ Error saving negotiation log: {e}")
            raise
        
        return filepath

def main():
    """Run negotiation tests with user-selected layouts"""
    print("🤖 LLM NEGOTIATION TEST SUITE")
    print("=" * 60)
    print("Test LLM negotiations with your chosen warehouse layout.")
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
