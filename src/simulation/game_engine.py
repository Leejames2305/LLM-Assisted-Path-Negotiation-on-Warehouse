"""
Main Game Engine for Multi-Robot Warehouse Simulation
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from colorama import init, Fore, Back, Style

from ..map_generator import WarehouseMap, CellType
from ..agents import RobotAgent
from ..llm.central_negotiator import CentralNegotiator
from ..navigation import ConflictDetector, SimplePathfinder
from ..logging import UnifiedLogger

# Initialize colorama for colored terminal output
init(autoreset=True)

class GameEngine:
    def __init__(self, width: int = 8, height: int = 6, num_agents: int = 2):
        """Initialize the game engine with specified parameters"""
        self.width = width
        self.height = height
        self.num_agents = max(2, min(num_agents, 4))  # Ensure 2-4 agents
        
        # Core components
        self.warehouse_map = WarehouseMap(width, height)
        self.agents = {}
        self.central_negotiator = CentralNegotiator()
        self.conflict_detector = ConflictDetector(width, height)
        self.pathfinder = SimplePathfinder(width, height)
        
        # Simulation state
        self.current_turn = 0
        self.max_turns = 100
        self.is_running = False
        self.simulation_complete = False
        
        # Deadlock detection and mitigation
        self.failed_move_counts = {}  # Track consecutive failed moves per agent
        self.agent_position_history = {}  # Track position history for stagnation detection
        self.agent_failed_move_history = {}  # Track actual move failures (not intentional waiting)
        self.max_failed_moves = 3  # Trigger deadlock breaking after 3 failures
        self.stagnation_turns = 3  # Number of turns with failed moves to consider stagnation
        
        # Unified Logging
        self.log_enabled = os.getenv('LOG_SIMULATION', 'true').lower() == 'true'
        self.logger = UnifiedLogger() if self.log_enabled else None
        
        # Track current negotiation data for logging
        self._current_negotiation_data = None
        
        # Performance Metrics Tracking
        self.simulation_start_time = None
        self.collision_count = 0
        self.total_token_usage = 0
        self.negotiation_times = []  # List of (start_time, end_time) tuples
        self.successful_deliveries = 0
        self.agent_paths = {}  # Track total path length per agent
        self.initial_agent_positions = {}  # For path efficiency calculation
        
        # Benchmark mode controls
        self.stop_requested = False  # External signal to stop simulation
        self.timeout_seconds = 0  # Time limit for simulation (0 = no limit)
        self.silent_mode = False  # Suppress print output for benchmark runs
    
    # Initialize a new simulation
    def initialize_simulation(self):
        print(f"{Fore.CYAN}Initializing Multi-Robot Warehouse Simulation...{Style.RESET_ALL}")
        
        # Note: warehouse_map is already loaded from layout in main.py
        # Verify map is properly initialized
        if not self.warehouse_map or self.warehouse_map.width == 0:
            raise ValueError("Warehouse map not properly initialized. Load a layout first.")
        
        # Initialize agents from the layout
        # Agents are already created in main.py, but set up their targets here
        for agent_id, agent in self.agents.items():
            # Assign box to pick up (proper warehouse task: box → target)
            if agent_id in self.warehouse_map.agent_goals:
                box_id = agent_id  # Each agent gets their own box
                if box_id in self.warehouse_map.boxes:
                    box_pos = self.warehouse_map.boxes[box_id]
                    agent.set_target(box_pos)  # First go to the box
            
            # Initialize metrics tracking for this agent
            self.initial_agent_positions[agent_id] = agent.position
            self.agent_paths[agent_id] = [agent.position]
        
        # Start tracking simulation time
        self.simulation_start_time = time.time()
        
        # Initial pathfinding
        self._plan_initial_paths()
        
        # Initialize unified logger with scenario data
        if self.log_enabled and self.logger:
            self.logger.initialize({
                'type': 'interactive_simulation',
                'map_size': [self.warehouse_map.width, self.warehouse_map.height],
                'grid': self.warehouse_map.grid.tolist(),
                'initial_agents': {str(k): list(v) for k, v in self.warehouse_map.agents.items()},
                'initial_targets': {str(k): list(v) for k, v in self.warehouse_map.targets.items()},
                'initial_boxes': {str(k): list(v) for k, v in self.warehouse_map.boxes.items()},
                'agent_goals': {str(k): v for k, v in self.warehouse_map.agent_goals.items()}
            })
        
        # Log initial state
        self._log_turn_state("SIMULATION_START")
        
        print(f"{Fore.GREEN}Simulation initialized successfully!{Style.RESET_ALL}")
        self._update_map_state()
        self.display_map()
    
    # Plan initial paths for all agents
    def _plan_initial_paths(self):
        print("Planning initial paths for all agents...")
        
        for agent_id, agent in self.agents.items():
            map_state = self.warehouse_map.get_state_dict()
            path = agent.plan_path(map_state)
    
    # Detect when agents have failed moves for multiple turns
    def detect_stagnation_conflicts(self) -> Dict:
        # Check for agents with consecutive failed moves (not just staying in same position)
        stagnant_agents = []
        
        for agent_id, agent in self.agents.items():
            # Only check agents with active targets
            if agent.target_position is not None:
                failed_move_count = len(self.agent_failed_move_history.get(agent_id, []))
                
                if failed_move_count >= self.stagnation_turns:
                    stagnant_agents.append(agent_id)
                    print(f"🚫 Agent {agent_id}: Stagnant due to {failed_move_count} consecutive failed moves")
        
        if stagnant_agents:
            print(f"🚫 STAGNATION DETECTED! Agents with failed moves: {stagnant_agents}")
            
            # Calculate fresh planned paths for stagnant agents to provide context
            map_state = self.warehouse_map.get_state_dict()
            agent_data = []
            
            for aid in stagnant_agents:
                agent = self.agents[aid]
                
                # Get fresh planned path for context
                current_path = agent.planned_path if hasattr(agent, 'planned_path') else []
                
                # If path is empty or agent has target, try to calculate a fresh path
                if not current_path and agent.target_position:
                    try:
                        fresh_path = agent.plan_path(map_state)
                        if fresh_path:
                            current_path = fresh_path
                            print(f"🗺️  Agent {aid}: Calculated fresh path with {len(fresh_path)} steps for stagnation context")
                    except Exception as e:
                        print(f"⚠️  Agent {aid}: Could not calculate fresh path: {e}")
                        current_path = []
                
                agent_data.append({
                    'id': aid,
                    'current_pos': agent.position,
                    'target_pos': agent.target_position,
                    'planned_path': current_path,
                    'stuck_reason': 'failed_moves',
                    'failed_move_count': len(self.agent_failed_move_history.get(aid, []))
                })
            
            return {
                'has_conflicts': True,
                'conflict_type': 'stagnation',
                'conflicting_agents': stagnant_agents,
                'conflict_points': [self.agents[aid].position for aid in stagnant_agents],
                'agents': agent_data
            }
        
        return {'has_conflicts': False}
    
    # Detect agents stuck due to too many failed moves
    def detect_move_failure_deadlocks(self, planned_moves: Dict) -> Dict:
        stuck_agents = []
        
        # Check for agents with too many failed moves
        for agent_id, failure_count in self.failed_move_counts.items():
            if failure_count >= self.max_failed_moves:
                stuck_agents.append(agent_id)
        
        if stuck_agents:
            print(f"🔥 DEADLOCK DETECTED! Agents with {self.max_failed_moves}+ failed moves: {stuck_agents}")
            return {
                'has_conflicts': True,
                'conflict_type': 'deadlock',
                'conflicting_agents': stuck_agents,
                'conflict_points': [self.agents[aid].position for aid in stuck_agents if aid in self.agents],
                'agents': [
                    {
                        'id': aid,
                        'current_pos': self.agents[aid].position,
                        'target_pos': self.agents[aid].target_position,
                        'planned_path': planned_moves.get(aid, []),
                        'stuck_reason': 'failed_moves',
                        'failure_count': self.failed_move_counts.get(aid, 0)
                    } for aid in stuck_agents if aid in self.agents
                ],
                'deadlock_breaking': True  # Special flag for negotiator
            }
        
        return {'has_conflicts': False}
    
    # Force deadlock negotiation for stuck agents
    def _force_deadlock_negotiation(self, stuck_agents: List[int], planned_moves: Dict):
        print("🛠️ DEADLOCK BREAKING: Creating artificial conflict to trigger negotiation")
        
        # Create artificial conflict data for deadlock breaking
        conflict_data = self.detect_move_failure_deadlocks(planned_moves)
        
        if conflict_data['has_conflicts']:
            # Force negotiation
            print(f"🤖 Forcing negotiation for deadlock resolution...")
            resolution = self._negotiate_conflicts(conflict_data, planned_moves)
            self._execute_negotiated_actions(resolution)
            
            # Reset failure counts after forced negotiation
            for agent_id in stuck_agents:
                self.failed_move_counts[agent_id] = 0
                # Also clear failed move history
                if agent_id in self.agent_failed_move_history:
                    self.agent_failed_move_history[agent_id] = []
                print(f"🔄 Agent {agent_id}: Reset failure count and move history after deadlock negotiation")
    
    # Run one step of the simulation with forced conflict detection
    def run_simulation_step(self) -> bool:
        # Check for external stop request (benchmark timeout)
        if self.stop_requested:
            return False
        
        # Check for timeout if configured
        if self.timeout_seconds > 0 and self.simulation_start_time:
            elapsed = time.time() - self.simulation_start_time
            if elapsed >= self.timeout_seconds:
                if not self.silent_mode:
                    print(f"\n{Fore.YELLOW}⏱️  Time limit ({self.timeout_seconds}s) reached!{Style.RESET_ALL}")
                return False
        
        if self.simulation_complete or self.current_turn >= self.max_turns:
            return False
        
        if not self.silent_mode:
            print(f"\n{Fore.YELLOW}=== TURN {self.current_turn + 1} ==={Style.RESET_ALL}")
        
        # Update all agents for new turn
        for agent in self.agents.values():
            agent.update_turn()
        
        # Reset HMAS-2 validation flags for new turn (so agents validate if needed)
        for agent_id, agent in self.agents.items():
            if hasattr(agent, '_hmas2_validated'):
                setattr(agent, '_hmas2_validated', False)
        
        # Always check for deliveries, even if agents don't move
        for agent_id in self.agents.keys():
            self._check_box_delivery(agent_id)
        
        # PHASE 0: Check for stagnation (agents stuck in same position)
        stagnation_conflict = self.detect_stagnation_conflicts()
        if stagnation_conflict['has_conflicts']:
            print(f"🚫 STAGNATION DETECTED! Forcing negotiation for stuck agents...")
            # Force negotiation for stagnant agents
            resolution = self._negotiate_conflicts(stagnation_conflict, {})
            self._execute_negotiated_actions(resolution)
            
            # Reset position history after forced resolution
            for agent_id in stagnation_conflict['conflicting_agents']:
                if agent_id in self.agent_position_history:
                    self.agent_position_history[agent_id] = []
                # Also clear failed move history after stagnation resolution
                if agent_id in self.agent_failed_move_history:
                    self.agent_failed_move_history[agent_id] = []
                print(f"🧹 Agent {agent_id}: Cleared move histories after stagnation resolution")
            
            # Increment turn and continue
            self.current_turn += 1
            return True
        
        # PHASE 1: Get forced planned moves (ignoring other agents) for conflict detection
        forced_moves = self._get_forced_planned_moves()
        
        if not forced_moves:
            print("No agents have targets to move towards.")
            # Check if all tasks are complete
            if self._check_completion():
                print("🎉 All tasks completed! Simulation complete!")
                self.simulation_complete = True
            return False
        
        # Check for conflicts using forced paths
        conflict_info = self.conflict_detector.detect_path_conflicts(forced_moves, self.current_turn)
        
        # Track if negotiation occurred this turn
        negotiation_occurred = False
        
        if conflict_info['has_conflicts']:
            print(f"{Fore.RED}CONFLICT DETECTED!{Style.RESET_ALL}")
            print(f"Conflicting agents: {conflict_info['conflicting_agents']}")
            print(f"Conflict points: {conflict_info['conflict_points']}")
            
            # Track collision for metrics
            self.collision_count += len(conflict_info['conflict_points'])
            
            # Use Central Negotiator to resolve conflicts
            resolution = self._negotiate_conflicts(conflict_info, forced_moves)
            self._execute_negotiated_actions(resolution)
            negotiation_occurred = True
        else:
            # PHASE 2: No conflicts in forced paths - try normal planning
            print(f"{Fore.GREEN}No conflicts in forced paths. Trying normal planning...{Style.RESET_ALL}")
            
            normal_moves = self._get_normal_planned_moves()
            if normal_moves:
                # Double-check for conflicts in normal paths
                normal_conflict_info = self.conflict_detector.detect_path_conflicts(normal_moves, self.current_turn)
                
                if normal_conflict_info['has_conflicts']:
                    print(f"{Fore.YELLOW}Conflicts found in normal paths - negotiating...{Style.RESET_ALL}")
                    # Track collision for metrics
                    self.collision_count += len(normal_conflict_info['conflict_points'])
                    resolution = self._negotiate_conflicts(normal_conflict_info, normal_moves)
                    self._execute_negotiated_actions(resolution)
                    negotiation_occurred = True
                else:
                    print(f"{Fore.GREEN}No conflicts detected. Executing planned moves...{Style.RESET_ALL}")
                    self._execute_planned_moves(normal_moves)
                    
                    # PHASE 3: Check for deadlock after move execution
                    deadlock_conflict = self.detect_move_failure_deadlocks(normal_moves)
                    if deadlock_conflict['has_conflicts']:
                        print(f"🔥 DEADLOCK AFTER MOVES! Forcing resolution...")
                        self._force_deadlock_negotiation(deadlock_conflict['conflicting_agents'], normal_moves)
                        negotiation_occurred = True
            else:
                print(f"{Fore.YELLOW}No agents can plan paths - all waiting...{Style.RESET_ALL}")
        
        # Update map with new agent positions
        self._update_map_state()
        
        # Check if simulation is complete
        if self._check_completion():
            print(f"{Fore.GREEN}🎉 All agents reached their targets! Simulation complete!{Style.RESET_ALL}")
            self.simulation_complete = True
        
        # Log turn state (with negotiation type if negotiation occurred)
        self._log_turn_state("negotiation" if negotiation_occurred else "TURN_COMPLETE")
        
        # Display current state
        self.display_map()
        self._display_agent_status()
        
        self.current_turn += 1
        return not self.simulation_complete
    
    # Get planned moves for all active agents
    def _get_planned_moves(self) -> Dict[int, List[Tuple[int, int]]]:
        return self._get_normal_planned_moves()
    
    # Get forced planned moves (ignoring other agents) for conflict detection
    def _get_forced_planned_moves(self) -> Dict[int, List[Tuple[int, int]]]:
        forced_moves = {}
        
        for agent_id, agent in self.agents.items():
            if not agent.is_waiting and agent.target_position:
                # CRITICAL FIX: Check if agent has a negotiated path first
                if hasattr(agent, '_has_negotiated_path') and getattr(agent, '_has_negotiated_path', False) and agent.planned_path:
                    # Use the existing negotiated path instead of replanning
                    forced_moves[agent_id] = agent.planned_path.copy()
                else:
                    # Force path planning ignoring other agents
                    map_state = self.warehouse_map.get_state_dict()
                    forced_path = self._plan_forced_path(agent, map_state)
                    
                    if forced_path:
                        forced_moves[agent_id] = forced_path
        
        return forced_moves
    
    # Get normal planned moves (avoiding other agents)
    def _get_normal_planned_moves(self) -> Dict[int, List[Tuple[int, int]]]:
        planned_moves = {}
        
        for agent_id, agent in self.agents.items():
            if not agent.is_waiting and agent.target_position:
                # CRITICAL FIX: Check if agent has a negotiated path that should be preserved
                has_negotiated_path = (hasattr(agent, '_has_negotiated_path') and 
                                     getattr(agent, '_has_negotiated_path', False) and
                                     agent.planned_path and 
                                     len(agent.planned_path) > 1)
                
                if has_negotiated_path:
                    # Agent has a negotiated path - preserve it, don't re-plan
                    planned_moves[agent_id] = agent.planned_path.copy()
                else:
                    # Check if we need to re-plan (no path, or path doesn't lead to current target)
                    needs_replan = (not agent.planned_path or 
                                  (agent.planned_path and agent.planned_path[-1] != agent.target_position))
                    
                    if needs_replan:
                        map_state = self.warehouse_map.get_state_dict()
                        normal_path = self._plan_normal_path(agent, map_state)
                        agent.planned_path = normal_path
                    
                    if agent.planned_path:
                        planned_moves[agent_id] = agent.planned_path.copy()
        
        return planned_moves
    
    # Plan path ignoring other agents - for conflict detection
    def _plan_forced_path(self, agent, map_state: Dict) -> List[Tuple[int, int]]:
        # Extract only walls as obstacles (ignore other agents)
        walls = set()
        grid = map_state.get('grid', [])
        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                if cell == '#':
                    walls.add((x, y))
        
        # Use pathfinder with only walls as obstacles
        forced_path = self.pathfinder.find_path_with_obstacles(
            start=agent.position,
            goal=agent.target_position,
            walls=walls,
            agent_positions={},  # Empty - ignore other agents
            exclude_agent=agent.agent_id
        )
        
        return forced_path
    
    # Plan path avoiding other agents - for actual movement
    def _plan_normal_path(self, agent, map_state: Dict) -> List[Tuple[int, int]]:
        # Extract walls
        walls = set()
        grid = map_state.get('grid', [])
        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                if cell == '#':
                    walls.add((x, y))
        
        # Get other agent positions to avoid
        agent_positions = {}
        for other_id, other_agent in self.agents.items():
            if other_id != agent.agent_id:
                agent_positions[other_id] = other_agent.position
        
        # Use pathfinder avoiding walls and other agents
        normal_path = self.pathfinder.find_path_with_obstacles(
            start=agent.position,
            goal=agent.target_position,
            walls=walls,
            agent_positions=agent_positions,
            exclude_agent=agent.agent_id
        )
        
        return normal_path
    
    # Use Central Negotiator to resolve conflicts
    def _negotiate_conflicts(self, conflict_info: Dict, planned_moves: Dict) -> Dict:
        print("🤖 Initiating LLM-based conflict negotiation...")
        
        # Track negotiation timing
        negotiation_start_time = time.time()
        
        # Prepare conflict data for negotiator
        conflict_data = {
            'agents': [],
            'conflict_points': conflict_info['conflict_points'],
            'map_state': self.warehouse_map.get_state_dict(),
            'turn': self.current_turn
        }
        
        # Add agent data for conflicting agents
        for agent_id in conflict_info['conflicting_agents']:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                conflict_data['agents'].append({
                    'id': agent_id,
                    'current_pos': agent.position,
                    'target_pos': agent.target_position,
                    'planned_path': planned_moves.get(agent_id, [])
                })
        
        # Prepare validators for refinement loop
        agent_validators = {}
        for agent_id in conflict_info['conflicting_agents']:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                if hasattr(agent, 'validator'):
                    agent_validators[agent_id] = agent.validator.validate_negotiated_action
        
        # Get negotiation result from Central LLM
        # Returns Tuple[Dict, List[Dict], Dict] = (final_actions, refinement_history, prompts)
        negotiation_result = self.central_negotiator.negotiate_path_conflict(
            conflict_data, 
            agent_validators=agent_validators
        )
        
        # Handle result format (tuple with prompts data)
        if isinstance(negotiation_result, tuple):
            if len(negotiation_result) >= 3:
                resolution, refinement_history, prompts_data = negotiation_result
            else:
                resolution, refinement_history = negotiation_result  # type: ignore
                prompts_data = {}
        else:
            resolution = negotiation_result
            refinement_history = []
            prompts_data = {}
        
        # Track negotiation time
        negotiation_end_time = time.time()
        negotiation_duration = negotiation_end_time - negotiation_start_time
        self.negotiation_times.append((negotiation_start_time, negotiation_end_time))
        
        # Check for deadlock (empty dict means turn should be skipped)
        if not resolution:  # Empty dict
            print(f"🛑 Negotiation deadlock - turn skipped (no movement)")
            self._current_negotiation_data = self._build_negotiation_log_data(
                conflict_data, prompts_data, {}, refinement_history, {}, {}
            )
            return {
                'agent_actions': {},
                'resolution': 'deadlock_skipped',
                'reasoning': 'Negotiation failed to resolve after max refinement iterations',
                'refinement_history': refinement_history
            }
        
        # Build negotiation log data for unified logger
        # Extract agent validations from refinement history or resolution
        agent_validations = self._extract_agent_validations(resolution, refinement_history)
        final_actions = self._extract_final_actions(resolution)
        
        self._current_negotiation_data = self._build_negotiation_log_data(
            conflict_data, prompts_data, resolution, refinement_history, agent_validations, final_actions
        )
        
        # Add refinement history if available
        if refinement_history:
            resolution['refinement_history'] = refinement_history
        
        return resolution
    
    # Build structured negotiation log data
    def _build_negotiation_log_data(
        self, 
        conflict_data: Dict, 
        prompts_data: Dict, 
        llm_response: Dict, 
        refinement_history: List, 
        agent_validations: Dict,
        final_actions: Dict
        ) -> Dict:

        from ..logging.unified_logger import create_negotiation_data
        
        # Extract prompts from prompts_data or use defaults
        system_prompt = prompts_data.get('system_prompt', '')
        user_prompt = prompts_data.get('user_prompt', '')
        model_used = prompts_data.get('model_used', getattr(self.central_negotiator, 'model', 'unknown'))
        
        # Build refinement loop structure
        refinement_loop = {
            'total_iterations': len(refinement_history),
            'final_status': refinement_history[-1].get('final_status', 'unknown') if refinement_history else 'none',
            'iterations': refinement_history
        }
        
        return create_negotiation_data(
            conflict_data=conflict_data,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            llm_response=llm_response,
            model_used=model_used,
            agent_validations=agent_validations,
            refinement_loop=refinement_loop,
            final_actions=final_actions
        )
    
    # Extract agent validation results from negotiation data and refinement history
    def _extract_agent_validations(self, resolution: Dict, refinement_history: List) -> Dict:
        validations = {}
        
        # Check refinement history for validation results
        for iteration in refinement_history:
            val_results = iteration.get('validation_results', {})
            if val_results:
                for agent_id, result in val_results.items():
                    validations[str(agent_id)] = {
                        'agent_id': agent_id,
                        'validation_result': result,
                        'alternative_suggested': result.get('alternative') if isinstance(result, dict) else None
                    }
        
        return validations
    
    # Extract final actions from negotiation resolution
    def _extract_final_actions(self, resolution: Dict) -> Dict:
        if 'agent_actions' in resolution:
            return resolution['agent_actions']
        # Check for agent IDs as top-level keys
        return {k: v for k, v in resolution.items() if str(k).isdigit()}
    
    # Execute actions determined by negotiation
    def _execute_negotiated_actions(self, resolution: Dict):
        # Handle both formats: response with 'agent_actions' key or agent IDs directly as keys
        if 'agent_actions' in resolution:
            agent_actions = resolution.get('agent_actions', {})
        else:
            # Agent IDs are top-level keys, but filter out non-numeric keys (metadata)
            agent_actions = {k: v for k, v in resolution.items() 
                           if isinstance(k, (int, str)) and (isinstance(k, int) or k.isdigit())}
        
        # Mark all agents as HMAS-2 pre-validated before executing
        # The central negotiator already validated these paths with agents validators,
        # so skip redundant validation in execute_negotiated_action
        for agent_id_str in agent_actions.keys():
            if isinstance(agent_id_str, str) and agent_id_str.isdigit():
                agent_id_key = int(agent_id_str)
            elif isinstance(agent_id_str, int):
                agent_id_key = agent_id_str
            else:
                continue
            
            if agent_id_key in self.agents:
                setattr(self.agents[agent_id_key], '_hmas2_validated', True)
        
        for agent_id, action_data in agent_actions.items():
            # Convert string agent_id to int if needed for lookup
            if isinstance(agent_id, str) and agent_id.isdigit():
                agent_id_key = int(agent_id)
            elif isinstance(agent_id, int):
                agent_id_key = agent_id
            else:
                continue
            
            if agent_id_key in self.agents:
                agent = self.agents[agent_id_key]
                map_state = self.warehouse_map.get_state_dict()
                
                # Update agent's planned path with negotiated path
                negotiated_path = action_data.get('path', [])
                if negotiated_path and len(negotiated_path) > 0:
                    # Convert path elements to tuples for consistency
                    # Store the path as-is from the LLM response
                    # If it includes current position, agent will "wait" on first execution
                    # If it doesn't include current position, agent will move immediately
                    updated_path = [tuple(pos) if isinstance(pos, (list, tuple)) else pos for pos in negotiated_path]
                    agent.set_path(updated_path)
                    
                    # Mark agent as having a negotiated path to preserve it
                    agent._has_negotiated_path = True
                
                success = agent.execute_negotiated_action(action_data, map_state)
                
                if success:
                    # After successful move, update the negotiated path by removing the first step
                    if hasattr(agent, '_has_negotiated_path') and getattr(agent, '_has_negotiated_path', False) and agent.planned_path:
                        if len(agent.planned_path) > 1:
                            # Remove the first step and keep the rest
                            agent.planned_path = agent.planned_path[1:]
                        else:
                            # Path completed, clear the negotiated path flag
                            agent._has_negotiated_path = False
                            agent.planned_path = []
                    
                    # Check for box interactions after successful move
                    action_type = action_data.get('action', 'unknown')
                    if action_type == 'move':
                        self._check_box_pickup(agent_id_key)
                        self._check_box_delivery(agent_id_key)
                        
                else:
                    # On failure, clear the negotiated path flag to allow replanning
                    if hasattr(agent, '_has_negotiated_path'):
                        agent._has_negotiated_path = False
    
    # Execute planned moves without conflicts
    def _execute_planned_moves(self, planned_moves: Dict):
        for agent_id, path in planned_moves.items():
            if agent_id in self.agents and len(path) > 0:
                agent = self.agents[agent_id]
                
                # Determine if this is a negotiated path or a normal path
                # Negotiated paths come from LLM and may/may not include current position
                # Normal paths come from pathfinder and always include current position as first element
                is_negotiated_path = (hasattr(agent, '_has_negotiated_path') and 
                                     getattr(agent, '_has_negotiated_path', False))
                
                # For normal paths, skip first element (current position)
                # For negotiated paths, start from first element (handles both inclusion/exclusion of current pos)
                if is_negotiated_path:
                    start_index = 0
                else:
                    start_index = 1 if len(path) > 1 else 0
                
                if start_index >= len(path):
                    print(f"⏸️  Agent {agent_id}: No moves remaining in path")
                    continue
                
                next_pos = path[start_index]
                
                # Handle "waiting in place" moves (when next_pos == current_pos)
                if next_pos == agent.position:
                    # This is a "wait" move - agent should stay in current position
                    print(f"⏸️  Agent {agent_id}: Waiting at {agent.position}")
                    success = True  # Waiting is always successful
                else:
                    # Normal move to different position
                    map_state = self.warehouse_map.get_state_dict()
                    success = agent.move_to(next_pos, map_state)
                
                if success:
                    if next_pos != agent.position:
                        print(f"✅ Agent {agent_id}: Moved to {next_pos}")
                        # Track path length for metrics
                        if agent_id not in self.agent_paths:
                            self.agent_paths[agent_id] = [agent.position]
                        self.agent_paths[agent_id].append(next_pos)
                    
                    # Reset failure count on successful move
                    self.failed_move_counts[agent_id] = 0
                    
                    # Clear failed move history on successful move or wait
                    if agent_id in self.agent_failed_move_history:
                        self.agent_failed_move_history[agent_id] = []
                    
                    # If agent has a negotiated path, advance it properly
                    if (hasattr(agent, '_has_negotiated_path') and 
                        getattr(agent, '_has_negotiated_path', False) and 
                        agent.planned_path and 
                        len(agent.planned_path) > 1):
                        
                        # Remove the first step (current position) from negotiated path
                        agent.planned_path = agent.planned_path[1:]
                        print(f"🔄 Agent {agent_id}: Advanced negotiated path, {len(agent.planned_path)} steps remaining")
                        
                        # If negotiated path is completed, clear the flag
                        if len(agent.planned_path) <= 1:
                            agent._has_negotiated_path = False
                            print(f"🏁 Agent {agent_id}: Negotiated path completed")
                    
                    # Check for box pickup
                    self._check_box_pickup(agent_id)
                    
                    # Check for box delivery
                    self._check_box_delivery(agent_id)
                    
                else:
                    # Track failed moves for deadlock detection
                    self.failed_move_counts[agent_id] = self.failed_move_counts.get(agent_id, 0) + 1
                    
                    # IMPORTANT: Track actual move failures for stagnation detection
                    if agent_id not in self.agent_failed_move_history:
                        self.agent_failed_move_history[agent_id] = []
                    
                    self.agent_failed_move_history[agent_id].append({
                        'turn': self.current_turn,
                        'attempted_move': next_pos,
                        'from_position': agent.position
                    })
                    
                    # Keep only recent failed move history
                    if len(self.agent_failed_move_history[agent_id]) > self.stagnation_turns:
                        self.agent_failed_move_history[agent_id] = self.agent_failed_move_history[agent_id][-self.stagnation_turns:]
                    
                    print(f"❌ Agent {agent_id}: Move to {next_pos} failed ({self.failed_move_counts[agent_id]} consecutive failures)")
    
    # Check if agent can pick up a box at current position
    def _check_box_pickup(self, agent_id: int):
        agent = self.agents[agent_id]
        
        # Only pick up if not already carrying and at box position
        if not agent.carrying_box:
            box_id = agent_id  # Each agent has their own box
            if box_id in self.warehouse_map.boxes:
                box_pos = self.warehouse_map.boxes[box_id]
                if agent.position == box_pos:
                    # Pick up the box
                    success = self.warehouse_map.pickup_box(agent_id, box_id)
                    if success:
                        agent.pickup_box(box_id)
                        print(f"📦 Agent {agent_id}: Picked up box {box_id}")
                        
                        # Set new target to delivery location
                        target_id = self.warehouse_map.agent_goals.get(agent_id)
                        if target_id is not None and target_id in self.warehouse_map.targets:
                            target_pos = self.warehouse_map.targets[target_id]
                            agent.set_target(target_pos)
                            print(f"🎯 Agent {agent_id}: New target set to delivery point {target_pos}")
                            
                            # Force immediate path re-planning after target change
                            map_state = self.warehouse_map.get_state_dict()
                            agent.plan_path(map_state)
    
    # Check if agent can deliver box at current position
    def _check_box_delivery(self, agent_id: int):
        agent = self.agents[agent_id]
        
        # Only deliver if carrying box and at target position
        if agent.carrying_box and agent.box_id is not None:
            target_id = self.warehouse_map.agent_goals.get(agent_id)
            if target_id is not None and target_id in self.warehouse_map.targets:
                target_pos = self.warehouse_map.targets[target_id]
                if agent.position == target_pos:
                    # Deliver the box
                    success = self.warehouse_map.drop_box(agent_id, target_id)
                    if success:
                        delivered_box_id = agent.drop_box()
                        print(f"🎉 Agent {agent_id}: Delivered box {delivered_box_id} to target {target_id}")
                        agent.set_target(None)  # Task complete
                        
                        # Track successful delivery for metrics
                        self.successful_deliveries += 1
                        
                        # Clear failed move history when task is completed
                        if agent_id in self.agent_failed_move_history:
                            self.agent_failed_move_history[agent_id] = []
                            print(f"🧹 Agent {agent_id}: Cleared failed move history (task completed)")
    
    # Update warehouse map with current agent positions
    def _update_map_state(self):
        # Clear agent positions on map, but preserve other elements
        for y in range(self.height):
            for x in range(self.width):
                cell_type = self.warehouse_map.grid[y, x]
                if cell_type in [CellType.AGENT.value, CellType.AGENT_WITH_BOX.value]:
                    self.warehouse_map.grid[y, x] = CellType.EMPTY.value
        
        # Restore targets that might have been overwritten
        for target_id, (target_x, target_y) in self.warehouse_map.targets.items():
            if self.warehouse_map.grid[target_y, target_x] == CellType.EMPTY.value:
                self.warehouse_map.grid[target_y, target_x] = CellType.TARGET.value
        
        # Restore boxes that might have been overwritten
        for box_id, (box_x, box_y) in self.warehouse_map.boxes.items():
            if self.warehouse_map.grid[box_y, box_x] == CellType.EMPTY.value:
                self.warehouse_map.grid[box_y, box_x] = CellType.BOX.value
        
        # Place agents at new positions (this will overwrite targets/boxes if agent is on them)
        for agent_id, agent in self.agents.items():
            x, y = agent.position
            if agent.carrying_box:
                self.warehouse_map.grid[y, x] = CellType.AGENT_WITH_BOX.value
            else:
                self.warehouse_map.grid[y, x] = CellType.AGENT.value
            
            # Update warehouse map's agent tracking
            self.warehouse_map.agents[agent_id] = agent.position
    
    # Check if all tasks are complete
    def _check_completion(self) -> bool:
        # Check if all boxes have been delivered (removed from the map)
        if self.warehouse_map.boxes:
            return False  # Still boxes to be delivered
        
        # Alternative check: all agents have no active targets (completed their tasks)
        for agent in self.agents.values():
            if agent.target_position is not None:
                return False  # Agent still has work to do
                
        return True
    
    # Display current warehouse map with color coding
    def display_map(self):
        print(f"\n{Fore.CYAN}Current Warehouse State:{Style.RESET_ALL}")
        
        # Compute fixed widths for scalable rendering
        row_label_width = len(str(self.height - 1))
        col_width = max(3, len(str(self.width - 1)) + 1)
        
        # Add column numbers with fixed-width formatting
        header = " " * (row_label_width + 2) + "".join(f"{i:^{col_width}}" for i in range(self.width))
        print(header)
        
        for y in range(self.height):
            row = f"{y:>{row_label_width}}: "
            for x in range(self.width):
                cell = self.warehouse_map.grid[y, x]
                
                # Padding to center cell symbol within col_width
                pad_left = (col_width - 1) // 2
                pad_right = col_width - 1 - pad_left
                
                # Color coding
                if cell == CellType.AGENT.value:
                    row += " " * pad_left + f"{Fore.BLUE}{cell}{Style.RESET_ALL}" + " " * pad_right
                elif cell == CellType.AGENT_WITH_BOX.value:
                    row += " " * pad_left + f"{Fore.MAGENTA}{cell}{Style.RESET_ALL}" + " " * pad_right
                elif cell == CellType.BOX.value:
                    row += " " * pad_left + f"{Fore.YELLOW}{cell}{Style.RESET_ALL}" + " " * pad_right
                elif cell == CellType.TARGET.value:
                    row += " " * pad_left + f"{Fore.GREEN}{cell}{Style.RESET_ALL}" + " " * pad_right
                elif cell == CellType.WALL.value:
                    row += " " * pad_left + f"{Back.BLACK}{cell}{Style.RESET_ALL}" + " " * pad_right
                else:
                    row += f"{cell:^{col_width}}"
            
            print(row)
    
    # Display detailed status of all agents
    def _display_agent_status(self):
        print(f"\n{Fore.CYAN}Agent Status:{Style.RESET_ALL}")
        for agent_id, agent in self.agents.items():
            status = agent.get_status()
            target_dist = agent.distance_to_target()
            
            # Determine task phase
            if agent.carrying_box:
                task_phase = "📦→🎯 (Delivering) [@]"
                target_type = "delivery"
            elif agent.target_position:
                task_phase = "🚶→📦 (Pickup) [A]"
                target_type = "pickup"
            else:
                task_phase = "✅ (Complete)"
                target_type = "none"
            
            status_color = Fore.GREEN if status['position'] == status['target'] else Fore.WHITE
            
            print(f"{status_color}Agent {agent_id}: {status['position']} → {status['target']} (dist: {target_dist:.0f}){Style.RESET_ALL}")
            print(f"  {task_phase}")
            
            if status['carrying_box']:
                print(f"  📦 Carrying box {status['box_id']}")
            
            if status['is_waiting']:
                print(f"  ⏳ Waiting {status['wait_turns_remaining']} more turns")
            
            if status['planned_path'] and len(status['planned_path']) > 1:
                print(f"  🗺️  Path: {status['planned_path'][:5]}{'...' if len(status['planned_path']) > 5 else ''}")
    
    # Log current simulation state
    def _log_turn_state(self, event_type: str):
        if not self.log_enabled or not self.logger:
            return
        
        # Collect agent states including executed_path
        agent_states = {}
        for agent_id, agent in self.agents.items():
            status = agent.get_status()
            agent_states[str(agent_id)] = {
                'position': list(status['position']) if status['position'] else None,
                'target_position': list(status['target']) if status['target'] else None,
                'planned_path': [list(pos) for pos in status['planned_path']] if status['planned_path'] else [],
                'executed_path': [list(pos) for pos in status.get('executed_path', [])] if status.get('executed_path') else [],
                'is_waiting': status.get('is_waiting', False),
                'wait_turns_remaining': status.get('wait_turns_remaining', 0),
                'has_negotiated_path': getattr(agent, '_has_negotiated_path', False),
                'carrying_box': status.get('carrying_box', False),
                'box_id': status.get('box_id'),
                'priority': status.get('priority', 0)
            }
        
        # Collect map state
        map_state = {
            'boxes': {str(k): list(v) for k, v in self.warehouse_map.boxes.items()},
            'targets': {str(k): list(v) for k, v in self.warehouse_map.targets.items()},
            'dimensions': [self.warehouse_map.width, self.warehouse_map.height]
        }
        
        # Get negotiation data if this was a negotiation turn
        negotiation_data = self._current_negotiation_data if event_type == 'negotiation' else None
        
        # Log the turn
        self.logger.log_turn(
            turn_num=self.current_turn,
            agent_states=agent_states,
            map_state=map_state,
            negotiation_data=negotiation_data
        )
        
        # Clear negotiation data after logging
        self._current_negotiation_data = None
    
    # Display performance metrics in console
    def _display_performance_metrics(self, metrics: Dict):
        print(f"\n{Fore.CYAN}{'=' * 60}")
        try:
            print(f"{Fore.CYAN}📊 PERFORMANCE METRICS")
            print(f"{'=' * 60}{Style.RESET_ALL}")
        except UnicodeEncodeError:
            print(f"{Fore.CYAN}{'=' * 60}")
            print(f"{Fore.CYAN}[PERFORMANCE METRICS]")
            print(f"{'=' * 60}{Style.RESET_ALL}")
        
        # Success Rate
        success_rate = metrics.get('cooperative_success_rate', 0)
        print(f"{Fore.GREEN}✓ Cooperative Success Rate: {success_rate}%{Style.RESET_ALL}")
        
        # Makespan
        makespan = metrics.get('makespan_seconds', 0)
        print(f"{Fore.YELLOW}⏱️  Makespan: {makespan} seconds{Style.RESET_ALL}")
        
        # Collisions
        collision_count = metrics.get('total_collisions', 0)
        collision_rate = metrics.get('collision_rate', 0)
        print(f"{Fore.RED}💥 Collisions: {collision_count} (rate: {collision_rate:.3f}/turn){Style.RESET_ALL}")
        
        # Path Efficiency
        path_eff = metrics.get('path_efficiency', 0)
        print(f"{Fore.CYAN}🗺️  Path Efficiency: {path_eff}%{Style.RESET_ALL}")
        
        # Token Cost
        tokens = metrics.get('total_tokens_used', 0)
        print(f"{Fore.MAGENTA}💰 Token Cost: {tokens} tokens{Style.RESET_ALL}")
        
        # Conflict Resolution Time
        res_time = metrics.get('avg_conflict_resolution_time_ms', 0)
        print(f"{Fore.BLUE}⚡ Avg Conflict Resolution Time: {res_time:.2f}ms{Style.RESET_ALL}")
        
        # Additional context
        total_turns = metrics.get('total_turns', 0)
        total_negotiations = metrics.get('total_negotiations', 0)
        print(f"\n{Fore.WHITE}Summary: {total_turns} turns, {total_negotiations} negotiations{Style.RESET_ALL}")
        
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}\n")
    
    # Calculate performance metrics after simulation completion, including success rate, makespan, conflict resolution time etc
    def calculate_performance_metrics(self) -> Dict:
        # Calculate makespan in seconds
        makespan_seconds = 0
        if self.simulation_start_time:
            makespan_seconds = time.time() - self.simulation_start_time
        
        # Calculate success rate
        cooperative_success_rate = (self.successful_deliveries / len(self.agents)) * 100 if self.agents else 0
        
        # Calculate collision rate (per turn)
        turns_with_negotiations = len(self.negotiation_times)
        total_turns = max(self.current_turn, 1)  # Avoid division by zero
        collision_rate = self.collision_count / total_turns if total_turns > 0 else 0
        
        # Calculate path efficiency
        # Optimal path would be straight line distance from start to end
        total_actual_path = 0
        total_optimal_path = 0
        
        for agent_id, agent in self.agents.items():
            if agent_id in self.agent_paths and len(self.agent_paths[agent_id]) > 0:
                # Actual path length
                actual_path = len(self.agent_paths[agent_id]) - 1  # -1 because we count edges, not nodes
                actual_path = max(actual_path, 0)
                total_actual_path += actual_path
                
                # Optimal path (Manhattan distance from start to end)
                start_pos = self.initial_agent_positions.get(agent_id, self.agent_paths[agent_id][0])
                end_pos = self.agent_paths[agent_id][-1]
                
                optimal_distance = abs(start_pos[0] - end_pos[0]) + abs(start_pos[1] - end_pos[1])
                total_optimal_path += optimal_distance
        
        path_efficiency = (total_optimal_path / total_actual_path * 100) if total_actual_path > 0 else 100
        path_efficiency = min(path_efficiency, 100)  # Cap at 100%
        
        # Calculate average conflict resolution time
        avg_resolution_time_ms = 0
        if self.negotiation_times:
            total_negotiation_time = sum(
                (end_time - start_time) * 1000 
                for start_time, end_time in self.negotiation_times
            )
            avg_resolution_time_ms = total_negotiation_time / len(self.negotiation_times)
        
        # Get token usage from the central negotiator's client
        token_usage = 0
        if hasattr(self.central_negotiator, 'client') and hasattr(self.central_negotiator.client, 'total_tokens_used'):
            token_usage = self.central_negotiator.client.total_tokens_used
        
        return {
            'cooperative_success_rate': round(cooperative_success_rate, 2),
            'makespan_seconds': round(makespan_seconds, 2),
            'collision_rate': round(collision_rate, 3),
            'path_efficiency': round(path_efficiency, 2),
            'total_tokens_used': token_usage,
            'avg_conflict_resolution_time_ms': round(avg_resolution_time_ms, 2),
            'total_turns': total_turns,
            'total_negotiations': turns_with_negotiations,
            'total_collisions': self.collision_count
        }
    
    # Save simulation log
    def save_simulation_log(self, filename: Optional[str] = None):
        if not self.log_enabled or not self.logger:
            return None
        
        # Calculate and pass performance metrics to logger
        metrics = self.calculate_performance_metrics()
        
        # Finalize and save log with metrics
        log_path = self.logger.finalize(performance_metrics=metrics)
        
        # Return metrics for display
        return log_path, metrics
    
    # Save simulation log to a specific directory with custom filename
    def save_simulation_log_to_path(self, output_dir: str, filename: str) -> Optional[Tuple[str, Dict]]:
        if not self.log_enabled or not self.logger:
            return None
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics()
        
        # Manually save to specified path instead of default
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, filename)
        
        # Calculate summary for logger
        turns = self.logger.log_data['turns']
        negotiation_turns = [t for t in turns if t.get('type') == 'negotiation']
        routine_turns = [t for t in turns if t.get('type') == 'routine']
        hmas2_metrics = self.logger._calculate_hmas2_metrics(negotiation_turns)
        
        self.logger.log_data['summary'] = {
            'total_turns': len(turns),
            'routine_turns': len(routine_turns),
            'negotiation_turns': len(negotiation_turns),
            'total_conflicts': len(negotiation_turns),
            'hmas2_metrics': hmas2_metrics,
            'performance_metrics': metrics,
            'completion_timestamp': datetime.now().isoformat()
        }
        
        # Save to specified path
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.logger.log_data, f, indent=2, default=str)
        
        self.logger._unsaved_data = False
        return log_path, metrics
    
    # Reset token usage counter in the central negotiator's client
    def reset_token_usage(self):
        if hasattr(self.central_negotiator, 'client') and hasattr(self.central_negotiator.client, 'reset_token_usage'):
            self.central_negotiator.client.reset_token_usage()

    # Run interactive simulation with user input for each step
    def run_interactive_simulation(self):
        self.initialize_simulation()
        
        print(f"\n{Fore.CYAN}🚀 Starting Interactive Simulation{Style.RESET_ALL}")
        
        auto_mode = False
        
        while self.run_simulation_step():
            if not auto_mode:
                user_input = input("\nPress Enter for next step (or command): ").strip().lower()
                
                if user_input == 'q':
                    print("Simulation terminated by user.")
                    break
                elif user_input == 'auto':
                    auto_mode = True
                    print("Switching to auto mode...")
            else:
                time.sleep(1)  # Auto delay
        
        # Save log when simulation ends
        result = self.save_simulation_log()
        
        # Display performance metrics
        if result and isinstance(result, tuple):
            log_path, metrics = result
            self._display_performance_metrics(metrics)
        
        print(f"\n{Fore.GREEN}Simulation completed in {self.current_turn} turns!{Style.RESET_ALL}")
