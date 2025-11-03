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
        
        # Logging - Use comprehensive format compatible with visualization
        self.log_enabled = os.getenv('LOG_SIMULATION', 'true').lower() == 'true'
        self.simulation_log = {
            'scenario': {},
            'turns': [],
            'summary': {}
        }
        
    def initialize_simulation(self):
        """Initialize a new simulation"""
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
        
        # Initial pathfinding
        self._plan_initial_paths()
        
        # Initialize scenario data for logging
        if self.log_enabled:
            self.simulation_log['scenario'] = {
                'type': 'interactive_simulation',
                'map_size': [self.warehouse_map.width, self.warehouse_map.height],
                'initial_agents': {str(k): list(v) for k, v in self.warehouse_map.agents.items()},
                'initial_targets': {str(k): list(v) for k, v in self.warehouse_map.targets.items()},
                'grid': self.warehouse_map.grid.tolist(),  # Include grid data for visualization
                'timestamp': datetime.now().isoformat()
            }
        
        # Log initial state
        self._log_turn_state("SIMULATION_START")
        
        print(f"{Fore.GREEN}Simulation initialized successfully!{Style.RESET_ALL}")
        self.display_map()
    
    def _plan_initial_paths(self):
        """Plan initial paths for all agents"""
        print("Planning initial paths for all agents...")
        
        for agent_id, agent in self.agents.items():
            map_state = self.warehouse_map.get_state_dict()
            path = agent.plan_path(map_state)
    
    def detect_stagnation_conflicts(self) -> Dict:
        """Detect when agents have failed moves for multiple turns (actual stagnation, not intentional waiting)"""
        
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
    
    def detect_move_failure_deadlocks(self, planned_moves: Dict) -> Dict:
        """Detect agents with too many consecutive failed moves"""
        
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
    
    def _force_deadlock_negotiation(self, stuck_agents: List[int], planned_moves: Dict):
        """Force negotiation when agents are stuck"""
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
    
    def run_simulation_step(self) -> bool:
        """
        Run one step of the simulation with forced conflict detection
        
        Returns:
            bool: True if simulation should continue, False if complete
        """
        if self.simulation_complete or self.current_turn >= self.max_turns:
            return False
        
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
        
        if conflict_info['has_conflicts']:
            print(f"{Fore.RED}CONFLICT DETECTED!{Style.RESET_ALL}")
            print(f"Conflicting agents: {conflict_info['conflicting_agents']}")
            print(f"Conflict points: {conflict_info['conflict_points']}")
            
            # Use Central Negotiator to resolve conflicts
            resolution = self._negotiate_conflicts(conflict_info, forced_moves)
            self._execute_negotiated_actions(resolution)
        else:
            # PHASE 2: No conflicts in forced paths - try normal planning
            print(f"{Fore.GREEN}No conflicts in forced paths. Trying normal planning...{Style.RESET_ALL}")
            
            normal_moves = self._get_normal_planned_moves()
            if normal_moves:
                # Double-check for conflicts in normal paths
                normal_conflict_info = self.conflict_detector.detect_path_conflicts(normal_moves, self.current_turn)
                
                if normal_conflict_info['has_conflicts']:
                    print(f"{Fore.YELLOW}Conflicts found in normal paths - negotiating...{Style.RESET_ALL}")
                    resolution = self._negotiate_conflicts(normal_conflict_info, normal_moves)
                    self._execute_negotiated_actions(resolution)
                else:
                    print(f"{Fore.GREEN}No conflicts detected. Executing planned moves...{Style.RESET_ALL}")
                    self._execute_planned_moves(normal_moves)
                    
                    # PHASE 3: Check for deadlock after move execution
                    deadlock_conflict = self.detect_move_failure_deadlocks(normal_moves)
                    if deadlock_conflict['has_conflicts']:
                        print(f"🔥 DEADLOCK AFTER MOVES! Forcing resolution...")
                        self._force_deadlock_negotiation(deadlock_conflict['conflicting_agents'], normal_moves)
            else:
                print(f"{Fore.YELLOW}No agents can plan paths - all waiting...{Style.RESET_ALL}")
        
        # Update map with new agent positions
        self._update_map_state()
        
        # Check if simulation is complete
        if self._check_completion():
            print(f"{Fore.GREEN}🎉 All agents reached their targets! Simulation complete!{Style.RESET_ALL}")
            self.simulation_complete = True
        
        # Log turn state
        self._log_turn_state("TURN_COMPLETE")
        
        # Display current state
        self.display_map()
        self._display_agent_status()
        
        self.current_turn += 1
        return not self.simulation_complete
    
    def _get_planned_moves(self) -> Dict[int, List[Tuple[int, int]]]:
        """Get planned moves for all active agents (legacy method - now calls normal planning)"""
        return self._get_normal_planned_moves()
    
    def _get_forced_planned_moves(self) -> Dict[int, List[Tuple[int, int]]]:
        """Get forced planned moves (ignoring other agents) for conflict detection"""
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
    
    def _get_normal_planned_moves(self) -> Dict[int, List[Tuple[int, int]]]:
        """Get normal planned moves (avoiding other agents)"""
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
    
    def _plan_forced_path(self, agent, map_state: Dict) -> List[Tuple[int, int]]:
        """Plan path ignoring other agents - for conflict detection"""
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
    
    def _plan_normal_path(self, agent, map_state: Dict) -> List[Tuple[int, int]]:
        """Plan path avoiding other agents - for actual movement"""
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
    
    def _negotiate_conflicts(self, conflict_info: Dict, planned_moves: Dict) -> Dict:
        """Use Central Negotiator to resolve conflicts"""
        print("🤖 Initiating LLM-based conflict negotiation...")
        
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
        # New: returns Tuple[Dict, List[Dict]] = (final_actions, refinement_history)
        negotiation_result = self.central_negotiator.negotiate_path_conflict(
            conflict_data, 
            agent_validators=agent_validators
        )
        
        # Handle both old format (just Dict) and new format (Tuple)
        if isinstance(negotiation_result, tuple):
            resolution, refinement_history = negotiation_result
        else:
            resolution = negotiation_result
            refinement_history = []
        
        # Check for deadlock (empty dict means turn should be skipped)
        if not resolution:  # Empty dict
            print(f"🛑 Negotiation deadlock - turn skipped (no movement)")
            return {
                'agent_actions': {},
                'resolution': 'deadlock_skipped',
                'reasoning': 'Negotiation failed to resolve after max refinement iterations',
                'refinement_history': refinement_history
            }
        
        print(f"🤖 Negotiation complete: {resolution.get('resolution', 'unknown')}")
        print(f"📝 Reasoning: {resolution.get('reasoning', 'No reasoning provided')}")
        
        # Add refinement history if available
        if refinement_history:
            resolution['refinement_history'] = refinement_history
        
        return resolution
    
    def _execute_negotiated_actions(self, resolution: Dict):
        """Execute actions determined by negotiation"""
        # Handle both formats: response with 'agent_actions' key or agent IDs directly as keys
        if 'agent_actions' in resolution:
            agent_actions = resolution.get('agent_actions', {})
        else:
            # Agent IDs are top-level keys, but filter out non-numeric keys (metadata)
            agent_actions = {k: v for k, v in resolution.items() 
                           if isinstance(k, (int, str)) and (isinstance(k, int) or k.isdigit())}
        
        # OPTIMIZATION: Mark all agents as HMAS-2 pre-validated before executing
        # The central negotiator already validated these paths with agents validators,
        # so we skip redundant validation in execute_negotiated_action
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
                
                # CRITICAL FIX: Update agent's planned path with negotiated path
                negotiated_path = action_data.get('path', [])
                if negotiated_path and len(negotiated_path) > 0:
                    # Convert path elements to tuples for consistency
                    # Store the path as-is from the LLM response
                    # If it includes current position, agent will "wait" on first execution
                    # If it doesn't include current position, agent will move immediately
                    updated_path = [tuple(pos) if isinstance(pos, (list, tuple)) else pos for pos in negotiated_path]
                    agent.set_path(updated_path)
                    
                    # IMPORTANT: Mark agent as having a negotiated path to preserve it
                    agent._has_negotiated_path = True
                
                success = agent.execute_negotiated_action(action_data, map_state)
                
                if success:
                    # IMPORTANT: After successful move, update the negotiated path by removing the first step
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
    
    def _execute_planned_moves(self, planned_moves: Dict):
        """Execute planned moves without conflicts"""
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
                
                # CRITICAL FIX: Handle "waiting in place" moves (when next_pos == current_pos)
                if next_pos == agent.position:
                    # This is a "wait" move - agent should stay in current position
                    print(f"⏸️  Agent {agent_id}: Waiting at {agent.position}")
                    success = True  # Waiting is always successful
                else:
                    # Normal move to different position
                    print(f"🔍 Agent {agent_id}: Attempting move from {agent.position} to {next_pos}")
                    map_state = self.warehouse_map.get_state_dict()
                    success = agent.move_to(next_pos, map_state)
                
                if success:
                    if next_pos != agent.position:
                        print(f"✅ Agent {agent_id}: Moved to {next_pos}")
                    
                    # Reset failure count on successful move
                    self.failed_move_counts[agent_id] = 0
                    
                    # IMPORTANT: Clear failed move history on successful move or wait
                    if agent_id in self.agent_failed_move_history:
                        self.agent_failed_move_history[agent_id] = []
                    
                    # CRITICAL FIX: If agent has a negotiated path, advance it properly
                    if (hasattr(agent, '_has_negotiated_path') and 
                        getattr(agent, '_has_negotiated_path', False) and 
                        agent.planned_path and 
                        len(agent.planned_path) > 1):
                        
                        print(f"🔍 Agent {agent_id}: Before path advance - Path: {agent.planned_path[:3]}... (len: {len(agent.planned_path)})")
                        print(f"🔍 Agent {agent_id}: Current position: {agent.position}")
                        
                        # Remove the first step (current position) from negotiated path
                        agent.planned_path = agent.planned_path[1:]
                        print(f"🔄 Agent {agent_id}: Advanced negotiated path, {len(agent.planned_path)} steps remaining")
                        print(f"🔍 Agent {agent_id}: After path advance - Path: {agent.planned_path[:3]}... (len: {len(agent.planned_path)})")
                        
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
    
    def _check_box_pickup(self, agent_id: int):
        """Check if agent can pick up a box at current position"""
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
    
    def _check_box_delivery(self, agent_id: int):
        """Check if agent can deliver box at current position"""
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
                        
                        # Clear failed move history when task is completed
                        if agent_id in self.agent_failed_move_history:
                            self.agent_failed_move_history[agent_id] = []
                            print(f"🧹 Agent {agent_id}: Cleared failed move history (task completed)")
    
    def _update_map_state(self):
        """Update warehouse map with current agent positions"""
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
        
        # Place agents at new positions (this will overwrite targets if agent is on them)
        for agent_id, agent in self.agents.items():
            x, y = agent.position
            if agent.carrying_box:
                self.warehouse_map.grid[y, x] = CellType.AGENT_WITH_BOX.value
            else:
                self.warehouse_map.grid[y, x] = CellType.AGENT.value
            
            # Update warehouse map's agent tracking
            self.warehouse_map.agents[agent_id] = agent.position
    
    def _check_completion(self) -> bool:
        """Check if all warehouse tasks are completed (all boxes delivered)"""
        # Check if all boxes have been delivered (removed from the map)
        if self.warehouse_map.boxes:
            return False  # Still boxes to be delivered
        
        # Alternative check: all agents have no active targets (completed their tasks)
        for agent in self.agents.values():
            if agent.target_position is not None:
                return False  # Agent still has work to do
                
        return True
    
    def display_map(self):
        """Display the current warehouse map with colors"""
        print(f"\n{Fore.CYAN}Current Warehouse State:{Style.RESET_ALL}")
        
        # Add column numbers
        header = "   " + " ".join([str(i) for i in range(self.width)])
        print(header)
        
        for y in range(self.height):
            row = f"{y}: "
            for x in range(self.width):
                cell = self.warehouse_map.grid[y, x]
                
                # Color coding
                if cell == CellType.AGENT.value:
                    row += f"{Fore.BLUE}{cell}{Style.RESET_ALL} "
                elif cell == CellType.AGENT_WITH_BOX.value:
                    row += f"{Fore.MAGENTA}{cell}{Style.RESET_ALL} "
                elif cell == CellType.BOX.value:
                    row += f"{Fore.YELLOW}{cell}{Style.RESET_ALL} "
                elif cell == CellType.TARGET.value:
                    row += f"{Fore.GREEN}{cell}{Style.RESET_ALL} "
                elif cell == CellType.WALL.value:
                    row += f"{Back.BLACK}{cell}{Style.RESET_ALL} "
                else:
                    row += f"{cell} "
            
            print(row)
    
    def _display_agent_status(self):
        """Display detailed status of all agents"""
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
    
    def _log_turn_state(self, event_type: str):
        """Log current simulation state in comprehensive format"""
        if not self.log_enabled:
            return
        
        # Collect agent states
        agent_states = {}
        for agent_id, agent in self.agents.items():
            status = agent.get_status()
            agent_states[str(agent_id)] = {
                'position': list(status['position']) if status['position'] else None,
                'target_position': list(status['target']) if status['target'] else None,
                'planned_path': [list(pos) for pos in status['planned_path']] if status['planned_path'] else [],
                'is_waiting': status.get('is_waiting', False),
                'wait_turns_remaining': status.get('wait_turns_remaining', 0),
                'has_negotiated_path': getattr(agent, '_has_negotiated_path', False),
                'is_at_target': status.get('is_at_target', False),
                'carrying_box': status.get('carrying_box', False),
                'box_id': status.get('box_id'),
                'priority': status.get('priority', 0),
                'current_action': status.get('current_action', 'idle')
            }
        
        # Create comprehensive turn log entry
        log_entry = {
            'turn': self.current_turn,
            'timestamp': datetime.now().isoformat(),
            'type': 'negotiation' if event_type == 'negotiation' else 'routine',
            'agent_states': agent_states,
            'map_state': {
                'boxes': {str(k): list(v) for k, v in self.warehouse_map.boxes.items()},  # Include current box positions
                'targets': {str(k): list(v) for k, v in self.warehouse_map.targets.items()},
                'dimensions': [self.warehouse_map.width, self.warehouse_map.height]
            },
            'conflicts_detected': event_type == 'conflict',
            'negotiation_occurred': event_type == 'negotiation',
            'results': {
                'agent_states_after': {},  # Could be populated post-move
                'map_state_after': {},
                'simulation_continued': not self.simulation_complete
            }
        }
        
        self.simulation_log['turns'].append(log_entry)
    
    def save_simulation_log(self, filename: Optional[str] = None):
        """Save comprehensive simulation log to file in visualization-compatible format"""
        if not self.simulation_log['turns']:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_log_{timestamp}.json"
        
        # Populate scenario information if not already set
        if not self.simulation_log['scenario']:
            self.simulation_log['scenario'] = {
                'type': 'interactive_simulation',
                'map_size': [self.warehouse_map.width, self.warehouse_map.height],
                'initial_agents': {str(k): list(v) for k, v in self.warehouse_map.agents.items()},
                'initial_targets': {str(k): list(v) for k, v in self.warehouse_map.targets.items()},
                'grid': self.warehouse_map.grid.tolist(),  # Include grid data for visualization
                'timestamp': datetime.now().isoformat()
            }
        
        # Populate summary information
        negotiation_turns = len([t for t in self.simulation_log['turns'] if t['type'] == 'negotiation'])
        routine_turns = len([t for t in self.simulation_log['turns'] if t['type'] == 'routine'])
        
        self.simulation_log['summary'] = {
            'total_turns': len(self.simulation_log['turns']),
            'total_conflicts': len([t for t in self.simulation_log['turns'] if t.get('conflicts_detected', False)]),
            'total_negotiations': negotiation_turns,
            'completion_timestamp': datetime.now().isoformat(),
            'negotiation_turns': negotiation_turns,
            'routine_turns': routine_turns
        }
        
        log_path = os.path.join("logs", filename)
        os.makedirs("logs", exist_ok=True)
        
        with open(log_path, 'w') as f:
            json.dump(self.simulation_log, f, indent=2)
        
        print(f"Comprehensive simulation log saved to: {log_path}")
        print(f"📊 Summary: {len(self.simulation_log['turns'])} turns, {negotiation_turns} negotiations")
        return log_path
    
    def run_interactive_simulation(self):
        """Run simulation with step-by-step user input"""
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
        self.save_simulation_log()
        print(f"\n{Fore.GREEN}Simulation completed in {self.current_turn} turns!{Style.RESET_ALL}")
