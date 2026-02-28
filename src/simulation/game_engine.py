"""
Main Game Engine for Multi-Robot Warehouse Simulation
"""

import json
import os
import queue
import random
import threading
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

        # Simulation mode: 'turn_based' (default), 'async', or 'lifelong'
        self.simulation_mode = 'turn_based'

        # Async mode: agents blocked waiting for LLM resolution
        self._pending_resolution_agents: set = set()

        # Async mode: per-agent last-known safe position (before conflict)
        self._conflict_hold_positions: Dict[int, Tuple[int, int]] = {}

        # Lifelong mode: per-task completion records and task start tracking
        self._lifelong_task_completions: List[Dict] = []
        self._agent_task_start_turns: Dict[int, int] = {}

        # Async mode: live display window and tick counter
        self._async_tick: int = 0
        self._async_fig = None
        self._async_ax = None

        # Truly-async execution: background LLM negotiation infrastructure
        # tick_interval controls display refresh rate (seconds)
        self._async_tick_interval: float = 0.3
        # Queue for results returned by background negotiation threads
        self._negotiation_result_queue: queue.Queue = queue.Queue()
        # frozensets of agent-id groups whose negotiation is currently in flight
        self._active_negotiations: set = set()
    
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
            self._agent_task_start_turns[agent_id] = 0
        
        # Start tracking simulation time
        self.simulation_start_time = time.time()
        
        # Check for agents already on their boxes and auto-pickup
        for agent_id, agent in self.agents.items():
            if not agent.carrying_box:
                box_id = agent_id
                if box_id in self.warehouse_map.boxes:
                    box_pos = self.warehouse_map.boxes[box_id]
                    if agent.position == box_pos:
                        print(f"🎯 Agent {agent_id} spawned on box {box_id}, picking up immediately...")
                        success = self.warehouse_map.pickup_box(agent_id, box_id)
                        if success:
                            agent.pickup_box(box_id)
                            # Set target to delivery location
                            target_id = self.warehouse_map.agent_goals.get(agent_id)
                            if target_id is not None and target_id in self.warehouse_map.targets:
                                target_pos = self.warehouse_map.targets[target_id]
                                agent.set_target(target_pos)
                                print(f"📦 Agent {agent_id}: Ready to deliver to {target_pos}")
        
        # Initial pathfinding
        self._plan_initial_paths()
        
        # Initialize unified logger with scenario data
        if self.log_enabled and self.logger:
            self.logger.initialize({
                'type': 'interactive_simulation',
                'simulation_mode': self.simulation_mode,
                'map_size': [self.warehouse_map.width, self.warehouse_map.height],
                'grid': self.warehouse_map.grid.tolist(),
                'initial_agents': {str(k): list(v) for k, v in self.warehouse_map.agents.items()},
                'initial_targets': {str(k): list(v) for k, v in self.warehouse_map.targets.items()},
                'initial_boxes': {str(k): list(v) for k, v in self.warehouse_map.boxes.items()},
                'agent_goals': {str(k): v for k, v in self.warehouse_map.agent_goals.items()}
            })

        print(f"{Fore.GREEN}Simulation initialized successfully!{Style.RESET_ALL}")
        self._update_map_state()

        if self.simulation_mode in ('async', 'lifelong'):
            # Async / lifelong: open live display window and use path-based logging
            self._setup_async_display()
            # Log initial positions to agent paths
            if self.log_enabled and self.logger:
                for agent_id, agent in self.agents.items():
                    self.logger.append_agent_path(agent_id, agent.position)
            self._update_async_display()
        else:
            # Turn-based: normal terminal display
            self._log_turn_state("SIMULATION_START")
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
                    'failed_move_count': len(self.agent_failed_move_history.get(aid, [])),
                    'failed_move_history': self.agent_failed_move_history.get(aid, [])
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
                        'failure_count': self.failed_move_counts.get(aid, 0),
                        'failed_move_history': self.agent_failed_move_history.get(aid, [])
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
            resolution, _ = self._negotiate_conflicts(conflict_data, planned_moves)
            self._execute_negotiated_actions(resolution)
            
            # Reset failure counts after forced negotiation
            for agent_id in stuck_agents:
                self.failed_move_counts[agent_id] = 0
                # Also clear failed move history
                if agent_id in self.agent_failed_move_history:
                    self.agent_failed_move_history[agent_id] = []
                print(f"🔄 Agent {agent_id}: Reset failure count and move history after deadlock negotiation")
    
    # Run one step of the simulation — dispatches to mode-specific implementation
    def run_simulation_step(self) -> bool:
        if self.simulation_mode in ('async', 'lifelong'):
            return self._run_async_step()
        return self._run_turn_based_step()

    # Run one turn-based simulation step (original logic, zero behavioural change)
    def _run_turn_based_step(self) -> bool:
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
            resolution, _ = self._negotiate_conflicts(stagnation_conflict, {})
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
            resolution, _ = self._negotiate_conflicts(conflict_info, forced_moves)
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
                    resolution, _ = self._negotiate_conflicts(normal_conflict_info, normal_moves)
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

    def _run_async_step(self) -> bool:
        """Run one truly-async simulation tick.

        Non-conflicting agents always advance each tick.  When a conflict is
        detected, *only* the involved agents are frozen while a background
        daemon thread calls the LLM.  All other agents keep moving and the
        matplotlib window stays live throughout the negotiation.
        """
        # ── Guard checks ──────────────────────────────────────────────────
        if self.stop_requested:
            return False

        if self.timeout_seconds > 0 and self.simulation_start_time:
            if time.time() - self.simulation_start_time >= self.timeout_seconds:
                if not self.silent_mode:
                    print(f"\n{Fore.YELLOW}⏱️  Time limit ({self.timeout_seconds}s) reached!{Style.RESET_ALL}")
                return False

        if self.simulation_complete or self.current_turn >= self.max_turns:
            return False

        if not self.silent_mode:
            print(f"\n{Fore.YELLOW}=== TICK {self._async_tick + 1} (ASYNC) ==={Style.RESET_ALL}")

        # ── Step 1: Apply completed background-negotiation results ────────
        self._apply_pending_negotiation_results()

        # ── Step 2: Tick all agents ───────────────────────────────────────
        for agent in self.agents.values():
            agent.update_turn()
        for agent_id, agent in self.agents.items():
            if hasattr(agent, '_hmas2_validated'):
                setattr(agent, '_hmas2_validated', False)

        for agent_id in list(self.agents.keys()):
            self._check_box_delivery(agent_id)

        # ── Step 3: Classify agents: frozen vs. free ─────────────────────
        free_agent_ids: set = set(self.agents.keys()) - self._pending_resolution_agents

        # ── Step 4: Stagnation check (free agents only) ───────────────────
        stagnant_free: set = set()
        stagnation_conflict = self.detect_stagnation_conflicts()
        if stagnation_conflict['has_conflicts']:
            stagnant_free = set(stagnation_conflict['conflicting_agents']) & free_agent_ids
            if stagnant_free:
                filtered = dict(stagnation_conflict)
                filtered['conflicting_agents'] = list(stagnant_free)
                filtered['agents'] = [
                    a for a in stagnation_conflict.get('agents', [])
                    if a['id'] in stagnant_free
                ]
                conflict_key = frozenset(stagnant_free)
                if conflict_key not in self._active_negotiations:
                    if not self.silent_mode:
                        print(f"🚫 STAGNATION! Background resolve for agents: {stagnant_free}")
                    for aid in stagnant_free:
                        self._conflict_hold_positions[aid] = self.agents[aid].position
                        self.agent_position_history.pop(aid, None)
                        self.agent_failed_move_history.pop(aid, None)
                    self._pending_resolution_agents.update(stagnant_free)
                    self._start_background_negotiation(conflict_key, filtered, {})

        # ── Step 5: Forced moves for active free agents ───────────────────
        active_free_ids = free_agent_ids - stagnant_free
        forced_moves: Dict[int, List] = {}
        for agent_id in active_free_ids:
            agent = self.agents[agent_id]
            if not agent.is_waiting and agent.target_position:
                if (hasattr(agent, '_has_negotiated_path') and
                        getattr(agent, '_has_negotiated_path', False) and
                        agent.planned_path):
                    forced_moves[agent_id] = agent.planned_path.copy()
                else:
                    map_state = self.warehouse_map.get_state_dict()
                    forced_path = self._plan_forced_path(agent, map_state)
                    if forced_path:
                        forced_moves[agent_id] = forced_path

        conflicting_ids: set = set()

        if forced_moves:
            # ── Step 6: Detect primary conflicts ─────────────────────────
            conflict_info = self.conflict_detector.detect_path_conflicts(
                forced_moves, self.current_turn)

            if conflict_info['has_conflicts']:
                conflicting_ids = set(conflict_info['conflicting_agents'])
                conflict_key = frozenset(conflicting_ids)
                if conflict_key not in self._active_negotiations:
                    if not self.silent_mode:
                        print(f"{Fore.RED}ASYNC CONFLICT! Agents: {conflicting_ids} "
                              f"→ background thread{Style.RESET_ALL}")
                    self.collision_count += len(conflict_info['conflict_points'])
                    for aid in conflicting_ids:
                        self._conflict_hold_positions[aid] = self.agents[aid].position
                    self._pending_resolution_agents.update(conflicting_ids)
                    self._start_background_negotiation(
                        conflict_key, conflict_info, forced_moves)

            # ── Step 7: Move non-conflicting free agents ──────────────────
            non_conflicting_ids = active_free_ids - conflicting_ids
            normal_moves: Dict[int, List] = {}
            for agent_id in non_conflicting_ids:
                agent = self.agents[agent_id]
                if not agent.is_waiting and agent.target_position:
                    if (hasattr(agent, '_has_negotiated_path') and
                            getattr(agent, '_has_negotiated_path', False) and
                            agent.planned_path and len(agent.planned_path) > 1):
                        normal_moves[agent_id] = agent.planned_path.copy()
                    else:
                        needs_replan = (
                            not agent.planned_path or
                            (agent.planned_path and
                             agent.planned_path[-1] != agent.target_position)
                        )
                        if needs_replan:
                            map_state = self.warehouse_map.get_state_dict()
                            agent.planned_path = self._plan_normal_path(agent, map_state)
                        if agent.planned_path:
                            normal_moves[agent_id] = agent.planned_path.copy()

            if normal_moves:
                normal_conflict = self.conflict_detector.detect_path_conflicts(
                    normal_moves, self.current_turn)
                if normal_conflict['has_conflicts']:
                    sec_ids = set(normal_conflict['conflicting_agents'])
                    sec_key = frozenset(sec_ids)
                    if sec_key not in self._active_negotiations:
                        if not self.silent_mode:
                            print(f"{Fore.YELLOW}Secondary conflict: {sec_ids} "
                                  f"→ background thread{Style.RESET_ALL}")
                        self.collision_count += len(normal_conflict['conflict_points'])
                        for aid in sec_ids:
                            self._conflict_hold_positions[aid] = self.agents[aid].position
                        self._pending_resolution_agents.update(sec_ids)
                        self._start_background_negotiation(
                            sec_key, normal_conflict, normal_moves)
                    # Execute only truly conflict-free agents
                    free_normal = {k: v for k, v in normal_moves.items()
                                   if k not in sec_ids}
                    if free_normal:
                        self._execute_planned_moves(free_normal)
                        deadlock = self.detect_move_failure_deadlocks(free_normal)
                        if deadlock['has_conflicts']:
                            dl_ids = set(deadlock['conflicting_agents'])
                            dl_key = frozenset(dl_ids)
                            if dl_key not in self._active_negotiations:
                                for aid in dl_ids:
                                    self._conflict_hold_positions[aid] = self.agents[aid].position
                                self._pending_resolution_agents.update(dl_ids)
                                self._start_background_negotiation(dl_key, deadlock, free_normal)
                else:
                    if not self.silent_mode:
                        print(f"{Fore.GREEN}No conflicts. Executing moves...{Style.RESET_ALL}")
                    self._execute_planned_moves(normal_moves)
                    deadlock = self.detect_move_failure_deadlocks(normal_moves)
                    if deadlock['has_conflicts']:
                        dl_ids = set(deadlock['conflicting_agents'])
                        dl_key = frozenset(dl_ids)
                        if dl_key not in self._active_negotiations:
                            for aid in dl_ids:
                                self._conflict_hold_positions[aid] = self.agents[aid].position
                            self._pending_resolution_agents.update(dl_ids)
                            self._start_background_negotiation(dl_key, deadlock, normal_moves)
        else:
            if not self.silent_mode:
                print("No free agents with targets.")

        # ── Step 8: Update map, log path positions, refresh display ───────
        self._update_map_state()
        self._async_tick += 1
        self.current_turn += 1
        self._log_async_state(negotiation_occurred=False)
        self._update_async_display()

        # Lifelong terminates only via timeout
        if self.simulation_mode == 'lifelong':
            return True

        if self._check_completion():
            if not self.silent_mode:
                print(f"{Fore.GREEN}🎉 All agents reached their targets!{Style.RESET_ALL}")
            self.simulation_complete = True
            return False

        return True

    # ------------------------------------------------------------------ #
    #  Truly-async infrastructure: background negotiation threads          #
    # ------------------------------------------------------------------ #

    def _start_background_negotiation(
        self,
        conflict_key: frozenset,
        conflict_info: Dict,
        forced_moves: Dict,
    ) -> None:
        """Fire a daemon thread that calls the LLM without blocking the sim."""
        self._active_negotiations.add(conflict_key)

        # Snapshot the map state now so the background thread works with a
        # consistent picture even as other agents continue to move.
        conflict_info_copy = dict(conflict_info)
        if 'map_state_snapshot' not in conflict_info_copy:
            conflict_info_copy['map_state_snapshot'] = self.warehouse_map.get_state_dict()
        forced_moves_copy = dict(forced_moves)

        def _negotiation_worker() -> None:
            try:
                resolution, neg_data = self._negotiate_conflicts(
                    conflict_info_copy, forced_moves_copy)
            except Exception as exc:
                if not self.silent_mode:
                    print(f"{Fore.RED}❌ Background negotiation error for agents "
                          f"{set(conflict_key)}: {exc}{Style.RESET_ALL}")
                resolution = {'agent_actions': {}, 'resolution': 'error'}
                neg_data = {}
            self._negotiation_result_queue.put({
                'conflict_key': conflict_key,
                'agent_ids': list(conflict_key),
                'resolution': resolution,
                'negotiation_data': neg_data,
            })

        threading.Thread(target=_negotiation_worker, daemon=True).start()

    def _apply_pending_negotiation_results(self) -> None:
        """Drain the result queue and unfreeze agents with their new paths."""
        while True:
            try:
                result = self._negotiation_result_queue.get_nowait()
            except queue.Empty:
                break

            conflict_key: frozenset = result['conflict_key']
            agent_ids: list = result['agent_ids']
            resolution: Dict = result['resolution']
            neg_data = result.get('negotiation_data')

            if not self.silent_mode:
                print(f"{Fore.GREEN}✅ Negotiation done for agents: "
                      f"{set(agent_ids)}{Style.RESET_ALL}")

            self._execute_negotiated_actions(resolution)

            # If the LLM returned a deadlock (no actions), clear the stale conflicting
            # paths so agents can replan from scratch instead of cycling forever.
            if resolution.get('resolution') == 'deadlock_skipped':
                for aid in agent_ids:
                    if aid in self.agents:
                        a = self.agents[aid]
                        if hasattr(a, '_has_negotiated_path'):
                            a._has_negotiated_path = False
                        a.planned_path = []
                if not self.silent_mode:
                    print(f"{Fore.YELLOW}⚠️  Deadlock: cleared paths for {set(agent_ids)}, "
                          f"agents will replan{Style.RESET_ALL}")

            # Unfreeze agents
            self._pending_resolution_agents -= set(agent_ids)
            for aid in agent_ids:
                self._conflict_hold_positions.pop(aid, None)
            self._active_negotiations.discard(conflict_key)

            # Log the negotiation event
            if self.log_enabled and self.logger and neg_data:
                event = {
                    'tick': self._async_tick,
                    'timestamp': datetime.now().isoformat(),
                    'conflicting_agents': agent_ids,
                    'negotiation_data': neg_data,
                    'agent_path_indices': {
                        str(aid): len(
                            self.logger.log_data.get('agent_paths', {})
                            .get(str(aid), [])
                        ) - 1
                        for aid in self.agents
                    },
                }
                self.logger.log_async_negotiation(event)

    def _drain_pending_negotiations(self, timeout: float = 60.0) -> None:
        """Block until all in-flight background negotiations finish or timeout.

        After returning, any negotiations that timed out are discarded from
        `_active_negotiations`.  Their conflicting agents may remain in
        `_pending_resolution_agents`; this is acceptable because the caller
        is about to save logs and shut down.
        """
        deadline = time.time() + timeout
        while self._active_negotiations and time.time() < deadline:
            self._apply_pending_negotiation_results()
            time.sleep(0.1)
        if self._active_negotiations:
            print(f"⚠️  Timed out waiting for background negotiation(s): "
                  f"{self._active_negotiations}")
            self._active_negotiations.clear()
        # Final drain to capture any last-second results
        self._apply_pending_negotiation_results()

    # ------------------------------------------------------------------ #
    #  Async mode helpers: live display + path-based logging               #
    # ------------------------------------------------------------------ #

    def _setup_async_display(self) -> None:
        """Open a non-blocking matplotlib window for live async simulation display."""
        if self.silent_mode:
            return
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            plt.ion()
            self._async_fig, self._async_ax = plt.subplots(
                figsize=(max(8, self.width * 0.6), max(6, self.height * 0.6)),
                num='Async Simulation — Live View'
            )
            try:
                self._async_fig.canvas.manager.set_window_title('Async Simulation — Live View')
            except Exception:
                pass
            print("🖥️  Live async display window opened")
        except Exception as e:
            print(f"⚠️  Could not open async live display: {e}")
            self._async_fig = None
            self._async_ax = None

    def _update_async_display(self) -> None:
        """Redraw the live matplotlib window with the current simulation state."""
        if self.silent_mode or self._async_fig is None:
            return
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            # Fixed colour list — no numpy needed
            _COLOURS = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
            ]

            ax = self._async_ax
            ax.clear()
            ax.set_xlim(-0.5, self.width - 0.5)
            ax.set_ylim(-0.5, self.height - 0.5)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            ax.set_title(
                f"{'Lifelong' if self.simulation_mode == 'lifelong' else 'Async'} "
                f"— Tick {self._async_tick}  "
                f"| Deliveries: {self.successful_deliveries}"
                + (f"  | ⏳ {len(self._pending_resolution_agents)} negotiating"
                   if self._pending_resolution_agents else ""),
                fontsize=10
            )

            grid = self.warehouse_map.grid
            for y in range(self.height):
                for x in range(self.width):
                    cell = grid[y, x]
                    if cell == '#':
                        ax.add_patch(patches.Rectangle(
                            (x - 0.5, y - 0.5), 1, 1,
                            facecolor='#444', alpha=0.85, zorder=1))
                    elif cell == 'T':
                        ax.add_patch(patches.Rectangle(
                            (x - 0.4, y - 0.4), 0.8, 0.8,
                            linewidth=2, edgecolor='red',
                            facecolor='#ffcccc', alpha=0.7, zorder=2))
                        ax.text(x, y, 'T', ha='center', va='center',
                                fontsize=7, color='red', fontweight='bold', zorder=3)
                    elif cell == 'B':
                        ax.add_patch(patches.Rectangle(
                            (x - 0.3, y - 0.3), 0.6, 0.6,
                            linewidth=2, edgecolor='#8B4513',
                            facecolor='#DEB887', alpha=0.85, zorder=2))
                        ax.text(x, y, 'B', ha='center', va='center',
                                fontsize=7, color='#5C3317', fontweight='bold', zorder=3)

            # Agent paths so far + current positions
            for i, (agent_id, agent) in enumerate(self.agents.items()):
                colour = _COLOURS[i % len(_COLOURS)]
                frozen = agent_id in self._pending_resolution_agents

                # Draw recorded path
                if self.log_enabled and self.logger:
                    path = self.logger.log_data.get('agent_paths', {}).get(str(agent_id), [])
                    if len(path) > 1:
                        xs = [p[0] for p in path]
                        ys = [p[1] for p in path]
                        ax.plot(xs, ys, color=colour, linewidth=1.5, alpha=0.4, zorder=4)

                # Current position marker
                x, y = agent.position
                edge = 'red' if frozen else 'black'
                lw = 2.5 if frozen else 1.5
                ax.add_patch(patches.Circle(
                    (x, y), 0.28, facecolor=colour,
                    edgecolor=edge, linewidth=lw, alpha=0.9, zorder=5))
                ax.text(x, y, str(agent_id),
                        ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold', zorder=6)
                if frozen:
                    ax.text(x, y - 0.45, '❄', ha='center', va='top',
                            fontsize=9, zorder=6)

            plt.pause(self._async_tick_interval)

        except Exception:
            # Display is non-critical — silently ignore errors
            pass

    def _close_async_display(self) -> None:
        """Keep the live window open when the simulation finishes (user can close manually)."""
        if self._async_fig is not None:
            try:
                import matplotlib.pyplot as plt
                self._async_fig.canvas.manager.set_window_title(
                    'Async Simulation — Complete (close to exit)')
                plt.ioff()
                plt.show(block=False)
            except Exception:
                pass

    def _log_async_state(self, negotiation_occurred: bool) -> None:
        """Append current agent positions to path logs; record negotiation events."""
        if not self.log_enabled or not self.logger:
            return

        for agent_id, agent in self.agents.items():
            self.logger.append_agent_path(agent_id, agent.position)

        if negotiation_occurred and self._current_negotiation_data:
            event = {
                'tick': self._async_tick,
                'timestamp': datetime.now().isoformat(),
                'conflicting_agents': [
                    aid for aid in self.agents
                    if aid in self._pending_resolution_agents
                ],
                'negotiation_data': self._current_negotiation_data,
                'agent_path_indices': {
                    str(aid): len(
                        self.logger.log_data.get('agent_paths', {}).get(str(aid), [])
                    ) - 1
                    for aid in self.agents
                },
            }
            self.logger.log_async_negotiation(event)
            self._current_negotiation_data = None

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
    def _negotiate_conflicts(self, conflict_info: Dict, planned_moves: Dict) -> Tuple[Dict, Optional[Dict]]:
        print("🤖 Initiating LLM-based conflict negotiation...")
        
        # Track negotiation timing
        negotiation_start_time = time.time()
        
        # Prepare conflict data for negotiator
        # Use a pre-captured map snapshot if available (set by _start_background_negotiation)
        # so the background thread works with a consistent map state.
        conflict_data = {
            'agents': [],
            'conflict_points': conflict_info['conflict_points'],
            'map_state': conflict_info.get(
                'map_state_snapshot', self.warehouse_map.get_state_dict()),
            'turn': self.current_turn
        }
        
        # Add agent data for conflicting agents
        # Use pre-built agent data from conflict_info if available (includes failed_move_history)
        if 'agents' in conflict_info and conflict_info['agents']:
            # Use the detailed agent data from conflict detection
            conflict_data['agents'] = conflict_info['agents']
        else:
            # Fallback: build basic agent data (shouldn't normally happen)
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
            neg_data = self._build_negotiation_log_data(
                conflict_data, prompts_data, {}, refinement_history, {}, {}
            )
            self._current_negotiation_data = neg_data  # keep for turn-based compat
            return {
                'agent_actions': {},
                'resolution': 'deadlock_skipped',
                'reasoning': 'Negotiation failed to resolve after max refinement iterations',
                'refinement_history': refinement_history
            }, neg_data
        
        # Build negotiation log data for unified logger
        # Extract agent validations from refinement history or resolution
        agent_validations = self._extract_agent_validations(resolution, refinement_history)
        final_actions = self._extract_final_actions(resolution)
        
        neg_data = self._build_negotiation_log_data(
            conflict_data, prompts_data, resolution, refinement_history, agent_validations, final_actions
        )
        self._current_negotiation_data = neg_data  # keep for turn-based compat
        
        # Add refinement history if available
        if refinement_history:
            resolution['refinement_history'] = refinement_history
        
        return resolution, neg_data
    
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
                    # Convert path elements to tuples and store the full LLM path.
                    # _execute_action will skip path[0] if it equals the current
                    # position, so the agent always advances on the first call.
                    updated_path = [tuple(pos) if isinstance(pos, (list, tuple)) else pos for pos in negotiated_path]
                    agent.set_path(updated_path)
                    
                    # Mark agent as having a negotiated path to preserve it
                    agent._has_negotiated_path = True
                
                success = agent.execute_negotiated_action(action_data, map_state)
                
                if success:
                    # After a successful move the agent is now at agent.position
                    # (the new cell).  Strip all leading planned_path entries up
                    # to and including that cell so the remainder is still ahead.
                    if hasattr(agent, '_has_negotiated_path') and getattr(agent, '_has_negotiated_path', False) and agent.planned_path:
                        # Find the first occurrence of the new position in the
                        # stored path and keep everything after it.
                        new_pos = agent.position
                        try:
                            idx = agent.planned_path.index(new_pos)
                            remaining = agent.planned_path[idx + 1:]
                        except ValueError:
                            # New position not found in path; consume one step.
                            remaining = agent.planned_path[1:]
                        if remaining:
                            agent.planned_path = remaining
                        else:
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
                    success, failure_reason = agent.move_to(next_pos, map_state)
                
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
                        'from_position': agent.position,
                        'failure_reason': failure_reason
                    })
                    
                    # Keep only recent failed move history
                    if len(self.agent_failed_move_history[agent_id]) > self.stagnation_turns:
                        self.agent_failed_move_history[agent_id] = self.agent_failed_move_history[agent_id][-self.stagnation_turns:]
                    
                    print(f"❌ Agent {agent_id}: Move to {next_pos} failed ({self.failed_move_counts[agent_id]} consecutive failures) - Reason: {failure_reason}")
    
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

                        # Lifelong mode: log completion and assign a new task immediately
                        if self.simulation_mode == 'lifelong':
                            self._log_lifelong_task_completion(agent_id)
                            self._assign_next_task_lifelong(agent_id)

    # Log task completion timing for lifelong metrics
    def _log_lifelong_task_completion(self, agent_id: int):
        task_start = self._agent_task_start_turns.get(agent_id, 0)
        duration = self.current_turn - task_start
        record = {
            'agent_id': agent_id,
            'turn': self.current_turn,
            'task_duration_turns': duration,
            'timestamp': datetime.now().isoformat()
        }
        self._lifelong_task_completions.append(record)
        if self.log_enabled and self.logger and hasattr(self.logger, 'log_task_completion'):
            self.logger.log_task_completion(agent_id, self.current_turn, duration)

    def _assign_next_task_lifelong(self, agent_id: int):
        """Assign a new random box+target pair to agent after delivery in lifelong mode."""
        # Collect currently occupied positions to avoid overlapping new tasks
        occupied = set()
        for pos in self.warehouse_map.agents.values():
            occupied.add(pos)
        for pos in self.warehouse_map.boxes.values():
            occupied.add(pos)
        for pos in self.warehouse_map.targets.values():
            occupied.add(pos)

        # Gather all walkable (non-wall) positions
        walkable = [
            (x, y)
            for y in range(self.warehouse_map.height)
            for x in range(self.warehouse_map.width)
            if self.warehouse_map.grid[y, x] != CellType.WALL.value
        ]
        available = [pos for pos in walkable if pos not in occupied]

        if len(available) < 2:
            print(f"⚠️  Agent {agent_id}: Not enough free cells for new lifelong task")
            return

        random.shuffle(available)
        box_pos = available[0]
        target_pos = available[1]

        # Place box and target (reuse agent_id as the entity ID for simplicity)
        box_x, box_y = box_pos
        self.warehouse_map.boxes[agent_id] = box_pos
        self.warehouse_map.grid[box_y, box_x] = CellType.BOX.value

        target_x, target_y = target_pos
        self.warehouse_map.targets[agent_id] = target_pos
        self.warehouse_map.grid[target_y, target_x] = CellType.TARGET.value

        # Ensure agent goal mapping is current
        self.warehouse_map.agent_goals[agent_id] = agent_id

        # Point agent at the new box and reset path state
        agent = self.agents[agent_id]
        agent.set_target(box_pos)
        if hasattr(agent, '_has_negotiated_path'):
            agent._has_negotiated_path = False
        agent.planned_path = []

        # Record task start turn for duration tracking
        self._agent_task_start_turns[agent_id] = self.current_turn

        print(f"🔄 Agent {agent_id}: New lifelong task — box at {box_pos}, target at {target_pos}")

        # Plan an initial path to the new box
        map_state = self.warehouse_map.get_state_dict()
        agent.plan_path(map_state)
    
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
            'total_collisions': self.collision_count,
            # Lifelong throughput metrics
            'throughput_tasks_per_second': round(
                self.successful_deliveries / makespan_seconds, 4
            ) if makespan_seconds > 0 else 0,
            'throughput_tasks_per_turn': round(
                self.successful_deliveries / total_turns, 4
            ),
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
        
        # Build mode-aware summary
        if self.simulation_mode == 'async':
            self.logger.log_data['summary'] = self.logger._compute_async_summary(metrics)
        elif self.simulation_mode == 'lifelong':
            self.logger.log_data['summary'] = self.logger._compute_lifelong_summary(metrics)
        else:
            self.logger.log_data['summary'] = self.logger._compute_turnbased_summary(metrics)
        
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
        
        if self.simulation_mode in ('async', 'lifelong'):
            # Async / lifelong: auto-run with live display
            print(f"{Fore.CYAN}{'Async' if self.simulation_mode == 'async' else 'Lifelong'} mode: running automatically with live display...{Style.RESET_ALL}")
            while self.run_simulation_step():
                pass
            # Wait for any in-flight background negotiations to finish
            self._drain_pending_negotiations()
            self._close_async_display()
        else:
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
        
        mode_label = f"{self._async_tick} ticks" if self.simulation_mode in ('async', 'lifelong') else f"{self.current_turn} turns"
        print(f"\n{Fore.GREEN}Simulation completed in {mode_label}!{Style.RESET_ALL}")
