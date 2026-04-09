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
from typing import Any, Dict, List, Tuple, Optional, Union
from colorama import init, Fore, Back, Style

from ..map_generator import WarehouseMap, CellType
from ..agents import RobotAgent
from ..llm.central_negotiator import CentralNegotiator
from ..llm.parallel_negotiator_manager import ParallelNegotiatorManager
from ..navigation import ConflictDetector, SimplePathfinder
from ..navigation.planners import (
    MultiAgentPlanner,
    PLANNER_STATUS_FAILED_NO_SOLUTION,
    create_multi_agent_planner,
)
from ..logging import UnifiedLogger

# Initialize colorama for colored terminal output
init(autoreset=True)

class GameEngine:
    # Default planning horizon parameters for reservation-based sequential planning.
    # _MIN_TIME_HORIZON is the minimum horizon in turns.
    # _GRID_HORIZON_MULTIPLIER scales horizon with map area for larger maps.
    _MIN_TIME_HORIZON = 16
    _GRID_HORIZON_MULTIPLIER = 2

    def __init__(self, width: int = 8, height: int = 6, num_agents: int = 2):
        """Initialize the game engine with specified parameters"""
        self.width = width
        self.height = height
        self.num_agents = max(2, min(num_agents, 4))  # Ensure 2-4 agents
        
        # Core components
        self.warehouse_map = WarehouseMap(width, height)
        self.agents = {}
        self.central_negotiator = CentralNegotiator()
        self.parallel_negotiator_manager = ParallelNegotiatorManager(
            enable_spatial_hints=self.central_negotiator.enable_spatial_hints
        )
        self.conflict_detector = ConflictDetector(width, height)
        self.pathfinder = SimplePathfinder(width, height)
        self.path_planner_mode = os.getenv('PATH_PLANNER_MODE', 'astar')
        self.multi_agent_planner: MultiAgentPlanner = create_multi_agent_planner(
            mode=self.path_planner_mode,
            pathfinder=self.pathfinder,
            width=width,
            height=height,
            min_time_horizon=self._MIN_TIME_HORIZON,
            grid_horizon_multiplier=self._GRID_HORIZON_MULTIPLIER,
        )

        # Enable parallel negotiation mode (can be disabled for benchmarking)
        self.use_parallel_negotiation = os.getenv('USE_PARALLEL_NEGOTIATION', 'true').lower() == 'true'
        # Disable LLM negotiation and rely on planner-only execution.
        self.disable_llm_negotiation = os.getenv('DISABLE_LLM_NEGOTIATION', 'false').strip().lower() == 'true'
        
        # Simulation state
        self.current_turn = 0
        self.max_turns = 100
        self.is_running = False
        self.simulation_complete = False
        self.simulation_failed = False
        self.failure_reason: Optional[str] = None
        
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
        self.initial_agent_box_positions = {}  # Initial box position per agent
        self.initial_agent_target_positions = {}  # Target position per agent
        self.total_actual_steps = 0  # Successful moves + failed move attempts
        self.total_min_required_steps = 0  # Sum of shortest-path distances for assigned task segments
        self._wall_positions_cache: Optional[set[tuple[int, int]]] = None
        
        # Benchmark mode controls
        self.stop_requested = False  # External signal to stop simulation
        self.timeout_seconds = 0  # Time limit for simulation (0 = no limit)
        self.silent_mode = False  # Suppress print output for benchmark runs

        # Simulation mode: 'turn_based' (default), 'async', or 'lifelong'
        self.simulation_mode = 'turn_based'

        # Difficulty level for turn-based mode: 0.0 (none), 0.25 (easy), 0.5 (hard)
        # Controls the fraction of remaining goals randomly relocated at turns 15, 30, and 50
        self.difficulty: float = 0.0

        # Async mode: agents blocked waiting for LLM resolution
        self._pending_resolution_agents: set = set()

        # Async mode: per-agent last-known safe position (before conflict)
        self._conflict_hold_positions: Dict[int, Tuple[int, int]] = {}

        # Lifelong mode: per-task completion records and task start tracking
        self._lifelong_task_completions: List[Dict] = []
        self._agent_task_start_turns: Dict[int, int] = {}
        self._agent_trail_start: Dict[int, int] = {}  # Path index where current task trail begins

        # Async mode: live display window and tick counter
        self._async_tick: int = 0
        self._async_fig: Optional[Any] = None
        self._async_ax: Optional[Any] = None

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

        # Reset path-efficiency counters for a fresh run.
        self.total_actual_steps = 0
        self.total_min_required_steps = 0
        self._wall_positions_cache = None
        self.initial_agent_positions = {}
        self.initial_agent_box_positions = {}
        self.initial_agent_target_positions = {}
        self.agent_paths = {}
        
        # Note: warehouse_map is already loaded from layout in main.py
        # Verify map is properly initialized
        if not self.warehouse_map or self.warehouse_map.width == 0:
            raise ValueError("Warehouse map not properly initialized. Load a layout first.")

        # Cache static walls for repeated shortest-path computations.
        self._wall_positions_cache = self._get_wall_positions()
        
        # Initialize agents from the layout
        # Agents are already created in main.py, but set up their targets here
        for agent_id, agent in self.agents.items():
            agent.on_move_attempt = self._record_move_attempt

            # Assign box to pick up (proper warehouse task: box → target)
            if agent_id in self.warehouse_map.agent_goals:
                box_id = agent_id  # Each agent gets their own box
                if box_id in self.warehouse_map.boxes:
                    box_pos = self.warehouse_map.boxes[box_id]
                    agent.set_target(box_pos)  # First go to the box
                    self._accumulate_min_required_segment(agent_id, box_pos)
                    # Record initial box position for path efficiency calculation
                    self.initial_agent_box_positions[agent_id] = box_pos
                target_id = self.warehouse_map.agent_goals.get(agent_id)
                if target_id is not None and target_id in self.warehouse_map.targets:
                    self.initial_agent_target_positions[agent_id] = self.warehouse_map.targets[target_id]
            
            # Initialize metrics tracking for this agent
            self.initial_agent_positions[agent_id] = agent.position
            self.agent_paths[agent_id] = [agent.position]
            self._agent_task_start_turns[agent_id] = 0
            self._agent_trail_start[agent_id] = 0
        
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
                                self._accumulate_min_required_segment(agent_id, target_pos)
                                print(f"📦 Agent {agent_id}: Ready to deliver to {target_pos}")
        
        # Initial pathfinding
        print(f"🧭 Planner backend: {self.multi_agent_planner.get_backend_name()}")
        self._plan_initial_paths()
        
        # Initialize unified logger with scenario data
        if self.log_enabled and self.logger:
            self.logger.initialize({
                'type': 'interactive_simulation',
                'simulation_mode': self.simulation_mode,
                'path_planner_mode': self.multi_agent_planner.get_backend_name(),
                'difficulty': self.difficulty,
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
            # Log initial positions to agent paths and record initial task assignments
            if self.log_enabled and self.logger:
                for agent_id, agent in self.agents.items():
                    self.logger.append_agent_path(agent_id, agent.position)
                    if self.simulation_mode == 'lifelong' and hasattr(self.logger, 'log_task_assignment'):
                        box_pos = self.warehouse_map.boxes.get(agent_id)
                        target_pos = self.warehouse_map.targets.get(agent_id)
                        self.logger.log_task_assignment(agent_id, 0, box_pos, target_pos)
            self._update_async_display()
        else:
            # Turn-based: normal terminal display
            self._log_turn_state("SIMULATION_START")
            self.display_map()
    
    # Plan initial paths for all agents
    def _plan_initial_paths(self):
        print("Planning initial paths for all agents...")
        self._replan_with_reservations()

    def _record_move_attempt(
        self,
        agent_id: int,
        from_position: Tuple[int, int],
        to_position: Tuple[int, int],
    ) -> None:
        """Count each attempted move (successful or failed) for efficiency metrics."""
        self.total_actual_steps += 1

    def _get_wall_positions(self) -> set[tuple[int, int]]:
        """Get wall coordinates for shortest-path calculations."""
        if self._wall_positions_cache is not None:
            return self._wall_positions_cache

        walls: set[tuple[int, int]] = set()
        grid = self.warehouse_map.grid
        for y in range(len(grid)):
            for x in range(len(grid[y])):
                if grid[y][x] == CellType.WALL.value:
                    walls.add((x, y))

        self._wall_positions_cache = walls
        return walls

    def _shortest_distance_with_fallback(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> int:
        """Compute segment minimum distance using A* and Manhattan fallback."""
        if start == goal:
            return 0

        walls = self._get_wall_positions()
        path = self.pathfinder.find_path(start, goal, walls)
        if path:
            return max(0, len(path) - 1)

        return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

    def _accumulate_min_required_segment(
        self,
        agent_id: int,
        goal_position: Optional[Tuple[int, int]],
    ) -> None:
        """Add a new required shortest-path segment when a task goal is assigned."""
        if goal_position is None:
            return
        if agent_id not in self.agents:
            return

        start_position = self.agents[agent_id].position
        self.total_min_required_steps += self._shortest_distance_with_fallback(
            start_position,
            goal_position,
        )
    
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
                    if not self.silent_mode:
                        print(f"🚫 Agent {agent_id}: Stagnant due to {failed_move_count} consecutive failed moves")

        if stagnant_agents:
            if not self.silent_mode:
                print(f"🚫 STAGNATION DETECTED! Agents with failed moves: {stagnant_agents}")

            # Try A* pathfinding first before triggering LLM negotiation
            map_state = self.warehouse_map.get_state_dict()
            agents_needing_llm = []
            a_star_resolved = []

            for aid in stagnant_agents:
                agent = self.agents[aid]

                # Try to find a valid path using A* pathfinding
                if not self.silent_mode:
                    print(f"🔍 Agent {aid}: Attempting planner pathfinding to resolve stuck state...")
                try:
                    replan_result = self.multi_agent_planner.replan_subset(self.agents, map_state, {aid})
                    fresh_path = replan_result.solutions.get(aid, [])

                    if fresh_path and len(fresh_path) > 0:
                        # Check if this path conflicts with other agents' paths
                        path_has_conflict = self._check_path_conflicts_with_others(aid, fresh_path)

                        if not path_has_conflict:
                            # Planner found a valid non-conflicting path, use it
                            agent.planned_path = fresh_path
                            agent._has_negotiated_path = False
                            a_star_resolved.append(aid)
                            if not self.silent_mode:
                                print(f"✅ Agent {aid}: Planner resolved stuck state with {len(fresh_path)}-step path")

                            # Clear failed move history for this agent
                            if aid in self.agent_failed_move_history:
                                self.agent_failed_move_history[aid] = []
                            if aid in self.failed_move_counts:
                                self.failed_move_counts[aid] = 0
                        else:
                            # Planner path conflicts with other agents, needs LLM
                            if not self.silent_mode:
                                print(f"⚠️  Agent {aid}: Planner path conflicts with other agents, needs LLM negotiation")
                            agents_needing_llm.append(aid)
                    else:
                        # Planner could not find a path, needs LLM
                        if not self.silent_mode:
                            print(f"⚠️  Agent {aid}: Planner could not find valid path, needs LLM negotiation")
                        agents_needing_llm.append(aid)
                except Exception as e:
                    if not self.silent_mode:
                        print(f"⚠️  Agent {aid}: Planner pathfinding failed: {e}, needs LLM negotiation")
                    agents_needing_llm.append(aid)

            # Report planner resolution results
            if a_star_resolved:
                if not self.silent_mode:
                    print(f"🎯 Planner successfully resolved {len(a_star_resolved)} stuck agent(s): {a_star_resolved}")

            # If all agents were resolved by planner, no LLM negotiation needed
            if not agents_needing_llm:
                if not self.silent_mode:
                    print(f"✅ All stagnant agents resolved by planner pathfinding!")
                return {'has_conflicts': False}

            # Only trigger LLM for agents that A* could not resolve
            if not self.silent_mode:
                print(f"🤖 Triggering LLM negotiation for {len(agents_needing_llm)} agent(s): {agents_needing_llm}")
            agent_data = []

            for aid in agents_needing_llm:
                agent = self.agents[aid]

                # Get fresh planned path for context
                current_path = agent.planned_path if hasattr(agent, 'planned_path') else []

                # If path is empty or agent has target, try to calculate a fresh path
                if not current_path and agent.target_position:
                    try:
                        replan_result = self.multi_agent_planner.replan_subset(self.agents, map_state, {aid})
                        fresh_path = replan_result.solutions.get(aid, [])
                        if fresh_path:
                            current_path = fresh_path
                            if not self.silent_mode:
                                print(f"🗺️  Agent {aid}: Calculated fresh path with {len(fresh_path)} steps for stagnation context")
                    except Exception as e:
                        if not self.silent_mode:
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
                'conflicting_agents': agents_needing_llm,
                'conflict_points': [self.agents[aid].position for aid in agents_needing_llm],
                'agents': agent_data
            }

        return {'has_conflicts': False}

    # Check if a path for an agent conflicts with other agents' current planned paths
    def _check_path_conflicts_with_others(self, agent_id: int, path: List[Tuple[int, int]]) -> bool:
        # Normalize the candidate path: ensure it starts with the agent's current position
        agent = self.agents[agent_id]
        normalized_candidate_path = path
        if path and path[0] != agent.position:
            # Path doesn't start with current position, prepend it
            normalized_candidate_path = [agent.position] + path

        # Get all other agents' planned paths and normalize them.
        # Include stationary agents as occupied at their current position.
        other_agents_paths = {}
        for aid, other_agent in self.agents.items():
            if aid == agent_id:
                continue

            if hasattr(other_agent, 'planned_path') and other_agent.planned_path:
                other_path = other_agent.planned_path
                # Normalize: ensure path starts with agent's current position
                if other_path and other_path[0] != other_agent.position:
                    # Negotiated path may not include current position, prepend it
                    normalized_other_path = [other_agent.position] + other_path
                else:
                    normalized_other_path = other_path
            else:
                # Agent has no planned path, treat as stationary on its current position
                normalized_other_path = [other_agent.position]

            other_agents_paths[aid] = normalized_other_path

        # If no other agents have paths, no conflict possible
        if not other_agents_paths:
            return False

        # Check for conflicts using conflict detector
        all_paths = {agent_id: normalized_candidate_path}
        all_paths.update(other_agents_paths)

        conflict_info = self.conflict_detector.detect_path_conflicts(all_paths, self.current_turn)

        # Return True if conflicts exist and involve this agent
        if conflict_info['has_conflicts'] and agent_id in conflict_info.get('conflicting_agents', []):
            return True

        return False

    # Detect agents stuck due to too many failed moves
    def detect_move_failure_deadlocks(self, planned_moves: Dict) -> Dict:
        stuck_agents = []

        # Check for agents with too many failed moves
        for agent_id, failure_count in self.failed_move_counts.items():
            if failure_count >= self.max_failed_moves:
                stuck_agents.append(agent_id)

        if stuck_agents:
            if not self.silent_mode:
                print(f"🔥 DEADLOCK DETECTED! Agents with {self.max_failed_moves}+ failed moves: {stuck_agents}")

            # Try A* pathfinding first before triggering LLM negotiation
            map_state = self.warehouse_map.get_state_dict()
            agents_needing_llm = []
            a_star_resolved = []

            for aid in stuck_agents:
                if aid not in self.agents:
                    continue

                agent = self.agents[aid]

                # Try to find a valid path using A* pathfinding
                if not self.silent_mode:
                    print(f"🔍 Agent {aid}: Attempting planner pathfinding to resolve deadlock...")
                try:
                    replan_result = self.multi_agent_planner.replan_subset(self.agents, map_state, {aid})
                    fresh_path = replan_result.solutions.get(aid, [])

                    if fresh_path and len(fresh_path) > 0:
                        # Check if this path conflicts with other agents' paths
                        path_has_conflict = self._check_path_conflicts_with_others(aid, fresh_path)

                        if not path_has_conflict:
                            # Planner found a valid non-conflicting path, use it
                            agent.planned_path = fresh_path
                            agent._has_negotiated_path = False
                            a_star_resolved.append(aid)
                            if not self.silent_mode:
                                print(f"✅ Agent {aid}: Planner resolved deadlock with {len(fresh_path)}-step path")

                            # Clear failed move history for this agent
                            if aid in self.agent_failed_move_history:
                                self.agent_failed_move_history[aid] = []
                            if aid in self.failed_move_counts:
                                self.failed_move_counts[aid] = 0
                        else:
                            # Planner path conflicts with other agents, needs LLM
                            if not self.silent_mode:
                                print(f"⚠️  Agent {aid}: Planner path conflicts with other agents, needs LLM negotiation")
                            agents_needing_llm.append(aid)
                    else:
                        # Planner could not find a path, needs LLM
                        if not self.silent_mode:
                            print(f"⚠️  Agent {aid}: Planner could not find valid path, needs LLM negotiation")
                        agents_needing_llm.append(aid)
                except Exception as e:
                    if not self.silent_mode:
                        print(f"⚠️  Agent {aid}: Planner pathfinding failed: {e}, needs LLM negotiation")
                    agents_needing_llm.append(aid)

            # Report planner resolution results
            if a_star_resolved:
                if not self.silent_mode:
                    print(f"🎯 Planner successfully resolved {len(a_star_resolved)} deadlocked agent(s): {a_star_resolved}")

            # If all agents were resolved by planner, no LLM negotiation needed
            if not agents_needing_llm:
                if not self.silent_mode:
                    print(f"✅ All deadlocked agents resolved by planner pathfinding!")
                return {'has_conflicts': False}

            # Only trigger LLM for agents that A* could not resolve
            if not self.silent_mode:
                print(f"🤖 Triggering LLM negotiation for {len(agents_needing_llm)} agent(s): {agents_needing_llm}")

            return {
                'has_conflicts': True,
                'conflict_type': 'deadlock',
                'conflicting_agents': agents_needing_llm,
                'conflict_points': [self.agents[aid].position for aid in agents_needing_llm if aid in self.agents],
                'agents': [
                    {
                        'id': aid,
                        'current_pos': self.agents[aid].position,
                        'target_pos': self.agents[aid].target_position,
                        'planned_path': planned_moves.get(aid, []),
                        'stuck_reason': 'failed_moves',
                        'failure_count': self.failed_move_counts.get(aid, 0),
                        'failed_move_history': self.agent_failed_move_history.get(aid, [])
                    } for aid in agents_needing_llm if aid in self.agents
                ],
                'deadlock_breaking': True  # Special flag for negotiator
            }

        return {'has_conflicts': False}
    
    # Force deadlock negotiation for stuck agents
    def _force_deadlock_negotiation(self, stuck_agents: List[int], planned_moves: Dict):
        if not self.silent_mode:
            print("🛠️ DEADLOCK BREAKING: Creating artificial conflict to trigger negotiation")

        # Create artificial conflict data for deadlock breaking
        conflict_data = self.detect_move_failure_deadlocks(planned_moves)

        if conflict_data['has_conflicts']:
            # Force negotiation
            if not self.silent_mode:
                print(f"🤖 Forcing negotiation for deadlock resolution...")
            resolution, _ = self._negotiate_conflicts(conflict_data, planned_moves)
            self._execute_negotiated_actions(resolution)

            # Reset failure counts after forced negotiation
            for agent_id in stuck_agents:
                self.failed_move_counts[agent_id] = 0
                # Also clear failed move history
                if agent_id in self.agent_failed_move_history:
                    self.agent_failed_move_history[agent_id] = []
                if not self.silent_mode:
                    print(f"🔄 Agent {agent_id}: Reset failure count and move history after deadlock negotiation")
    
    def _apply_difficulty_goal_alteration(self):
        """Alter a fraction of remaining goals in place at turns 15, 30, and 50."""
        if self.difficulty == 0.0:
            return

        # Find agents whose delivery target has not yet been completed
        remaining_agent_ids = [
            agent_id for agent_id in self.agents
            if self.warehouse_map.agent_goals.get(agent_id) in self.warehouse_map.targets
        ]

        if not remaining_agent_ids:
            return

        # Ensure at least one goal is altered when difficulty is non-zero
        num_to_alter = max(1, int(len(remaining_agent_ids) * self.difficulty))
        num_to_alter = min(num_to_alter, len(remaining_agent_ids))

        selected_ids = random.sample(remaining_agent_ids, num_to_alter)

        if not self.silent_mode:
            print(f"\n{Fore.MAGENTA}⚡ DIFFICULTY EVENT (Turn {self.current_turn + 1}): "
                  f"Altering {num_to_alter}/{len(remaining_agent_ids)} remaining goals "
                  f"(difficulty={self.difficulty}){Style.RESET_ALL}")

        # Build occupied set to avoid placing new targets on existing entities
        occupied: set = set()
        for pos in self.warehouse_map.agents.values():
            occupied.add(pos)
        for pos in self.warehouse_map.boxes.values():
            occupied.add(pos)
        for pos in self.warehouse_map.targets.values():
            occupied.add(pos)

        for agent_id in selected_ids:
            target_id = self.warehouse_map.agent_goals[agent_id]
            old_target_pos = self.warehouse_map.targets[target_id]

            # Free the old target position only when no agent or box currently occupies it,
            # so subsequent iterations may reuse the cell without creating overlapping entities.
            agent_positions = set(self.warehouse_map.agents.values())
            box_positions = set(self.warehouse_map.boxes.values())
            if old_target_pos not in agent_positions and old_target_pos not in box_positions:
                occupied.discard(old_target_pos)

            available = [
                (x, y)
                for y in range(self.warehouse_map.height)
                for x in range(self.warehouse_map.width)
                if self.warehouse_map.grid[y, x] != CellType.WALL.value
                and (x, y) not in occupied
            ]

            if not available:
                if not self.silent_mode:
                    print(f"  ⚠️  Agent {agent_id}: No available positions for goal alteration, skipping")
                occupied.add(old_target_pos)
                continue

            new_target_pos = random.choice(available)
            occupied.add(new_target_pos)

            # Update grid: clear old target marker, place new one
            old_x, old_y = old_target_pos
            new_x, new_y = new_target_pos
            if self.warehouse_map.grid[old_y, old_x] == CellType.TARGET.value:
                self.warehouse_map.grid[old_y, old_x] = CellType.EMPTY.value
            self.warehouse_map.grid[new_y, new_x] = CellType.TARGET.value

            # Update the targets dictionary
            self.warehouse_map.targets[target_id] = new_target_pos

            # Keep the target position used for path efficiency calculation in sync
            self.initial_agent_target_positions[agent_id] = new_target_pos

            # Redirect the agent if it is currently carrying its box toward the old target
            agent = self.agents[agent_id]
            if agent.carrying_box and agent.target_position == old_target_pos:
                agent.set_target(new_target_pos)
                self._accumulate_min_required_segment(agent_id, new_target_pos)
                agent.planned_path = []
                if hasattr(agent, '_has_negotiated_path'):
                    agent._has_negotiated_path = False
                if not self.silent_mode:
                    print(f"  🎯 Agent {agent_id}: Goal redirected {old_target_pos} → {new_target_pos} (currently delivering)")
            else:
                if not self.silent_mode:
                    print(f"  🎯 Agent {agent_id}: Goal altered {old_target_pos} → {new_target_pos}")

        # Rebuild reservations and replan after difficulty-based target alterations
        self._replan_with_reservations()

    # Run one step of the simulation — dispatches to mode-specific implementation
    def run_simulation_step(self) -> bool:
        if self.simulation_mode in ('async', 'lifelong'):
            return self._run_async_step()
        return self._run_turn_based_step()

    def _mark_simulation_failed(self, reason: str) -> None:
        self.simulation_failed = True
        self.failure_reason = reason
        if not self.silent_mode:
            print(f"{Fore.RED}❌ Simulation failed: {reason}{Style.RESET_ALL}")

    # Run one turn-based simulation step (original logic, zero behavioural change)
    def _run_turn_based_step(self) -> bool:
        # Check for external stop request (benchmark timeout)
        if self.stop_requested:
            return False

        if self.simulation_failed:
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
        
        # Apply difficulty goal alteration at turns 15, 30, and 50
        if (self.current_turn + 1) in {15, 30, 50}:
            self._apply_difficulty_goal_alteration()
        
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
            if self.disable_llm_negotiation:
                if not self.silent_mode:
                    print("🚫 STAGNATION DETECTED! LLM negotiation is disabled, continuing with planner-only execution...")
            else:
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
        
        # PHASE 1: Sequential A* planning (time-aware against earlier planned agents)
        plan_result = self._get_sequential_plan_result()
        planned_moves = plan_result.solutions

        if plan_result.status == PLANNER_STATUS_FAILED_NO_SOLUTION:
            self._mark_simulation_failed('mapf_failed_no_solution')
            return False

        if not planned_moves:
            print("No agents have targets to move towards.")
            # Check if all tasks are complete
            if self._check_completion():
                print("🎉 All tasks completed! Simulation complete!")
                self.simulation_complete = True
            return False

        # Check for conflicts after sequential A* planning
        conflict_info = self.conflict_detector.detect_path_conflicts(planned_moves, self.current_turn)

        # Track if negotiation occurred this turn
        negotiation_occurred = False

        if conflict_info['has_conflicts']:
            print(f"{Fore.RED}CONFLICT DETECTED!{Style.RESET_ALL}")
            print(f"Conflicting agents: {conflict_info['conflicting_agents']}")
            print(f"Conflict points: {conflict_info['conflict_points']}")

            # Track collision for metrics
            self.collision_count += len(conflict_info['conflict_points'])

            if self.disable_llm_negotiation:
                if not self.silent_mode:
                    print("⚠️  LLM negotiation disabled. Executing planned moves and monitoring for deadlock...")
                self._execute_planned_moves(planned_moves)
                deadlock_conflict = self.detect_move_failure_deadlocks(planned_moves)
                if deadlock_conflict['has_conflicts']:
                    self._mark_simulation_failed('deadlock_triggered_llm_disabled')
            else:
                # Use parallel or single negotiation based on configuration
                if self.use_parallel_negotiation:
                    resolution, negotiation_data = self._negotiate_parallel_conflicts(conflict_info, planned_moves)
                    self._current_negotiation_data = negotiation_data
                else:
                    resolution, neg_data = self._negotiate_conflicts(conflict_info, planned_moves)
                    self._current_negotiation_data = neg_data

                self._execute_negotiated_actions(resolution)
                negotiation_occurred = True
        else:
            print(f"{Fore.GREEN}No conflicts detected. Executing planned moves...{Style.RESET_ALL}")
            self._execute_planned_moves(planned_moves)

            # PHASE 2: Check for deadlock after move execution
            deadlock_conflict = self.detect_move_failure_deadlocks(planned_moves)
            if deadlock_conflict['has_conflicts']:
                if self.disable_llm_negotiation:
                    self._mark_simulation_failed('deadlock_triggered_llm_disabled')
                else:
                    print(f"🔥 DEADLOCK AFTER MOVES! Forcing resolution...")
                    self._force_deadlock_negotiation(deadlock_conflict['conflicting_agents'], planned_moves)
                    negotiation_occurred = True
        
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
        return not self.simulation_complete and not self.simulation_failed

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

        if self.simulation_failed:
            return False

        if self.timeout_seconds > 0 and self.simulation_start_time:
            if time.time() - self.simulation_start_time >= self.timeout_seconds:
                if not self.silent_mode:
                    print(f"\n{Fore.YELLOW}⏱️  Time limit ({self.timeout_seconds}s) reached!{Style.RESET_ALL}")
                return False

        # For async/lifelong the simulation is bounded by timeout_seconds, not max_turns.
        if self.simulation_complete or (self.simulation_mode not in ('async', 'lifelong') and self.current_turn >= self.max_turns):
            return False

        if not self.silent_mode:
            print(f"\n{Fore.YELLOW}=== TICK {self._async_tick + 1} (ASYNC) ==={Style.RESET_ALL}")

        if self.disable_llm_negotiation:
            return self._run_async_step_without_llm()

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

        # ── Step 5: Sequential time-aware planned moves for active free agents ──
        active_free_ids = free_agent_ids - stagnant_free
        planned_moves = self._get_sequential_planned_moves(active_free_ids)

        conflicting_ids: set = set()

        if planned_moves:
            # ── Step 6: Detect primary conflicts ─────────────────────────
            conflict_info = self.conflict_detector.detect_path_conflicts(
                planned_moves, self.current_turn)

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
                        conflict_key, conflict_info, planned_moves)

            # ── Step 7: Move non-conflicting free agents ──────────────────
            non_conflicting_ids = active_free_ids - conflicting_ids
            normal_moves = {
                agent_id: planned_moves[agent_id]
                for agent_id in non_conflicting_ids
                if agent_id in planned_moves
            }

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

    def _run_async_step_without_llm(self) -> bool:
        """Run one async tick without any LLM negotiation calls."""
        self._apply_pending_negotiation_results()

        for agent in self.agents.values():
            agent.update_turn()
        for agent_id, agent in self.agents.items():
            if hasattr(agent, '_hmas2_validated'):
                setattr(agent, '_hmas2_validated', False)

        for agent_id in list(self.agents.keys()):
            self._check_box_delivery(agent_id)

        plan_result = self._get_sequential_plan_result()
        planned_moves = plan_result.solutions
        if plan_result.status == PLANNER_STATUS_FAILED_NO_SOLUTION:
            self._mark_simulation_failed('mapf_failed_no_solution')
        elif planned_moves:
            conflict_info = self.conflict_detector.detect_path_conflicts(planned_moves, self.current_turn)
            if conflict_info['has_conflicts']:
                self.collision_count += len(conflict_info['conflict_points'])
                if not self.silent_mode:
                    print("⚠️  Async conflict detected with LLM disabled. Executing planner moves and monitoring deadlock...")

            self._execute_planned_moves(planned_moves)
            deadlock = self.detect_move_failure_deadlocks(planned_moves)
            if deadlock['has_conflicts']:
                self._mark_simulation_failed('deadlock_triggered_llm_disabled')
        else:
            if not self.silent_mode:
                print("No free agents with targets.")

        self._update_map_state()
        self._async_tick += 1
        self.current_turn += 1
        self._log_async_state(negotiation_occurred=False)
        self._update_async_display()

        if self.simulation_mode == 'lifelong':
            return not self.simulation_failed

        if self._check_completion():
            if not self.silent_mode:
                print(f"{Fore.GREEN}🎉 All agents reached their targets!{Style.RESET_ALL}")
            self.simulation_complete = True
            return False

        return not self.simulation_failed

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
            # Close any existing figure with the same name so each round starts clean.
            plt.close('Async Simulation — Live View')
            plt.ion()
            self._async_fig, self._async_ax = plt.subplots(
                figsize=(max(8, self.width * 0.6), max(6, self.height * 0.6)),
                num='Async Simulation — Live View'
            )
            try:
                manager = getattr(self._async_fig.canvas, 'manager', None)
                if manager is not None and hasattr(manager, 'set_window_title'):
                    manager.set_window_title('Async Simulation — Live View')
            except Exception:
                pass
            print("🖥️  Live async display window opened")
        except Exception as e:
            print(f"⚠️  Could not open async live display: {e}")
            self._async_fig = None
            self._async_ax = None

    def _update_async_display(self) -> None:
        """Redraw the live matplotlib window with the current simulation state."""
        if self.silent_mode or self._async_fig is None or self._async_ax is None:
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

                # Draw recorded path trimmed to the current task
                if self.log_enabled and self.logger:
                    full_path = self.logger.log_data.get('agent_paths', {}).get(str(agent_id), [])
                    trail_start = self._agent_trail_start.get(agent_id, 0)
                    path = full_path[trail_start:]
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
                manager = getattr(self._async_fig.canvas, 'manager', None)
                if manager is not None and hasattr(manager, 'set_window_title'):
                    manager.set_window_title('Async Simulation — Complete (close to exit)')
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
        return self._get_sequential_planned_moves()

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

    # Plan paths sequentially; each next agent avoids previous agents at specific turns
    def _get_sequential_planned_moves(self, agent_ids: Optional[set] = None) -> Dict[int, List[Tuple[int, int]]]:
        plan_result = self._get_sequential_plan_result(agent_ids)
        return plan_result.solutions

    def _get_sequential_plan_result(self, agent_ids: Optional[set] = None):
        map_state = self.warehouse_map.get_state_dict()
        subset_ids = set(agent_ids) if agent_ids is not None else None
        return self.multi_agent_planner.plan_all(self.agents, map_state, subset_ids)

    # Replan paths with reservation-based sequential A* and refresh agent path buffers
    def _replan_with_reservations(self, agent_ids: Optional[set] = None) -> Dict[int, List[Tuple[int, int]]]:
        map_state = self.warehouse_map.get_state_dict()
        if agent_ids is None:
            planned_moves = self.multi_agent_planner.plan_all(self.agents, map_state, None).solutions
        else:
            planned_moves = self.multi_agent_planner.replan_subset(self.agents, map_state, set(agent_ids)).solutions
        for agent_id, path in planned_moves.items():
            if agent_id in self.agents:
                self.agents[agent_id].planned_path = path.copy()
        return planned_moves
        
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

    # Negotiate conflicts using parallel negotiators for independent groups
    def _negotiate_parallel_conflicts(
        self,
        conflict_info: Dict,
        planned_moves: Dict
    ) -> Tuple[Dict, Optional[Union[Dict, List[Dict]]]]:
        """
        Handle parallel conflict negotiation by grouping conflicts and resolving them simultaneously.
        Returns (merged_resolution, list_of_negotiation_logs)
        """
        # Keep parallel negotiators aligned with runtime central-negotiator configuration
        # (e.g., benchmark code toggles central spatial hints directly).
        self.parallel_negotiator_manager.enable_spatial_hints = self.central_negotiator.enable_spatial_hints

        if not self.silent_mode:
            print("🔄 Initiating PARALLEL conflict negotiation...")

        # Track negotiation timing
        negotiation_start_time = time.time()

        # Step 1: Group conflicts into independent groups
        # Get all planned paths for grouping
        all_paths = {}
        for agent_id in conflict_info['conflicting_agents']:
            path = planned_moves.get(agent_id, [])
            if not path and agent_id in self.agents:
                # Use agent's current position if no planned move
                path = [self.agents[agent_id].position]
            if path:
                all_paths[agent_id] = path

        # Use conflict detector to group conflicts
        conflict_groups = self.conflict_detector.group_conflicts(all_paths, self.current_turn)

        if not conflict_groups:
            # No groups found, fallback to single-group negotiation
            if not self.silent_mode:
                print("⚠️  No conflict groups identified, falling back to single negotiation")
            return self._negotiate_conflicts(conflict_info, planned_moves)

        if len(conflict_groups) == 1:
            # Only one group, use regular negotiation
            if not self.silent_mode:
                print(f"📋 Only 1 conflict group found, using single negotiation")
            return self._negotiate_conflicts(conflict_info, planned_moves)

        # Step 2: Prepare conflict data for each group
        map_state = conflict_info.get('map_state_snapshot', self.warehouse_map.get_state_dict())

        group_conflict_data_list = []
        for group in conflict_groups:
            # Build agent data for this group
            group_agent_data = []
            for agent_id in group['conflicting_agents']:
                if agent_id in self.agents:
                    agent = self.agents[agent_id]
                    agent_data = {
                        'id': agent_id,
                        'current_pos': agent.position,
                        'target_pos': agent.target_position,
                        'planned_path': planned_moves.get(agent_id, [])
                    }
                    # Add failure history if available
                    if agent_id in self.agent_failed_move_history:
                        agent_data['failed_move_history'] = self.agent_failed_move_history[agent_id]
                    group_agent_data.append(agent_data)

            group_conflict_data = {
                'agents': group_agent_data,
                'conflicting_agents': group['conflicting_agents'],
                'conflict_points': group['conflict_points'],
                'map_state': map_state,
                'turn': self.current_turn
            }
            group_conflict_data_list.append(group_conflict_data)

        # Step 3: Prepare validators for ALL agents
        all_agent_validators = {}
        for agent_id in conflict_info['conflicting_agents']:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                if hasattr(agent, 'validator'):
                    all_agent_validators[agent_id] = agent.validator.validate_negotiated_action

        # Step 4: Run parallel negotiations
        resolutions, negotiation_logs = self.parallel_negotiator_manager.negotiate_parallel_conflicts(
            group_conflict_data_list,
            all_agent_validators,
            silent_mode=self.silent_mode
        )

        # Step 5: Merge resolutions
        merged_resolution = self.parallel_negotiator_manager.merge_resolutions(resolutions)
        merged_resolution = self._apply_sequential_astar_fallback_for_missing_actions(
            merged_resolution,
            conflict_info.get('conflicting_agents', [])
        )

        # Track negotiation time
        negotiation_end_time = time.time()
        negotiation_duration = negotiation_end_time - negotiation_start_time
        self.negotiation_times.append((negotiation_start_time, negotiation_end_time))

        if not self.silent_mode:
            print(f"⏱️  Total negotiation time: {negotiation_duration:.2f}s")

        # Step 6: Build log data for each group negotiation
        all_neg_data = []
        for i, (group_data, log_info) in enumerate(zip(group_conflict_data_list, negotiation_logs)):
            resolution = log_info.get('resolution', {})
            refinement_history = log_info.get('refinement_history', [])
            prompts_data = log_info.get('prompts_data', {})

            # Extract agent validations and final actions
            agent_validations = self._extract_agent_validations(resolution, refinement_history)
            final_actions = self._extract_final_actions(resolution)

            neg_data = self._build_negotiation_log_data(
                group_data,
                prompts_data,
                resolution,
                refinement_history,
                agent_validations,
                final_actions
            )

            # Add group metadata
            neg_data['group_index'] = i
            neg_data['total_groups'] = len(conflict_groups)
            neg_data['negotiation_mode'] = 'parallel'

            all_neg_data.append(neg_data)

        # For backward compatibility, store the merged negotiation data
        if all_neg_data:
            self._current_negotiation_data = all_neg_data[0]  # Store first group as default

        # Add refinement history to merged resolution for compatibility
        if negotiation_logs and 'refinement_history' in negotiation_logs[0]:
            merged_resolution['refinement_history'] = negotiation_logs[0]['refinement_history']

        return merged_resolution, all_neg_data

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

        resolution = self._apply_sequential_astar_fallback_for_missing_actions(
            resolution,
            conflict_info.get('conflicting_agents', [])
        )
        
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

    # Fill missing negotiated actions with a deterministic sequential A* retry.
    def _apply_sequential_astar_fallback_for_missing_actions(
        self,
        resolution: Dict,
        conflicting_agents: List[int]
    ) -> Dict:
        if not resolution or not conflicting_agents:
            return resolution

        if 'agent_actions' in resolution and isinstance(resolution.get('agent_actions'), dict):
            agent_actions = resolution['agent_actions']
        else:
            # Normalize top-level numeric keys into agent_actions for consistent handling.
            normalized_actions = {
                str(k): v for k, v in resolution.items()
                if isinstance(k, int) or (isinstance(k, str) and k.isdigit())
            }
            non_action_fields = {
                k: v for k, v in resolution.items()
                if not (isinstance(k, int) or (isinstance(k, str) and k.isdigit()))
            }
            resolution = dict(non_action_fields)
            resolution['agent_actions'] = normalized_actions
            agent_actions = resolution['agent_actions']

        missing_agent_ids = [
            aid for aid in conflicting_agents
            if str(aid) not in agent_actions
        ]

        if not missing_agent_ids:
            return resolution

        retry_paths = self._get_sequential_planned_moves(set(missing_agent_ids))

        for agent_id in missing_agent_ids:
            agent = self.agents.get(agent_id)
            if not agent:
                continue

            fallback_path = retry_paths.get(agent_id)
            if not fallback_path:
                fallback_path = agent.planned_path if agent.planned_path else [agent.position]
            if not fallback_path:
                fallback_path = [agent.position]

            serializable_path = [
                [int(pos[0]), int(pos[1])] if isinstance(pos, (list, tuple)) and len(pos) >= 2 else pos
                for pos in fallback_path
            ]

            agent_actions[str(agent_id)] = {
                'action': 'wait' if len(serializable_path) <= 1 else 'move',
                'path': serializable_path,
                'reasoning': 'sequential_a_star_retry_for_missing_negotiation_action'
            }

        existing_reasoning = resolution.get('reasoning', '')
        fallback_note = (
            f"Applied sequential A* retry for agents with missing LLM actions: {missing_agent_ids}"
        )
        resolution['reasoning'] = f"{existing_reasoning} {fallback_note}".strip()

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
                failure_reason = ''
                
                # Handle "waiting in place" moves (when next_pos == current_pos)
                if next_pos == agent.position:
                    # This is a "wait" move - agent should stay in current position
                    print(f"⏸️  Agent {agent_id}: Waiting at {agent.position}")
                    success = True  # Waiting is always successful
                    failure_reason = 'wait'
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
                    # Count failed moves (safety check violations) towards collision rate
                    self.collision_count += 1
                    
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
                            self._accumulate_min_required_segment(agent_id, target_pos)
                            print(f"🎯 Agent {agent_id}: New target set to delivery point {target_pos}")

                            # Rebuild reservations and replan all active agents after target change
                            self._replan_with_reservations()
    
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
        self._accumulate_min_required_segment(agent_id, box_pos)
        if hasattr(agent, '_has_negotiated_path'):
            agent._has_negotiated_path = False
        agent.planned_path = []

        # Record task start turn for duration tracking
        self._agent_task_start_turns[agent_id] = self.current_turn

        # Record new trail start so live display and offline visualizer only show the current task
        if self.log_enabled and self.logger:
            current_path_len = len(self.logger.log_data.get('agent_paths', {}).get(str(agent_id), []))
            self._agent_trail_start[agent_id] = current_path_len
            if hasattr(self.logger, 'log_task_boundary'):
                self.logger.log_task_boundary(agent_id, current_path_len)
            if hasattr(self.logger, 'log_task_assignment'):
                self.logger.log_task_assignment(agent_id, current_path_len, box_pos, target_pos)
        else:
            self._agent_trail_start[agent_id] = 0

        print(f"🔄 Agent {agent_id}: New lifelong task — box at {box_pos}, target at {target_pos}")

        # Rebuild reservations and replan all active agents after task assignment
        self._replan_with_reservations()
    
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
        
        # Calculate path efficiency using cumulative required-vs-actual movement.
        if self.total_actual_steps > 0:
            path_efficiency = (self.total_min_required_steps / self.total_actual_steps) * 100
        elif self.total_min_required_steps == 0:
            path_efficiency = 100
        else:
            path_efficiency = 0
        
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
            'total_actual_steps': self.total_actual_steps,
            'total_min_required_steps': self.total_min_required_steps,
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
