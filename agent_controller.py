"""
Agent Controller: bridges POGEMA per-step API with A* pathfinding and LLM negotiation.
Maintains action queues and triggers replanning + negotiation when needed.
"""

import time
from collections import deque
from typing import List, Tuple, Optional, Dict, Set
from pogema import GridConfig

from pathfinding.astar import SimplePathfinder, path_to_pogema_actions, pogema_action_to_delta
from negotiation.conflict_detector import ConflictDetector
from negotiation.central_negotiator import CentralNegotiator
from negotiation.agent_validator import AgentValidator
from config import extract_walls


class LLMNegotiationController:
    """
    Per-step controller for POGEMA environments.
    Computes A* paths for all agents, detects conflicts, and triggers
    LLM negotiation when necessary. Feeds resolved paths one action
    at a time into POGEMA.
    """

    def __init__(
        self,
        grid_config: GridConfig,
        enable_negotiation: bool = True,
        enable_spatial_hints: bool = True,
    ):
        num_rows = grid_config.height
        num_cols = grid_config.width
        num_agents = grid_config.num_agents

        self.num_agents = num_agents
        self.num_rows = num_rows
        self.num_cols = num_cols

        # Extract starting positions and targets from GridConfig
        self.agent_positions: List[Tuple[int, int]] = [
            tuple(pos) for pos in grid_config.agents_xy  # type: ignore
        ]
        self.agent_targets: List[Tuple[int, int]] = [
            tuple(pos) for pos in grid_config.targets_xy  # type: ignore
        ]
        self.agent_done: List[bool] = [False] * num_agents

        # Extract wall positions
        self.walls: Set[Tuple[int, int]] = extract_walls(grid_config)

        # Core components
        self.pathfinder = SimplePathfinder(num_rows=num_rows, num_cols=num_cols)
        self.conflict_detector = ConflictDetector(num_rows=num_rows, num_cols=num_cols)

        self.enable_negotiation = enable_negotiation
        if enable_negotiation:
            self.negotiator = CentralNegotiator(enable_spatial_hints=enable_spatial_hints)
            self.validators = {i: AgentValidator() for i in range(num_agents)}
        else:
            self.negotiator = None
            self.validators = {}

        # Action queues — one deque per agent, containing POGEMA action ints
        self.action_queues: Dict[int, deque] = {i: deque() for i in range(num_agents)}

        # LLM stats tracking
        self._negotiation_count = 0
        self._negotiation_durations: List[float] = []
        self._total_tokens = 0

    # --- Main entry point ---

    def get_actions(
        self,
        observations,
        terminated: List[bool],
        step_number: int,
    ) -> List[int]:
        """
        Called once per POGEMA step.
        Returns a list of POGEMA actions (0–4), one per agent.
        """
        # Mark newly done agents
        for i in range(self.num_agents):
            if terminated[i]:
                self.agent_done[i] = True

        # Check if any active agent has an empty queue → replan
        active_agents = [i for i in range(self.num_agents) if not self.agent_done[i]]
        needs_replan = any(len(self.action_queues[i]) == 0 for i in active_agents)

        if needs_replan and active_agents:
            self._replan(step_number)

        # Pop one action per agent
        actions = []
        for i in range(self.num_agents):
            if self.agent_done[i]:
                actions.append(0)  # idle
            elif self.action_queues[i]:
                actions.append(self.action_queues[i].popleft())
            else:
                actions.append(0)  # idle if queue still empty after replan

        return actions

    def update_positions(self, env=None, actions: Optional[List[int]] = None):
        """
        Sync internal position tracking after a POGEMA step.
        Prefers env.get_agents_xy() if available; falls back to applying action deltas.
        """
        if env is not None:
            inner = getattr(env, 'unwrapped', env)
            if hasattr(inner, 'get_agents_xy'):
                try:
                    r = inner.grid_config.obs_radius
                    positions = inner.get_agents_xy()
                    for i, pos in enumerate(positions):
                        # Subtract obs_radius offset added by POGEMA's artificial border
                        self.agent_positions[i] = (pos[0] - r, pos[1] - r)
                    return
                except Exception:
                    pass

        # Fallback: apply delta from the last action
        if actions is not None:
            for i in range(self.num_agents):
                if not self.agent_done[i]:
                    dr, dc = pogema_action_to_delta(actions[i])
                    r, c = self.agent_positions[i]
                    nr, nc = r + dr, c + dc
                    # Only apply if new position is within bounds and not a wall
                    if (
                        0 <= nr < self.num_rows
                        and 0 <= nc < self.num_cols
                        and (nr, nc) not in self.walls
                    ):
                        self.agent_positions[i] = (nr, nc)

    # --- Internal replanning logic ---

    def _replan(self, step_number: int):
        """Compute A* paths for all active agents, detect conflicts, negotiate if needed."""
        active = [i for i in range(self.num_agents) if not self.agent_done[i]]
        if not active:
            return

        # Run A* for each active agent (ignoring other agents as static obstacles)
        agent_paths: Dict[int, List[Tuple[int, int]]] = {}
        for i in active:
            start = self.agent_positions[i]
            goal = self.agent_targets[i]

            if start == goal:
                self.agent_done[i] = True
                continue

            # Other active agent positions as soft obstacles
            other_positions = {
                j: self.agent_positions[j]
                for j in active
                if j != i
            }

            path = self.pathfinder.find_path_with_obstacles(
                start=start,
                goal=goal,
                walls=self.walls,
                agent_positions=other_positions,
                exclude_agent=i,
            )

            if not path:
                # Try without agent obstacles
                path = self.pathfinder.find_path(start, goal, self.walls)

            if path:
                agent_paths[i] = path

        if not agent_paths:
            return

        # Detect conflicts
        conflicts = self.conflict_detector.detect_path_conflicts(agent_paths, step_number)

        if conflicts['has_conflicts'] and self.enable_negotiation and self.negotiator:
            self._run_negotiation(agent_paths, conflicts, step_number)
        else:
            # No conflict — fill queues from A* paths
            for i, path in agent_paths.items():
                actions = path_to_pogema_actions(path)
                self.action_queues[i].extend(actions)

    def _run_negotiation(
        self,
        agent_paths: Dict[int, List[Tuple[int, int]]],
        conflicts: Dict,
        step_number: int,
    ):
        """Call CentralNegotiator and populate action queues from resolution."""
        conflicting_agents = list(set(conflicts['conflicting_agents']))

        # Build conflict_data in the format CentralNegotiator expects
        agents_data = []
        for i in conflicting_agents:
            if i not in agent_paths:
                continue
            agents_data.append({
                'id': i,
                'current_pos': list(self.agent_positions[i]),
                'target_pos': list(self.agent_targets[i]),
                'planned_path': [list(p) for p in agent_paths[i]],
            })

        # All agents' positions for map context
        all_agent_positions = {
            i: list(self.agent_positions[i])
            for i in range(self.num_agents)
            if not self.agent_done[i]
        }

        # Build grid representation (list of lists of int)
        grid = []
        for r in range(self.num_rows):
            row = []
            for c in range(self.num_cols):
                row.append(1 if (r, c) in self.walls else 0)
            grid.append(row)

        conflict_data = {
            'turn': step_number,
            'agents': agents_data,
            'conflict_points': [list(p) for p in conflicts['conflict_points']],
            'map_state': {
                'agents': all_agent_positions,
                'targets': {i: list(self.agent_targets[i]) for i in range(self.num_agents) if not self.agent_done[i]},
                'grid': grid,
            },
        }

        # Build validator callables for conflicting agents
        agent_validators = {
            i: self.validators[i].validate_negotiated_action
            for i in conflicting_agents
            if i in self.validators
        }

        t0 = time.time()
        plan, history, prompts = self.negotiator.negotiate_path_conflict(
            conflict_data, agent_validators
        )
        duration = time.time() - t0

        self._negotiation_count += 1
        self._negotiation_durations.append(duration)

        # Track token usage if available
        if self.negotiator and self.negotiator.client:
            usage = self.negotiator.client.get_token_usage()
            self._total_tokens = usage.get('total_tokens', 0)

        # Populate queues from negotiated plan
        resolved_agents = set()
        if plan:
            for agent_id_str, action_data in plan.items():
                try:
                    i = int(agent_id_str)
                except (ValueError, TypeError):
                    continue

                if i not in range(self.num_agents) or self.agent_done[i]:
                    continue

                resolved_agents.add(i)
                path = action_data.get('path', [])
                if action_data.get('action') == 'wait' or not path:
                    self.action_queues[i].append(0)  # idle one step
                else:
                    rc_path = [tuple(p) for p in path]
                    # Only include steps starting from current position
                    if rc_path and rc_path[0] != self.agent_positions[i]:
                        rc_path = [self.agent_positions[i]] + rc_path
                    actions = path_to_pogema_actions(rc_path)
                    self.action_queues[i].extend(actions if actions else [0])

        # Non-conflicting agents: use A* path as-is
        for i, path in agent_paths.items():
            if i not in conflicting_agents:
                actions = path_to_pogema_actions(path)
                self.action_queues[i].extend(actions)

        # Conflicting agents with no resolution: idle one step
        for i in conflicting_agents:
            if i not in resolved_agents and not self.agent_done[i]:
                if len(self.action_queues[i]) == 0:
                    self.action_queues[i].append(0)

    # --- Stats ---

    def get_llm_stats(self) -> Dict:
        """Return LLM negotiation statistics."""
        return {
            'negotiation_count': self._negotiation_count,
            'total_tokens_used': self._total_tokens,
            'negotiation_durations_sec': self._negotiation_durations,
            'avg_negotiation_duration_sec': (
                sum(self._negotiation_durations) / len(self._negotiation_durations)
                if self._negotiation_durations
                else 0.0
            ),
        }
