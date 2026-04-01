"""
Pluggable multi-agent path planners.

MAPF-LNS2 porting references used for this module:
- https://raw.githubusercontent.com/Jiaoyang-Li/MAPF-LNS2/master/src/LNS.cpp
  - run(), getInitialSolution(), chooseDestroyHeuristicbyALNS(),
    generateNeighborByRandomWalk(), generateNeighborByIntersection()
- https://raw.githubusercontent.com/Jiaoyang-Li/MAPF-LNS2/master/inc/LNS.h
  - destroy_heuristic enum, solver fields, LNS loop structure
- https://raw.githubusercontent.com/Jiaoyang-Li/MAPF-LNS2/master/inc/BasicLNS.h
  - adaptive-LNS weight controls and neighborhood bookkeeping
"""

import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..agents import RobotAgent
from . import ConflictDetector, SimplePathfinder
from .minicbs import MiniCBSRepair
from .path_table import PathTable
from .sipp import SIPPLowLevelSolver

PLANNER_STATUS_SUCCESS = "success"
PLANNER_STATUS_PARTIAL_SUCCESS = "partial_success"
PLANNER_STATUS_FAILED_NO_SOLUTION = "failed_no_solution"


@dataclass
class PlannerResult:
    """Minimal planner result contract."""

    solutions: Dict[int, List[Tuple[int, int]]]
    status: str


class MultiAgentPlanner(ABC):
    """Base interface for all multi-agent path planners."""

    @abstractmethod
    def plan_all(
        self,
        agents: Dict[int, RobotAgent],
        map_state: Dict,
        agent_ids: Optional[Set[int]] = None,
    ) -> PlannerResult:
        pass

    @abstractmethod
    def replan_subset(
        self,
        agents: Dict[int, RobotAgent],
        map_state: Dict,
        agent_ids: Set[int],
    ) -> PlannerResult:
        pass

    @abstractmethod
    def supports_dynamic_repair(self) -> bool:
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        pass


class AStarReservationPlanner(MultiAgentPlanner):
    """Sequential reservation-based A* planner (existing behavior)."""

    def __init__(
        self,
        pathfinder: SimplePathfinder,
        width: int,
        height: int,
        min_time_horizon: int,
        grid_horizon_multiplier: int,
        low_level_backend: str = "astar",
    ):
        self.pathfinder = pathfinder
        self.width = width
        self.height = height
        self.min_time_horizon = min_time_horizon
        self.grid_horizon_multiplier = grid_horizon_multiplier
        self.low_level_backend = (low_level_backend or "astar").strip().lower()
        self.sipp_solver = SIPPLowLevelSolver(pathfinder)

    def get_backend_name(self) -> str:
        return "astar"

    def supports_dynamic_repair(self) -> bool:
        return True

    def _extract_walls(self, map_state: Dict) -> Set[Tuple[int, int]]:
        grid = map_state.get('grid', [])
        walls = set()
        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                if cell == '#':
                    walls.add((x, y))
        return walls

    def _ordered_active_ids(
        self,
        agents: Dict[int, RobotAgent],
        agent_ids: Optional[Set[int]],
    ) -> List[int]:
        if agent_ids is None:
            candidate_ids = set(agents.keys())
        else:
            candidate_ids = set(agent_ids)

        active_ids: List[int] = []
        for aid in sorted(candidate_ids):
            agent = agents.get(aid)
            if agent and (not agent.is_waiting) and agent.target_position:
                active_ids.append(aid)
        return active_ids

    def _infer_status(self, requested_count: int, solved_count: int) -> str:
        if requested_count == 0:
            return PLANNER_STATUS_SUCCESS
        if solved_count == 0:
            return PLANNER_STATUS_FAILED_NO_SOLUTION
        if solved_count < requested_count:
            return PLANNER_STATUS_PARTIAL_SUCCESS
        return PLANNER_STATUS_SUCCESS

    def plan_with_order(
        self,
        agents: Dict[int, RobotAgent],
        map_state: Dict,
        ordered_agent_ids: List[int],
    ) -> Dict[int, List[Tuple[int, int]]]:
        if not ordered_agent_ids:
            return {}
        return self._plan_with_fixed_reservations(
            agents,
            map_state,
            ordered_agent_ids,
            reserved_positions_by_turn={},
            reserved_edges_by_turn={},
            path_table=PathTable(),
        )

    def _plan_with_fixed_reservations(
        self,
        agents: Dict[int, RobotAgent],
        map_state: Dict,
        planned_agent_ids: List[int],
        reserved_positions_by_turn: Dict[int, Set[Tuple[int, int]]],
        reserved_edges_by_turn: Dict[int, Set[Tuple[Tuple[int, int], Tuple[int, int]]]],
        path_table: Optional[PathTable] = None,
        existing_planned_moves: Optional[Dict[int, List[Tuple[int, int]]]] = None,
    ) -> Dict[int, List[Tuple[int, int]]]:
        planned_moves: Dict[int, List[Tuple[int, int]]] = {}
        if existing_planned_moves:
            planned_moves.update(existing_planned_moves)
        path_table = path_table or PathTable()

        walls = self._extract_walls(map_state)
        planning_time_horizon = max(
            self.min_time_horizon,
            self.width * self.height * self.grid_horizon_multiplier,
        )

        for agent_id in planned_agent_ids:
            agent = agents[agent_id]
            goal = agent.target_position
            if goal is None:
                continue
            has_negotiated_path = (
                hasattr(agent, '_has_negotiated_path')
                and getattr(agent, '_has_negotiated_path', False)
                and agent.planned_path
                and len(agent.planned_path) > 1
            )

            if has_negotiated_path:
                path = agent.planned_path.copy()
            else:
                if self.low_level_backend == "sipp":
                    path = self.sipp_solver.find_path(
                        start=agent.position,
                        goal=goal,
                        walls=walls,
                        path_table=path_table,
                        max_time_steps=planning_time_horizon,
                    )
                else:
                    path = self.pathfinder.find_path_with_time_constraints(
                        start=agent.position,
                        goal=goal,
                        walls=walls,
                        reserved_positions_by_turn=reserved_positions_by_turn,
                        reserved_edges_by_turn=reserved_edges_by_turn,
                        max_time_steps=planning_time_horizon,
                        path_table=path_table,
                    )

                if not path:
                    other_positions = {
                        oid: oagent.position
                        for oid, oagent in agents.items()
                        if oid != agent_id
                    }
                    path = self.pathfinder.find_path_with_obstacles(
                        start=agent.position,
                        goal=goal,
                        walls=walls,
                        agent_positions=other_positions,
                        exclude_agent=agent_id,
                    )

                agent.planned_path = path
                if hasattr(agent, '_has_negotiated_path'):
                    agent._has_negotiated_path = False

            if not path:
                continue

            planned_moves[agent_id] = path.copy()
            path_table.insert_path(agent_id, path, hold_until=planning_time_horizon)
            for turn_idx, pos in enumerate(path):
                reserved_positions_by_turn.setdefault(turn_idx, set()).add(pos)
                if turn_idx > 0:
                    prev = path[turn_idx - 1]
                    reserved_edges_by_turn.setdefault(turn_idx - 1, set()).add((prev, pos))

        return planned_moves

    def plan_all(
        self,
        agents: Dict[int, RobotAgent],
        map_state: Dict,
        agent_ids: Optional[Set[int]] = None,
    ) -> PlannerResult:
        active_ids = self._ordered_active_ids(agents, agent_ids)
        if not active_ids:
            return PlannerResult(solutions={}, status=PLANNER_STATUS_SUCCESS)
        planned = self._plan_with_fixed_reservations(
            agents,
            map_state,
            active_ids,
            reserved_positions_by_turn={},
            reserved_edges_by_turn={},
            path_table=PathTable(),
        )
        status = self._infer_status(len(active_ids), len(planned))
        return PlannerResult(solutions=planned, status=status)

    def replan_subset(
        self,
        agents: Dict[int, RobotAgent],
        map_state: Dict,
        agent_ids: Set[int],
    ) -> PlannerResult:
        if not agent_ids:
            return PlannerResult(solutions={}, status=PLANNER_STATUS_SUCCESS)

        subset_ids = self._ordered_active_ids(agents, set(agent_ids))
        if not subset_ids:
            return PlannerResult(solutions={}, status=PLANNER_STATUS_FAILED_NO_SOLUTION)

        reserved_positions_by_turn: Dict[int, Set[Tuple[int, int]]] = {}
        reserved_edges_by_turn: Dict[int, Set[Tuple[Tuple[int, int], Tuple[int, int]]]] = {}
        path_table = PathTable()

        # Seed reservation tables with existing paths from non-replanned active agents.
        reservation_horizon = max(
            self.min_time_horizon,
            self.width * self.height * self.grid_horizon_multiplier,
        )
        for aid, agent in agents.items():
            if aid in subset_ids or agent.is_waiting or not agent.target_position:
                continue
            existing_path = agent.planned_path if agent.planned_path else [agent.position]
            path_table.insert_path(aid, existing_path, hold_until=reservation_horizon)
            for turn_idx, pos in enumerate(existing_path):
                reserved_positions_by_turn.setdefault(turn_idx, set()).add(pos)
                if turn_idx > 0:
                    prev = existing_path[turn_idx - 1]
                    reserved_edges_by_turn.setdefault(turn_idx - 1, set()).add((prev, pos))
            if existing_path:
                final_pos = existing_path[-1]
                for turn_idx in range(len(existing_path), reservation_horizon + 1):
                    reserved_positions_by_turn.setdefault(turn_idx, set()).add(final_pos)

        replanned = self._plan_with_fixed_reservations(
            agents,
            map_state,
            subset_ids,
            reserved_positions_by_turn=reserved_positions_by_turn,
            reserved_edges_by_turn=reserved_edges_by_turn,
            path_table=path_table,
        )
        filtered = {aid: replanned[aid] for aid in subset_ids if aid in replanned}
        status = self._infer_status(len(subset_ids), len(filtered))
        return PlannerResult(solutions=filtered, status=status)


class LNS2Planner(MultiAgentPlanner):
    """Python port structure based on MAPF-LNS2 `LNS` run loop in `src/LNS.cpp`."""
    # Sentinel used when a solution is missing; kept large so any valid solution wins.
    _INFINITE_COST = 10**9
    # Match MAPF-LNS2's bounded repeated random-walk attempts when expanding neighborhood.
    _RANDOMWALK_EXPANSION_ATTEMPTS = 10
    _DESTROY_RANDOMAGENTS = "randomagents"
    _DESTROY_RANDOMWALK = "randomwalk"
    _DESTROY_INTERSECTION = "intersection"

    def __init__(
        self,
        base_planner: AStarReservationPlanner,
        width: int,
        height: int,
        iterations: int = 8,
        destroy_ratio: float = 0.35,
        seed: int = 0,
        destroy_strategy: str = "randomwalk",
        phase1_ratio: float = 0.7,
        max_init_retries: int = 3,
        adaptive_stall_iterations: int = 3,
        destroy_ratio_step: float = 0.1,
        repair_backend: str = "minicbs",
        minicbs_max_nodes: int = 128,
        low_level_backend: str = "astar",
    ):
        self.base_planner = base_planner
        self.width = width
        self.height = height
        self.iterations = max(1, iterations)
        self.destroy_ratio = min(max(destroy_ratio, 0.1), 0.8)
        self.phase1_ratio = min(max(phase1_ratio, 0.1), 0.9)
        self.max_init_retries = max(1, max_init_retries)
        self.adaptive_stall_iterations = max(1, adaptive_stall_iterations)
        self.destroy_ratio_step = min(max(destroy_ratio_step, 0.01), 0.3)
        self.repair_backend = (repair_backend or "minicbs").strip().lower()
        self.low_level_backend = (low_level_backend or "astar").strip().lower()
        self.conflict_detector = ConflictDetector(width, height)
        self.random = random.Random(seed)
        self.destroy_strategy = (destroy_strategy or "randomwalk").strip().lower()
        self.minicbs_repair = MiniCBSRepair(
            width=width,
            height=height,
            min_time_horizon=base_planner.min_time_horizon,
            grid_horizon_multiplier=base_planner.grid_horizon_multiplier,
            max_nodes=minicbs_max_nodes,
        )

        # Adaptive-LNS controls (mirrors MAPF-LNS2 Adaptive destroy mode)
        self.alns = self.destroy_strategy == "adaptive"
        self.destroy_weights = [1.0, 1.0, 1.0]
        self.decay_factor = 0.01
        self.reaction_factor = 0.01
        self.selected_neighbor = 0

        # RandomWalk-style tabu memory for repeated troublemaker selection.
        self._tabu_list: List[int] = []
        self._tabu_set: Set[int] = set()
        self._tabu_tenure = 5

    def get_backend_name(self) -> str:
        return "LNS2"

    def supports_dynamic_repair(self) -> bool:
        return True

    def _path_cost(self, path: List[Tuple[int, int]]) -> int:
        return max(0, len(path) - 1)

    def _conflict_count(self, paths: Dict[int, List[Tuple[int, int]]], current_turn: int = 0) -> int:
        if not paths:
            return self._INFINITE_COST
        conflict_info = self.conflict_detector.detect_path_conflicts(paths, current_turn)
        return len(conflict_info.get('conflict_points', []))

    def _soc_cost(self, paths: Dict[int, List[Tuple[int, int]]]) -> int:
        if not paths:
            return self._INFINITE_COST
        return sum(self._path_cost(path) for path in paths.values())

    def _solution_metrics(self, paths: Dict[int, List[Tuple[int, int]]]) -> Tuple[int, int]:
        return self._conflict_count(paths, 0), self._soc_cost(paths)

    def _agent_delay(
        self,
        agent_id: int,
        solution: Dict[int, List[Tuple[int, int]]],
        agents: Dict[int, RobotAgent],
    ) -> int:
        path = solution.get(agent_id, [])
        agent = agents.get(agent_id)
        if not path or not agent or not agent.target_position:
            return 0
        start = path[0]
        target = agent.target_position
        heuristic = abs(start[0] - target[0]) + abs(start[1] - target[1])
        waits = 0
        for idx in range(1, len(path)):
            if path[idx] == path[idx - 1]:
                waits += 1
        return max(0, self._path_cost(path) - heuristic) + waits

    def _reset_tabu(self, agent_count: int) -> None:
        self._tabu_tenure = max(5, agent_count // 4)
        self._tabu_list = []
        self._tabu_set = set()

    def _mark_tabu(self, agent_id: int) -> None:
        if agent_id in self._tabu_set:
            return
        self._tabu_set.add(agent_id)
        self._tabu_list.append(agent_id)
        while len(self._tabu_list) > self._tabu_tenure:
            expired = self._tabu_list.pop(0)
            self._tabu_set.discard(expired)

    def _find_most_delayed_agent(
        self,
        solution: Dict[int, List[Tuple[int, int]]],
        agents: Dict[int, RobotAgent],
    ) -> Optional[int]:
        best_agent: Optional[int] = None
        best_delay = -1
        for aid in solution.keys():
            if aid in self._tabu_set:
                continue
            delay = self._agent_delay(aid, solution, agents)
            if delay > best_delay:
                best_delay = delay
                best_agent = aid

        if best_agent is None:
            # Reset tabu when all candidates are blocked and retry once.
            self._tabu_set.clear()
            self._tabu_list.clear()
            for aid in solution.keys():
                delay = self._agent_delay(aid, solution, agents)
                if delay > best_delay:
                    best_delay = delay
                    best_agent = aid

        if best_delay <= 0:
            return None
        self._mark_tabu(best_agent)
        return best_agent

    def _select_random_agents_subset(self, solution: Dict[int, List[Tuple[int, int]]]) -> Set[int]:
        if not solution:
            return set()

        agent_ids = list(solution.keys())
        subset_size = max(1, int(len(agent_ids) * self.destroy_ratio))

        by_path_length = sorted(
            agent_ids,
            key=lambda aid: len(solution.get(aid, [])),
            reverse=True,
        )
        selected: List[int] = by_path_length[: min(subset_size, max(1, subset_size // 2))]
        remaining = [aid for aid in agent_ids if aid not in selected]
        self.random.shuffle(remaining)
        selected.extend(remaining[: max(0, subset_size - len(selected))])
        return set(selected)

    def _select_intersection_subset(self, solution: Dict[int, List[Tuple[int, int]]]) -> Set[int]:
        if not solution:
            return set()

        subset_size = max(1, int(len(solution) * self.destroy_ratio))
        conflict_info = self.conflict_detector.detect_path_conflicts(solution, 0)
        conflict_points = conflict_info.get('conflict_points', [])
        if not conflict_points:
            return self._select_random_agents_subset(solution)

        pivot = self.random.choice(conflict_points)
        selected = {
            aid for aid, path in solution.items()
            if pivot in path
        }
        if len(selected) < subset_size:
            remaining = [aid for aid in solution.keys() if aid not in selected]
            self.random.shuffle(remaining)
            selected.update(remaining[: subset_size - len(selected)])
        return set(list(selected)[:subset_size])

    def _select_randomwalk_subset(
        self,
        solution: Dict[int, List[Tuple[int, int]]],
        agents: Dict[int, RobotAgent],
    ) -> Set[int]:
        if not solution:
            return set()
        subset_size = max(1, int(len(solution) * self.destroy_ratio))
        seed_agent = self._find_most_delayed_agent(solution, agents)
        if seed_agent is None:
            return self._select_random_agents_subset(solution)

        selected: Set[int] = {seed_agent}
        seed_path = solution.get(seed_agent, [])
        if not seed_path:
            return self._select_random_agents_subset(solution)

        for _ in range(self._RANDOMWALK_EXPANSION_ATTEMPTS):
            if len(selected) >= subset_size:
                break
            t = self.random.randrange(len(seed_path))
            loc = seed_path[t]
            prev_loc = seed_path[t - 1] if t > 0 else loc
            for aid, path in solution.items():
                if aid in selected:
                    continue
                if t < len(path):
                    pos_t = path[t]
                    prev_t = path[t - 1] if t > 0 else path[t]
                else:
                    pos_t = path[-1] if path else None
                    prev_t = path[-1] if path else None
                if pos_t is None:
                    continue
                vertex_conflict = pos_t == loc
                edge_conflict = t > 0 and prev_t == loc and pos_t == prev_loc
                if vertex_conflict or edge_conflict:
                    selected.add(aid)
                if len(selected) >= subset_size:
                    break

        if len(selected) < subset_size:
            remaining = [aid for aid in solution.keys() if aid not in selected]
            self.random.shuffle(remaining)
            selected.update(remaining[: subset_size - len(selected)])
        return set(selected)

    def _choose_destroy_subset(
        self,
        solution: Dict[int, List[Tuple[int, int]]],
        agents: Dict[int, RobotAgent],
    ) -> Set[int]:
        # Mirrors chooseDestroyHeuristicbyALNS() in MAPF-LNS2/src/LNS.cpp.
        if self.alns:
            total = sum(self.destroy_weights)
            threshold = self.random.random() * total
            running = 0.0
            for idx, weight in enumerate(self.destroy_weights):
                running += weight
                if running >= threshold:
                    self.selected_neighbor = idx
                    break
            strategies = [
                self._DESTROY_RANDOMWALK,
                self._DESTROY_INTERSECTION,
                self._DESTROY_RANDOMAGENTS,
            ]
            chosen_strategy = strategies[self.selected_neighbor]
        else:
            chosen_strategy = self.destroy_strategy

        if chosen_strategy == self._DESTROY_INTERSECTION:
            return self._select_intersection_subset(solution)
        if chosen_strategy == self._DESTROY_RANDOMAGENTS:
            return self._select_random_agents_subset(solution)
        return self._select_randomwalk_subset(solution, agents)

    def _repair_subset(
        self,
        agents: Dict[int, RobotAgent],
        map_state: Dict,
        subset: Set[int],
    ) -> Dict[int, List[Tuple[int, int]]]:
        if self.repair_backend == "minicbs":
            repaired = self.minicbs_repair.replan_subset(agents, map_state, subset)
            if repaired:
                return repaired
        return self.base_planner.replan_subset(agents, map_state, subset).solutions

    def _infer_status(self, requested_count: int, solved_count: int) -> str:
        if requested_count == 0:
            return PLANNER_STATUS_SUCCESS
        if solved_count == 0:
            return PLANNER_STATUS_FAILED_NO_SOLUTION
        if solved_count < requested_count:
            return PLANNER_STATUS_PARTIAL_SUCCESS
        return PLANNER_STATUS_SUCCESS

    def _merge_solution(
        self,
        base_solution: Dict[int, List[Tuple[int, int]]],
        repaired_subset: Dict[int, List[Tuple[int, int]]],
    ) -> Dict[int, List[Tuple[int, int]]]:
        merged = {aid: path.copy() for aid, path in base_solution.items()}
        for aid, path in repaired_subset.items():
            merged[aid] = path.copy()
        return merged

    def _update_adaptive_weights(self, old_subset_cost: int, new_subset_cost: int, subset_size: int) -> None:
        if not self.alns or subset_size <= 0:
            return
        if new_subset_cost < old_subset_cost:
            reward = (old_subset_cost - new_subset_cost) / float(subset_size)
            self.destroy_weights[self.selected_neighbor] = (
                self.reaction_factor * reward
                + (1.0 - self.reaction_factor) * self.destroy_weights[self.selected_neighbor]
            )
        else:
            self.destroy_weights[self.selected_neighbor] = (
                (1.0 - self.decay_factor) * self.destroy_weights[self.selected_neighbor]
            )

    def _reorder_for_retry(
        self,
        ordered_agent_ids: List[int],
        previous_conflicting_agents: Set[int],
    ) -> List[int]:
        shuffled = ordered_agent_ids.copy()
        self.random.shuffle(shuffled)
        if not previous_conflicting_agents:
            return shuffled

        front = [aid for aid in shuffled if aid in previous_conflicting_agents]
        back = [aid for aid in shuffled if aid not in previous_conflicting_agents]
        return front + back

    def _get_initial_solution(
        self,
        agents: Dict[int, RobotAgent],
        map_state: Dict,
        agent_ids: Optional[Set[int]],
    ) -> Dict[int, List[Tuple[int, int]]]:
        ordered_agent_ids = self.base_planner._ordered_active_ids(agents, agent_ids)
        if not ordered_agent_ids:
            return {}

        best_solution: Dict[int, List[Tuple[int, int]]] = {}
        best_conflicts = self._INFINITE_COST
        best_soc = self._INFINITE_COST
        previous_conflicting_agents: Set[int] = set()

        for attempt in range(self.max_init_retries):
            if attempt == 0:
                order = ordered_agent_ids
            else:
                order = self._reorder_for_retry(ordered_agent_ids, previous_conflicting_agents)

            candidate = self.base_planner.plan_with_order(agents, map_state, order)
            if not candidate:
                continue

            candidate_conflicts, candidate_soc = self._solution_metrics(candidate)
            if (
                candidate_conflicts < best_conflicts
                or (candidate_conflicts == best_conflicts and candidate_soc < best_soc)
            ):
                best_solution = {aid: path.copy() for aid, path in candidate.items()}
                best_conflicts = candidate_conflicts
                best_soc = candidate_soc

            conflict_info = self.conflict_detector.detect_path_conflicts(candidate, 0)
            previous_conflicting_agents = set(conflict_info.get('conflicting_agents', []))

            if candidate_conflicts == 0 and len(candidate) == len(ordered_agent_ids):
                break

        return best_solution

    def _accept_phase_one(
        self,
        candidate_conflicts: int,
        candidate_soc: int,
        best_conflicts: int,
        best_soc: int,
    ) -> bool:
        if candidate_conflicts < best_conflicts:
            return True
        if candidate_conflicts > best_conflicts:
            return False
        if candidate_soc <= best_soc:
            return True
        # Allow limited exploration when C stays unchanged.
        allowance = max(1, int(best_soc * 0.05))
        return candidate_soc <= best_soc + allowance

    def plan_all(
        self,
        agents: Dict[int, RobotAgent],
        map_state: Dict,
        agent_ids: Optional[Set[int]] = None,
    ) -> PlannerResult:
        # Robust initial solution: randomized priority retries with conflict-aware reseeding.
        initial_solution = self._get_initial_solution(agents, map_state, agent_ids)
        if not initial_solution:
            return PlannerResult(solutions={}, status=PLANNER_STATUS_FAILED_NO_SOLUTION)

        self._reset_tabu(len(initial_solution))
        best_solution = {aid: path.copy() for aid, path in initial_solution.items()}
        best_conflicts, best_soc = self._solution_metrics(best_solution)

        phase1_iterations = max(1, int(self.iterations * self.phase1_ratio))
        phase2_iterations = max(0, self.iterations - phase1_iterations)
        stalled_iterations = 0

        # Phase 1: reduce conflict count C (best-effort).
        for _ in range(phase1_iterations):
            subset = self._choose_destroy_subset(best_solution, agents)
            if not subset:
                stalled_iterations += 1
                continue

            old_subset_cost = sum(
                self._path_cost(best_solution.get(aid, []))
                for aid in subset
            )
            repaired_subset = self._repair_subset(agents, map_state, subset)
            if not repaired_subset:
                stalled_iterations += 1
                if stalled_iterations >= self.adaptive_stall_iterations:
                    self.destroy_ratio = min(0.8, self.destroy_ratio + self.destroy_ratio_step)
                    stalled_iterations = 0
                continue

            new_subset_cost = sum(self._path_cost(path) for path in repaired_subset.values())
            candidate_solution = self._merge_solution(best_solution, repaired_subset)
            candidate_conflicts, candidate_soc = self._solution_metrics(candidate_solution)
            self._update_adaptive_weights(old_subset_cost, new_subset_cost, len(subset))

            if self._accept_phase_one(candidate_conflicts, candidate_soc, best_conflicts, best_soc):
                best_solution = candidate_solution
                best_conflicts = candidate_conflicts
                best_soc = candidate_soc
                stalled_iterations = 0
            else:
                stalled_iterations += 1
                if stalled_iterations >= self.adaptive_stall_iterations:
                    self.destroy_ratio = min(0.8, self.destroy_ratio + self.destroy_ratio_step)
                    stalled_iterations = 0

        # Phase 2: optimize SOC S while preserving best-achieved C.
        for _ in range(phase2_iterations):
            subset = self._choose_destroy_subset(best_solution, agents)
            if not subset:
                continue

            repaired_subset = self._repair_subset(agents, map_state, subset)
            if not repaired_subset:
                continue

            candidate_solution = self._merge_solution(best_solution, repaired_subset)
            candidate_conflicts, candidate_soc = self._solution_metrics(candidate_solution)
            if candidate_conflicts == best_conflicts and candidate_soc < best_soc:
                best_solution = candidate_solution
                best_conflicts = candidate_conflicts
                best_soc = candidate_soc

        status = PLANNER_STATUS_SUCCESS if best_conflicts == 0 else PLANNER_STATUS_PARTIAL_SUCCESS
        return PlannerResult(solutions=best_solution, status=status)

    def replan_subset(
        self,
        agents: Dict[int, RobotAgent],
        map_state: Dict,
        agent_ids: Set[int],
    ) -> PlannerResult:
        if not agent_ids:
            return PlannerResult(solutions={}, status=PLANNER_STATUS_SUCCESS)
        ordered_subset = self.base_planner._ordered_active_ids(agents, set(agent_ids))
        repaired_subset = self._repair_subset(agents, map_state, set(agent_ids))
        status = self._infer_status(len(ordered_subset), len(repaired_subset))
        return PlannerResult(solutions=repaired_subset, status=status)


def create_multi_agent_planner(
    mode: str,
    pathfinder: SimplePathfinder,
    width: int,
    height: int,
    min_time_horizon: int,
    grid_horizon_multiplier: int,
) -> MultiAgentPlanner:
    normalized = (mode or "astar").strip().lower()
    base = AStarReservationPlanner(
        pathfinder=pathfinder,
        width=width,
        height=height,
        min_time_horizon=min_time_horizon,
        grid_horizon_multiplier=grid_horizon_multiplier,
        low_level_backend="astar",
    )
    if normalized == "lns2":
        iterations = int(os.getenv('LNS2_MAX_ITERATIONS', '8'))
        destroy_ratio = float(os.getenv('LNS2_DESTROY_RATIO', '0.35'))
        seed = int(os.getenv('LNS2_RANDOM_SEED', '0'))
        destroy_strategy = os.getenv('LNS2_DESTROY_STRATEGY', 'randomwalk')
        phase1_ratio = float(os.getenv('LNS2_PHASE1_RATIO', '0.7'))
        max_init_retries = int(os.getenv('LNS2_MAX_INIT_RETRIES', '3'))
        adaptive_stall_iterations = int(os.getenv('LNS2_ADAPTIVE_STALL_ITERS', '3'))
        destroy_ratio_step = float(os.getenv('LNS2_DESTROY_RATIO_STEP', '0.1'))
        repair_backend = os.getenv('LNS2_REPAIR_BACKEND', 'minicbs')
        minicbs_max_nodes = int(os.getenv('LNS2_MINICBS_MAX_NODES', '128'))
        low_level_backend = os.getenv('LNS2_LOW_LEVEL_BACKEND', 'astar').strip().lower()
        if low_level_backend not in {'astar', 'sipp'}:
            low_level_backend = 'astar'
        base.low_level_backend = low_level_backend
        return LNS2Planner(
            base_planner=base,
            width=width,
            height=height,
            iterations=iterations,
            destroy_ratio=destroy_ratio,
            seed=seed,
            destroy_strategy=destroy_strategy,
            phase1_ratio=phase1_ratio,
            max_init_retries=max_init_retries,
            adaptive_stall_iterations=adaptive_stall_iterations,
            destroy_ratio_step=destroy_ratio_step,
            repair_backend=repair_backend,
            minicbs_max_nodes=minicbs_max_nodes,
            low_level_backend=low_level_backend,
        )
    return base
