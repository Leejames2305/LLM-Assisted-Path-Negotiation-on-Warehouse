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
from typing import Dict, List, Optional, Set, Tuple

from . import ConflictDetector, SimplePathfinder


class MultiAgentPlanner(ABC):
    """Base interface for all multi-agent path planners."""

    @abstractmethod
    def plan_all(
        self,
        agents: Dict[int, object],
        map_state: Dict,
        agent_ids: Optional[Set[int]] = None,
    ) -> Dict[int, List[Tuple[int, int]]]:
        pass

    @abstractmethod
    def replan_subset(
        self,
        agents: Dict[int, object],
        map_state: Dict,
        agent_ids: Set[int],
    ) -> Dict[int, List[Tuple[int, int]]]:
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
    ):
        self.pathfinder = pathfinder
        self.width = width
        self.height = height
        self.min_time_horizon = min_time_horizon
        self.grid_horizon_multiplier = grid_horizon_multiplier

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
        agents: Dict[int, object],
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

    def _plan_with_fixed_reservations(
        self,
        agents: Dict[int, object],
        map_state: Dict,
        planned_agent_ids: List[int],
        reserved_positions_by_turn: Dict[int, Set[Tuple[int, int]]],
        reserved_edges_by_turn: Dict[int, Set[Tuple[Tuple[int, int], Tuple[int, int]]]],
        existing_planned_moves: Optional[Dict[int, List[Tuple[int, int]]]] = None,
    ) -> Dict[int, List[Tuple[int, int]]]:
        planned_moves: Dict[int, List[Tuple[int, int]]] = {}
        if existing_planned_moves:
            planned_moves.update(existing_planned_moves)

        walls = self._extract_walls(map_state)
        planning_time_horizon = max(
            self.min_time_horizon,
            self.width * self.height * self.grid_horizon_multiplier,
        )

        for agent_id in planned_agent_ids:
            agent = agents[agent_id]
            has_negotiated_path = (
                hasattr(agent, '_has_negotiated_path')
                and getattr(agent, '_has_negotiated_path', False)
                and agent.planned_path
                and len(agent.planned_path) > 1
            )

            if has_negotiated_path:
                path = agent.planned_path.copy()
            else:
                path = self.pathfinder.find_path_with_time_constraints(
                    start=agent.position,
                    goal=agent.target_position,
                    walls=walls,
                    reserved_positions_by_turn=reserved_positions_by_turn,
                    reserved_edges_by_turn=reserved_edges_by_turn,
                    max_time_steps=planning_time_horizon,
                )

                if not path:
                    other_positions = {
                        oid: oagent.position
                        for oid, oagent in agents.items()
                        if oid != agent_id
                    }
                    path = self.pathfinder.find_path_with_obstacles(
                        start=agent.position,
                        goal=agent.target_position,
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
            for turn_idx, pos in enumerate(path):
                reserved_positions_by_turn.setdefault(turn_idx, set()).add(pos)
                if turn_idx > 0:
                    prev = path[turn_idx - 1]
                    reserved_edges_by_turn.setdefault(turn_idx - 1, set()).add((prev, pos))

        return planned_moves

    def plan_all(
        self,
        agents: Dict[int, object],
        map_state: Dict,
        agent_ids: Optional[Set[int]] = None,
    ) -> Dict[int, List[Tuple[int, int]]]:
        active_ids = self._ordered_active_ids(agents, agent_ids)
        if not active_ids:
            return {}
        return self._plan_with_fixed_reservations(
            agents,
            map_state,
            active_ids,
            reserved_positions_by_turn={},
            reserved_edges_by_turn={},
        )

    def replan_subset(
        self,
        agents: Dict[int, object],
        map_state: Dict,
        agent_ids: Set[int],
    ) -> Dict[int, List[Tuple[int, int]]]:
        if not agent_ids:
            return {}

        subset_ids = self._ordered_active_ids(agents, set(agent_ids))
        if not subset_ids:
            return {}

        reserved_positions_by_turn: Dict[int, Set[Tuple[int, int]]] = {}
        reserved_edges_by_turn: Dict[int, Set[Tuple[Tuple[int, int], Tuple[int, int]]]] = {}

        # Seed reservation tables with existing paths from non-replanned active agents.
        reservation_horizon = max(
            self.min_time_horizon,
            self.width * self.height * self.grid_horizon_multiplier,
        )
        for aid, agent in agents.items():
            if aid in subset_ids or agent.is_waiting or not agent.target_position:
                continue
            existing_path = agent.planned_path if agent.planned_path else [agent.position]
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
        )
        return {aid: replanned[aid] for aid in subset_ids if aid in replanned}


class LNS2Planner(MultiAgentPlanner):
    """Python port structure based on MAPF-LNS2 `LNS` run loop in `src/LNS.cpp`."""
    # Sentinel used when a solution is missing; kept large so any valid solution wins.
    _INFINITE_COST = 10**9
    # Penalize unresolved conflicts heavily relative to per-step path cost.
    _CONFLICT_PENALTY_WEIGHT = 100
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
    ):
        self.base_planner = base_planner
        self.width = width
        self.height = height
        self.iterations = max(1, iterations)
        self.destroy_ratio = min(max(destroy_ratio, 0.1), 0.8)
        self.conflict_detector = ConflictDetector(width, height)
        self.random = random.Random(seed)
        self.destroy_strategy = (destroy_strategy or "randomwalk").strip().lower()

        # Adaptive-LNS controls (mirrors MAPF-LNS2 Adaptive destroy mode)
        self.alns = self.destroy_strategy == "adaptive"
        self.destroy_weights = [1.0, 1.0, 1.0]
        self.decay_factor = 0.01
        self.reaction_factor = 0.01
        self.selected_neighbor = 0

    def get_backend_name(self) -> str:
        return "LNS2"

    def supports_dynamic_repair(self) -> bool:
        return True

    def _path_cost(self, path: List[Tuple[int, int]]) -> int:
        return max(0, len(path) - 1)

    def _solution_cost(self, paths: Dict[int, List[Tuple[int, int]]], current_turn: int = 0) -> int:
        if not paths:
            return self._INFINITE_COST
        base_cost = sum(self._path_cost(path) for path in paths.values())
        conflict_info = self.conflict_detector.detect_path_conflicts(paths, current_turn)
        conflict_penalty = len(conflict_info.get('conflict_points', [])) * self._CONFLICT_PENALTY_WEIGHT
        return base_cost + conflict_penalty

    def _agent_delay(
        self,
        agent_id: int,
        solution: Dict[int, List[Tuple[int, int]]],
        agents: Dict[int, object],
    ) -> int:
        path = solution.get(agent_id, [])
        agent = agents.get(agent_id)
        if not path or not agent or not agent.target_position:
            return 0
        start = path[0]
        target = agent.target_position
        heuristic = abs(start[0] - target[0]) + abs(start[1] - target[1])
        return max(0, self._path_cost(path) - heuristic)

    def _find_most_delayed_agent(
        self,
        solution: Dict[int, List[Tuple[int, int]]],
        agents: Dict[int, object],
    ) -> Optional[int]:
        best_agent: Optional[int] = None
        best_delay = -1
        for aid in solution.keys():
            delay = self._agent_delay(aid, solution, agents)
            if delay > best_delay:
                best_delay = delay
                best_agent = aid
        if best_delay <= 0:
            return None
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
        agents: Dict[int, object],
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
        agents: Dict[int, object],
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

    def plan_all(
        self,
        agents: Dict[int, object],
        map_state: Dict,
        agent_ids: Optional[Set[int]] = None,
    ) -> Dict[int, List[Tuple[int, int]]]:
        # MAPF-LNS2 flow: getInitialSolution() first (see LNS::getInitialSolution).
        initial_solution = self.base_planner.plan_all(agents, map_state, agent_ids)
        if not initial_solution:
            return {}

        best_solution = {aid: path.copy() for aid, path in initial_solution.items()}
        best_cost = self._solution_cost(best_solution)

        # MAPF-LNS2 flow: iterative destroy/repair/evaluate loop (see LNS::run).
        for _ in range(self.iterations):
            subset = self._choose_destroy_subset(best_solution, agents)
            if not subset:
                continue

            old_subset_cost = sum(
                self._path_cost(best_solution.get(aid, []))
                for aid in subset
            )
            repaired_subset = self.base_planner.replan_subset(agents, map_state, subset)
            if not repaired_subset:
                continue

            new_subset_cost = sum(self._path_cost(path) for path in repaired_subset.values())
            candidate_solution = self._merge_solution(best_solution, repaired_subset)
            candidate_cost = self._solution_cost(candidate_solution)
            self._update_adaptive_weights(old_subset_cost, new_subset_cost, len(subset))

            # MAPF-LNS2 acceptance follows local replan improvement and global evaluation.
            if new_subset_cost <= old_subset_cost and candidate_cost <= best_cost:
                best_solution = candidate_solution
                best_cost = candidate_cost

        return best_solution

    def replan_subset(
        self,
        agents: Dict[int, object],
        map_state: Dict,
        agent_ids: Set[int],
    ) -> Dict[int, List[Tuple[int, int]]]:
        if not agent_ids:
            return {}
        return self.base_planner.replan_subset(agents, map_state, set(agent_ids))


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
    )
    if normalized == "lns2":
        iterations = int(os.getenv('LNS2_MAX_ITERATIONS', '8'))
        destroy_ratio = float(os.getenv('LNS2_DESTROY_RATIO', '0.35'))
        seed = int(os.getenv('LNS2_RANDOM_SEED', '0'))
        destroy_strategy = os.getenv('LNS2_DESTROY_STRATEGY', 'randomwalk')
        return LNS2Planner(
            base_planner=base,
            width=width,
            height=height,
            iterations=iterations,
            destroy_ratio=destroy_ratio,
            seed=seed,
            destroy_strategy=destroy_strategy,
        )
    return base
