"""Small bounded CBS-style repair solver for subset replanning."""

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..agents import RobotAgent
from .path_table import PathTable

Location = Tuple[int, int]
Constraint = Tuple[str, Tuple]


@dataclass(order=True)
class _CBSNode:
    priority: int
    paths: Dict[int, List[Location]] = field(compare=False)
    constraints: Dict[int, List[Constraint]] = field(compare=False)


class MiniCBSRepair:
    """Bounded mini-CBS for local subset repairs."""

    def __init__(
        self,
        width: int,
        height: int,
        min_time_horizon: int,
        grid_horizon_multiplier: int,
        max_nodes: int = 128,
    ):
        self.width = width
        self.height = height
        self.min_time_horizon = min_time_horizon
        self.grid_horizon_multiplier = grid_horizon_multiplier
        self.max_nodes = max(8, max_nodes)

    def _planning_horizon(self) -> int:
        return max(
            self.min_time_horizon,
            self.width * self.height * self.grid_horizon_multiplier,
        )

    def _extract_walls(self, map_state: Dict) -> Set[Location]:
        walls: Set[Location] = set()
        grid = map_state.get("grid", [])
        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                if cell == "#":
                    walls.add((x, y))
        return walls

    def _pos_at(self, path: List[Location], timestep: int) -> Optional[Location]:
        if not path:
            return None
        if timestep < len(path):
            return path[timestep]
        return path[-1]

    def _first_conflict(self, paths: Dict[int, List[Location]]) -> Optional[Dict]:
        if not paths:
            return None

        agent_ids = list(paths.keys())
        max_len = max(len(path) for path in paths.values())

        for timestep in range(max_len):
            turn_positions: Dict[Location, int] = {}
            for aid in agent_ids:
                pos = self._pos_at(paths[aid], timestep)
                if pos is None:
                    continue
                other = turn_positions.get(pos)
                if other is not None and other != aid:
                    return {
                        "type": "vertex",
                        "a1": other,
                        "a2": aid,
                        "loc": pos,
                        "t": timestep,
                    }
                turn_positions[pos] = aid

            if timestep == 0:
                continue

            for i in range(len(agent_ids)):
                for j in range(i + 1, len(agent_ids)):
                    a1 = agent_ids[i]
                    a2 = agent_ids[j]
                    a1_prev = self._pos_at(paths[a1], timestep - 1)
                    a1_curr = self._pos_at(paths[a1], timestep)
                    a2_prev = self._pos_at(paths[a2], timestep - 1)
                    a2_curr = self._pos_at(paths[a2], timestep)
                    if None in (a1_prev, a1_curr, a2_prev, a2_curr):
                        continue
                    if a1_prev == a2_curr and a2_prev == a1_curr:
                        return {
                            "type": "edge",
                            "a1": a1,
                            "a2": a2,
                            "from1": a1_prev,
                            "to1": a1_curr,
                            "from2": a2_prev,
                            "to2": a2_curr,
                            "t": timestep,
                        }
        return None

    def _build_constraint_sets(self, constraints: List[Constraint]) -> Tuple[Set[Tuple[Location, int]], Set[Tuple[Location, Location, int]]]:
        vertex_constraints: Set[Tuple[Location, int]] = set()
        edge_constraints: Set[Tuple[Location, Location, int]] = set()
        for kind, payload in constraints:
            if kind == "vertex":
                loc, timestep = payload
                vertex_constraints.add((loc, timestep))
            elif kind == "edge":
                from_loc, to_loc, timestep = payload
                edge_constraints.add((from_loc, to_loc, timestep))
        return vertex_constraints, edge_constraints

    def _find_path(
        self,
        start: Location,
        goal: Location,
        walls: Set[Location],
        static_table: PathTable,
        constraints: List[Constraint],
        max_time_steps: int,
    ) -> List[Location]:
        if start == goal:
            return [start]

        def heuristic(a: Location, b: Location) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def neighbors(pos: Location) -> List[Location]:
            x, y = pos
            cand = [pos]
            for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                nx, ny = x + dx, y + dy
                nxt = (nx, ny)
                if 0 <= nx < self.width and 0 <= ny < self.height and nxt not in walls:
                    cand.append(nxt)
            return cand

        vertex_constraints, edge_constraints = self._build_constraint_sets(constraints)

        open_set = [(heuristic(start, goal), 0, start, 0)]
        came_from: Dict[Tuple[Location, int], Tuple[Location, int]] = {}
        best_cost: Dict[Tuple[Location, int], int] = {(start, 0): 0}

        while open_set:
            _, current_g, current_pos, current_t = heapq.heappop(open_set)
            current_state = (current_pos, current_t)

            if current_g > best_cost.get(current_state, 10**9):
                continue

            if current_pos == goal:
                path = [current_pos]
                trace_state = current_state
                while trace_state in came_from:
                    trace_state = came_from[trace_state]
                    path.append(trace_state[0])
                return path[::-1]

            if current_t >= max_time_steps:
                continue

            next_t = current_t + 1
            for nxt in neighbors(current_pos):
                if (nxt, next_t) in vertex_constraints:
                    continue
                if (current_pos, nxt, next_t) in edge_constraints:
                    continue
                if static_table.constrained(current_pos, nxt, next_t):
                    continue

                next_state = (nxt, next_t)
                tentative_g = current_g + 1
                if tentative_g < best_cost.get(next_state, 10**9):
                    best_cost[next_state] = tentative_g
                    came_from[next_state] = current_state
                    heapq.heappush(
                        open_set,
                        (tentative_g + heuristic(nxt, goal), tentative_g, nxt, next_t),
                    )

        return []

    def _seed_static_table(
        self,
        agents: Dict[int, RobotAgent],
        replanned_ids: Set[int],
        horizon: int,
    ) -> PathTable:
        table = PathTable()
        for aid, agent in agents.items():
            if aid in replanned_ids:
                continue

            # Pinned/finished agents stay in place and must remain hard constraints.
            if agent.is_waiting or not agent.target_position:
                existing_path = [agent.position]
            else:
                existing_path = agent.planned_path if agent.planned_path else [agent.position]

            table.insert_path(aid, existing_path, hold_until=horizon)
        return table

    def _initial_paths(
        self,
        agents: Dict[int, RobotAgent],
        subset_ids: List[int],
        walls: Set[Location],
        static_table: PathTable,
        constraints: Dict[int, List[Constraint]],
        max_time_steps: int,
    ) -> Dict[int, List[Location]]:
        paths: Dict[int, List[Location]] = {}
        for aid in subset_ids:
            agent = agents[aid]
            goal = agent.target_position
            if goal is None:
                continue
            path = self._find_path(
                start=agent.position,
                goal=goal,
                walls=walls,
                static_table=static_table,
                constraints=constraints.get(aid, []),
                max_time_steps=max_time_steps,
            )
            if not path:
                return {}
            paths[aid] = path
        return paths

    def _copy_constraints(self, constraints: Dict[int, List[Constraint]]) -> Dict[int, List[Constraint]]:
        return {aid: cons.copy() for aid, cons in constraints.items()}

    def _sum_cost(self, paths: Dict[int, List[Location]]) -> int:
        return sum(max(0, len(path) - 1) for path in paths.values())

    def replan_subset(
        self,
        agents: Dict[int, RobotAgent],
        map_state: Dict,
        agent_ids: Set[int],
    ) -> Dict[int, List[Location]]:
        subset_ids = [
            aid
            for aid in sorted(agent_ids)
            if aid in agents and (not agents[aid].is_waiting) and agents[aid].target_position
        ]
        if not subset_ids:
            return {}

        max_time_steps = self._planning_horizon()
        walls = self._extract_walls(map_state)
        static_table = self._seed_static_table(agents, set(subset_ids), max_time_steps)

        root_constraints: Dict[int, List[Constraint]] = {aid: [] for aid in subset_ids}
        root_paths = self._initial_paths(
            agents=agents,
            subset_ids=subset_ids,
            walls=walls,
            static_table=static_table,
            constraints=root_constraints,
            max_time_steps=max_time_steps,
        )
        if len(root_paths) != len(subset_ids):
            return {}

        open_nodes: List[_CBSNode] = []
        heapq.heappush(
            open_nodes,
            _CBSNode(
                priority=self._sum_cost(root_paths),
                paths=root_paths,
                constraints=root_constraints,
            ),
        )

        expanded = 0
        while open_nodes and expanded < self.max_nodes:
            expanded += 1
            node = heapq.heappop(open_nodes)
            conflict = self._first_conflict(node.paths)
            if conflict is None:
                return {aid: path.copy() for aid, path in node.paths.items()}

            if conflict["type"] == "vertex":
                branches = [
                    (conflict["a1"], ("vertex", (conflict["loc"], conflict["t"]))),
                    (conflict["a2"], ("vertex", (conflict["loc"], conflict["t"]))),
                ]
            else:
                branches = [
                    (
                        conflict["a1"],
                        ("edge", (conflict["from1"], conflict["to1"], conflict["t"])),
                    ),
                    (
                        conflict["a2"],
                        ("edge", (conflict["from2"], conflict["to2"], conflict["t"])),
                    ),
                ]

            for replanned_agent, new_constraint in branches:
                child_constraints = self._copy_constraints(node.constraints)
                child_constraints.setdefault(replanned_agent, []).append(new_constraint)

                agent = agents[replanned_agent]
                goal = agent.target_position
                if goal is None:
                    continue

                new_path = self._find_path(
                    start=agent.position,
                    goal=goal,
                    walls=walls,
                    static_table=static_table,
                    constraints=child_constraints[replanned_agent],
                    max_time_steps=max_time_steps,
                )
                if not new_path:
                    continue

                child_paths = {aid: path.copy() for aid, path in node.paths.items()}
                child_paths[replanned_agent] = new_path
                heapq.heappush(
                    open_nodes,
                    _CBSNode(
                        priority=self._sum_cost(child_paths),
                        paths=child_paths,
                        constraints=child_constraints,
                    ),
                )

        return {}
