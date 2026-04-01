"""Time-space path table for fast conflict checks and reservations."""

from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

Location = Tuple[int, int]
EdgeKey = Tuple[Location, Location, int]
VertexKey = Tuple[Location, int]


class PathTable:
    """Stores path occupancy by location and timestep for O(1)-style lookups."""

    def __init__(self):
        self._vertex_agents: DefaultDict[VertexKey, Set[int]] = defaultdict(set)
        self._edge_agents: DefaultDict[EdgeKey, Set[int]] = defaultdict(set)
        self._goal_arrival_time: DefaultDict[Location, Dict[int, int]] = defaultdict(dict)
        self._agent_vertex_keys: Dict[int, List[VertexKey]] = defaultdict(list)
        self._agent_edge_keys: Dict[int, List[EdgeKey]] = defaultdict(list)
        self._agent_goal_location: Dict[int, Location] = {}

    def clear(self) -> None:
        self._vertex_agents.clear()
        self._edge_agents.clear()
        self._goal_arrival_time.clear()
        self._agent_vertex_keys.clear()
        self._agent_edge_keys.clear()
        self._agent_goal_location.clear()

    def build_from_paths(self, paths: Dict[int, List[Location]], hold_until: Optional[int] = None) -> None:
        self.clear()
        if hold_until is None:
            hold_until = max((len(path) - 1 for path in paths.values() if path), default=0)
        for agent_id, path in paths.items():
            self.insert_path(agent_id, path, hold_until=hold_until)

    def insert_path(
        self,
        agent_id: int,
        path: List[Location],
        hold_until: Optional[int] = None,
    ) -> None:
        if not path:
            return

        self.delete_path(agent_id, path)

        for timestep, loc in enumerate(path):
            vertex_key = (loc, timestep)
            self._vertex_agents[vertex_key].add(agent_id)
            self._agent_vertex_keys[agent_id].append(vertex_key)
            if timestep > 0:
                prev = path[timestep - 1]
                edge_key = (prev, loc, timestep)
                self._edge_agents[edge_key].add(agent_id)
                self._agent_edge_keys[agent_id].append(edge_key)

        goal_loc = path[-1]
        goal_time = len(path) - 1
        self._goal_arrival_time[goal_loc][agent_id] = goal_time
        self._agent_goal_location[agent_id] = goal_loc

        if hold_until is not None and hold_until > goal_time:
            for timestep in range(goal_time + 1, hold_until + 1):
                vertex_key = (goal_loc, timestep)
                self._vertex_agents[vertex_key].add(agent_id)
                self._agent_vertex_keys[agent_id].append(vertex_key)
                edge_key = (goal_loc, goal_loc, timestep)
                self._edge_agents[edge_key].add(agent_id)
                self._agent_edge_keys[agent_id].append(edge_key)

    def delete_path(self, agent_id: int, path: Optional[List[Location]] = None) -> None:
        del path  # The table keeps direct insertion bookkeeping per agent.

        for vertex_key in self._agent_vertex_keys.pop(agent_id, []):
            agents = self._vertex_agents.get(vertex_key)
            if not agents:
                continue
            agents.discard(agent_id)
            if not agents:
                self._vertex_agents.pop(vertex_key, None)

        for edge_key in self._agent_edge_keys.pop(agent_id, []):
            agents = self._edge_agents.get(edge_key)
            if not agents:
                continue
            agents.discard(agent_id)
            if not agents:
                self._edge_agents.pop(edge_key, None)

        goal_loc = self._agent_goal_location.pop(agent_id, None)
        if goal_loc is not None:
            goals = self._goal_arrival_time.get(goal_loc)
            if goals:
                goals.pop(agent_id, None)
                if not goals:
                    self._goal_arrival_time.pop(goal_loc, None)

    def constrained(self, start_loc: Location, end_loc: Location, timestep: int, agent_id: Optional[int] = None) -> bool:
        if timestep < 0:
            return False

        vertex_agents = self._vertex_agents.get((end_loc, timestep), set())
        if any(aid != agent_id for aid in vertex_agents):
            return True

        reverse_edge_agents = self._edge_agents.get((end_loc, start_loc, timestep), set())
        if any(aid != agent_id for aid in reverse_edge_agents):
            return True

        goals = self._goal_arrival_time.get(end_loc, {})
        for other_agent, goal_time in goals.items():
            if other_agent != agent_id and goal_time <= timestep:
                return True

        return False

    def get_conflicting_agents(
        self,
        agent_id: int,
        start_loc: Location,
        end_loc: Location,
        timestep: int,
    ) -> Set[int]:
        conflicts: Set[int] = set()

        for other in self._vertex_agents.get((end_loc, timestep), set()):
            if other != agent_id:
                conflicts.add(other)

        for other in self._edge_agents.get((end_loc, start_loc, timestep), set()):
            if other != agent_id:
                conflicts.add(other)

        for other, goal_time in self._goal_arrival_time.get(end_loc, {}).items():
            if other != agent_id and goal_time <= timestep:
                conflicts.add(other)

        return conflicts

    def get_conflict_points(self) -> List[Tuple[Location, int]]:
        return [
            (loc, timestep)
            for (loc, timestep), agents in self._vertex_agents.items()
            if len(agents) > 1
        ]

    def get_conflicting_agents_set(self) -> Set[int]:
        conflicting: Set[int] = set()

        for agents in self._vertex_agents.values():
            if len(agents) > 1:
                conflicting.update(agents)

        seen_pairs: Set[Tuple[Location, Location, int]] = set()
        for (start_loc, end_loc, timestep), agents in self._edge_agents.items():
            if start_loc == end_loc:
                continue
            reverse_key = (end_loc, start_loc, timestep)
            if reverse_key in seen_pairs:
                continue
            reverse_agents = self._edge_agents.get(reverse_key)
            if reverse_agents:
                conflicting.update(agents)
                conflicting.update(reverse_agents)
                seen_pairs.add((start_loc, end_loc, timestep))
                seen_pairs.add(reverse_key)

        return conflicting

    def conflict_count(self) -> int:
        vertex_conflicts = sum(1 for agents in self._vertex_agents.values() if len(agents) > 1)

        edge_conflicts = 0
        seen_pairs: Set[Tuple[Location, Location, int]] = set()
        for (start_loc, end_loc, timestep), agents in self._edge_agents.items():
            if start_loc == end_loc:
                continue
            reverse_key = (end_loc, start_loc, timestep)
            if reverse_key in seen_pairs:
                continue
            reverse_agents = self._edge_agents.get(reverse_key)
            if reverse_agents and agents:
                edge_conflicts += 1
                seen_pairs.add((start_loc, end_loc, timestep))
                seen_pairs.add(reverse_key)

        return vertex_conflicts + edge_conflicts
