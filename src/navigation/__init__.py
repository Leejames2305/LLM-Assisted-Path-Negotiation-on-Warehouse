"""
Conflict Detection and Basic Pathfinding
"""

import heapq
from typing import List, Tuple, Set, Dict, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .path_table import PathTable

class ConflictDetector:
    def __init__(self, map_width: int = 8, map_height: int = 6):
        self.map_width = map_width
        self.map_height = map_height
    
    # Detect conflicts in planned paths between agents
    def detect_path_conflicts(self, agents_paths: Dict[int, List[Tuple[int, int]]], current_turn: int = 0) -> Dict:
        conflicts = {
            'has_conflicts': False,
            'conflict_points': [],
            'conflicting_agents': [],
            'conflict_turns': []
        }
        
        # Track positions for each turn
        max_path_length = max(len(path) for path in agents_paths.values()) if agents_paths else 0
        
        for turn in range(max_path_length):
            turn_positions = {}
            
            # Get position of each agent at this turn
            for agent_id, path in agents_paths.items():
                if turn < len(path):
                    pos = path[turn]
                else:
                    # Agent has reached destination, stays there
                    pos = path[-1] if path else None
                
                if pos:
                    if pos in turn_positions:
                        # Conflict detected!
                        conflicts['has_conflicts'] = True
                        conflicts['conflict_points'].append(pos)
                        conflicts['conflicting_agents'].extend([agent_id, turn_positions[pos]])
                        conflicts['conflict_turns'].append(current_turn + turn)
                    else:
                        turn_positions[pos] = agent_id
            
            # Also check for "swapping" conflicts (agents crossing paths)
            self._detect_swap_conflicts(agents_paths, turn, conflicts, current_turn)
        
        # Remove duplicates
        conflicts['conflict_points'] = list(set(conflicts['conflict_points']))
        conflicts['conflicting_agents'] = list(set(conflicts['conflicting_agents']))
        conflicts['conflict_turns'] = list(set(conflicts['conflict_turns']))
        
        return conflicts
    
    # Detect when two agents swap positions
    def _detect_swap_conflicts(self, agents_paths: Dict, turn: int, conflicts: Dict, current_turn: int):
        agent_ids = list(agents_paths.keys())

        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                agent1_id, agent2_id = agent_ids[i], agent_ids[j]
                path1, path2 = agents_paths[agent1_id], agents_paths[agent2_id]

                if turn < len(path1) - 1 and turn < len(path2) - 1:
                    # Check if agents are swapping positions
                    agent1_current = path1[turn]
                    agent1_next = path1[turn + 1]
                    agent2_current = path2[turn]
                    agent2_next = path2[turn + 1]

                    if agent1_current == agent2_next and agent2_current == agent1_next:
                        # Swap conflict detected
                        conflicts['has_conflicts'] = True
                        conflicts['conflict_points'].extend([agent1_current, agent2_current])
                        conflicts['conflicting_agents'].extend([agent1_id, agent2_id])
                        conflicts['conflict_turns'].append(current_turn + turn)

    # Group conflicts into independent conflict groups based on agent connectivity
    def group_conflicts(self, agents_paths: Dict[int, List[Tuple[int, int]]], current_turn: int = 0) -> List[Dict]:
        """
        Group conflicts into independent groups based on agent connectivity.
        Each group represents agents whose paths interfere with each other.
        Independent groups can be negotiated in parallel.
        """
        conflicts = self.detect_path_conflicts(agents_paths, current_turn)

        if not conflicts['has_conflicts']:
            return []

        conflicting_agents = set(conflicts['conflicting_agents'])

        # Build adjacency graph: which agents conflict with which
        agent_graph = {agent_id: set() for agent_id in conflicting_agents}

        # Track all direct conflicts between agent pairs
        max_path_length = max(len(path) for path in agents_paths.values()) if agents_paths else 0

        for turn in range(max_path_length):
            turn_positions = {}

            # Detect position conflicts
            for agent_id, path in agents_paths.items():
                if agent_id not in conflicting_agents:
                    continue

                if turn < len(path):
                    pos = path[turn]
                else:
                    pos = path[-1] if path else None

                if pos and pos in turn_positions:
                    other_agent = turn_positions[pos]
                    agent_graph[agent_id].add(other_agent)
                    agent_graph[other_agent].add(agent_id)
                elif pos:
                    turn_positions[pos] = agent_id

            # Detect swap conflicts
            agent_ids = list(conflicting_agents)
            for i in range(len(agent_ids)):
                for j in range(i + 1, len(agent_ids)):
                    agent1_id, agent2_id = agent_ids[i], agent_ids[j]
                    path1, path2 = agents_paths[agent1_id], agents_paths[agent2_id]

                    if turn < len(path1) - 1 and turn < len(path2) - 1:
                        agent1_current = path1[turn]
                        agent1_next = path1[turn + 1]
                        agent2_current = path2[turn]
                        agent2_next = path2[turn + 1]

                        if agent1_current == agent2_next and agent2_current == agent1_next:
                            agent_graph[agent1_id].add(agent2_id)
                            agent_graph[agent2_id].add(agent1_id)

        # Find connected components using DFS
        visited = set()
        conflict_groups = []

        def dfs(agent_id: int, group: Set[int]):
            """Depth-first search to find all connected agents in conflict group"""
            if agent_id in visited:
                return
            visited.add(agent_id)
            group.add(agent_id)

            for neighbor in agent_graph[agent_id]:
                dfs(neighbor, group)

        # Build conflict groups
        for agent_id in conflicting_agents:
            if agent_id not in visited:
                group = set()
                dfs(agent_id, group)

                # Get conflict points for this group
                group_conflict_points = []
                for turn in range(max_path_length):
                    turn_positions = {}
                    for aid in group:
                        path = agents_paths[aid]
                        if turn < len(path):
                            pos = path[turn]
                        else:
                            pos = path[-1] if path else None

                        if pos:
                            if pos in turn_positions:
                                group_conflict_points.append(pos)
                            else:
                                turn_positions[pos] = aid

                # Remove duplicates
                group_conflict_points = list(set(group_conflict_points))

                conflict_groups.append({
                    'has_conflicts': True,
                    'conflicting_agents': list(group),
                    'conflict_points': group_conflict_points,
                    'conflict_turns': conflicts['conflict_turns']  # Shared for simplicity
                })

        return conflict_groups

class SimplePathfinder:
    # Default planning horizon parameters for time-aware reservation planning.
    # Keep this aligned with GameEngine planning-horizon defaults.
    _MIN_TIME_HORIZON = 16
    _GRID_HORIZON_MULTIPLIER = 2

    def __init__(self, map_width: int = 8, map_height: int = 6):
        self.map_width = map_width
        self.map_height = map_height
    
    # Basic A* pathfinding implementation
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], obstacles: Optional[Set[Tuple[int, int]]] = None) -> List[Tuple[int, int]]:
        if obstacles is None:
            obstacles = set()
        
        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance
        
        def get_neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
            x, y = pos
            neighbors = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # 4-directional movement
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.map_width and 0 <= ny < self.map_height and 
                    (nx, ny) not in obstacles):
                    neighbors.append((nx, ny))
            return neighbors
        
        # A* algorithm
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # Reverse to get start-to-goal order
            
            for neighbor in get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # No path found
    
    # Find path while avoiding other agents
    def find_path_avoiding_agents(self, start: Tuple[int, int], goal: Tuple[int, int], 
                                  agent_positions: Dict[int, Tuple[int, int]], 
                                  exclude_agent: Optional[int] = None) -> List[Tuple[int, int]]:
        
        obstacles = set()
        
        # Add other agents as obstacles
        for agent_id, pos in agent_positions.items():
            if exclude_agent is None or agent_id != exclude_agent:
                obstacles.add(pos)
        
        return self.find_path(start, goal, obstacles)
    
    # Find path while avoiding walls and other agents
    def find_path_with_obstacles(self, start: Tuple[int, int], goal: Tuple[int, int],
                                walls: Set[Tuple[int, int]], 
                                agent_positions: Dict[int, Tuple[int, int]], 
                                exclude_agent: Optional[int] = None) -> List[Tuple[int, int]]:

        obstacles = set(walls)  # Start with walls
        
        # Add other agents as obstacles
        for agent_id, pos in agent_positions.items():
            if exclude_agent is None or agent_id != exclude_agent:
                obstacles.add(pos)
        
        return self.find_path(start, goal, obstacles)

    # Find a path while avoiding time-specific reservations from previously planned agents
    def find_path_with_time_constraints(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        walls: Set[Tuple[int, int]],
        reserved_positions_by_turn: Optional[Dict[int, Set[Tuple[int, int]]]] = None,
        reserved_edges_by_turn: Optional[Dict[int, Set[Tuple[Tuple[int, int], Tuple[int, int]]]]] = None,
        max_time_steps: Optional[int] = None,
        path_table: Optional['PathTable'] = None,
    ) -> List[Tuple[int, int]]:
        if start == goal:
            return [start]

        reserved_positions_by_turn = reserved_positions_by_turn or {}
        reserved_edges_by_turn = reserved_edges_by_turn or {}
        if max_time_steps is None:
            max_time_steps = max(
                self._MIN_TIME_HORIZON,
                self.map_width * self.map_height * self._GRID_HORIZON_MULTIPLIER
            )

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def get_neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
            x, y = pos
            # Include current position so an agent can wait and delay movement
            # until future turns when reserved cells become available.
            # max_time_steps bounds how long the planner can keep waiting.
            neighbors = [pos]
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                    nxt = (nx, ny)
                    if nxt not in walls:
                        neighbors.append(nxt)
            return neighbors

        # (f_score, g_score, position, time_step)
        open_set = [(heuristic(start, goal), 0, start, 0)]
        came_from: Dict[Tuple[Tuple[int, int], int], Tuple[Tuple[int, int], int]] = {}
        best_cost: Dict[Tuple[Tuple[int, int], int], int] = {(start, 0): 0}

        while open_set:
            _, current_g, current_pos, current_t = heapq.heappop(open_set)
            current_state = (current_pos, current_t)

            if current_g > best_cost.get(current_state, float('inf')):
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
            reserved_next_positions = reserved_positions_by_turn.get(next_t, set())
            reserved_edges = reserved_edges_by_turn.get(current_t, set())

            for neighbor in get_neighbors(current_pos):
                if path_table is not None:
                    if path_table.constrained(current_pos, neighbor, next_t):
                        continue
                else:
                # Vertex conflict: cannot occupy a reserved position at next turn
                    if neighbor in reserved_next_positions:
                        continue

                # Edge swap conflict: if another agent moves from current_pos to
                # neighbor at this turn, this agent cannot simultaneously swap
                # by moving from neighbor to current_pos.
                    if (neighbor, current_pos) in reserved_edges:
                        continue

                next_state = (neighbor, next_t)
                tentative_g = current_g + 1
                if tentative_g < best_cost.get(next_state, float('inf')):
                    best_cost[next_state] = tentative_g
                    came_from[next_state] = current_state
                    tentative_f = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (tentative_f, tentative_g, neighbor, next_t))

        return []
    
    # Calculate the cost (length) of a path
    def get_path_cost(self, path: List[Tuple[int, int]]) -> int:
        return len(path) - 1 if len(path) > 1 else 0
    
    # Check if position is within map bounds
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.map_width and 0 <= y < self.map_height
