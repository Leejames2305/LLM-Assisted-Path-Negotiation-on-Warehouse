"""
A* Pathfinder for POGEMA grid using (row, col) coordinates.
Ported from src/navigation/__init__.py with coordinate system updated.
"""

import heapq
from typing import List, Tuple, Set, Dict, Optional

# POGEMA action space constants
ACTION_IDLE = 0
ACTION_UP = 1    # row - 1
ACTION_DOWN = 2  # row + 1
ACTION_LEFT = 3  # col - 1
ACTION_RIGHT = 4  # col + 1

# Delta (d_row, d_col) for each action
ACTION_DELTAS = {
    ACTION_IDLE: (0, 0),
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}

# Reverse lookup: delta -> action
DELTA_TO_ACTION = {v: k for k, v in ACTION_DELTAS.items()}


def path_to_pogema_actions(path: List[Tuple[int, int]]) -> List[int]:
    """Convert a sequence of (row, col) positions into POGEMA action integers."""
    actions = []
    for i in range(len(path) - 1):
        r0, c0 = path[i]
        r1, c1 = path[i + 1]
        delta = (r1 - r0, c1 - c0)
        action = DELTA_TO_ACTION.get(delta, ACTION_IDLE)
        actions.append(action)
    return actions


def pogema_action_to_delta(action: int) -> Tuple[int, int]:
    """Return the (d_row, d_col) for a given POGEMA action integer."""
    return ACTION_DELTAS.get(action, (0, 0))


class SimplePathfinder:
    """A* pathfinder operating in POGEMA (row, col) coordinate space."""

    def __init__(self, num_rows: int = 8, num_cols: int = 8):
        self.num_rows = num_rows
        self.num_cols = num_cols

    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacles: Optional[Set[Tuple[int, int]]] = None,
    ) -> List[Tuple[int, int]]:
        """A* search from start to goal. Returns list of (row, col) positions."""
        if obstacles is None:
            obstacles = set()

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def get_neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
            r, c = pos
            result = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.num_rows
                    and 0 <= nc < self.num_cols
                    and (nr, nc) not in obstacles
                ):
                    result.append((nr, nc))
            return result

        open_set = [(0, start)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        f_score: Dict[Tuple[int, int], int] = {start: heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for neighbor in get_neighbors(current):
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def find_path_avoiding_agents(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        agent_positions: Dict[int, Tuple[int, int]],
        exclude_agent: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """Find path while treating other agents as temporary obstacles."""
        obstacles = {
            pos
            for agent_id, pos in agent_positions.items()
            if exclude_agent is None or agent_id != exclude_agent
        }
        return self.find_path(start, goal, obstacles)

    def find_path_with_obstacles(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        walls: Set[Tuple[int, int]],
        agent_positions: Dict[int, Tuple[int, int]],
        exclude_agent: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """Find path avoiding walls and other agents."""
        obstacles = set(walls)
        for agent_id, pos in agent_positions.items():
            if exclude_agent is None or agent_id != exclude_agent:
                obstacles.add(pos)
        return self.find_path(start, goal, obstacles)

    def get_path_cost(self, path: List[Tuple[int, int]]) -> int:
        return len(path) - 1 if len(path) > 1 else 0

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.num_rows and 0 <= c < self.num_cols
