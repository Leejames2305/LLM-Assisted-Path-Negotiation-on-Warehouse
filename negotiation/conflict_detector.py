"""
Conflict detection for POGEMA multi-agent pathfinding.
Adapted from src/navigation/__init__.py to use (row, col) coordinates.
"""

from typing import Dict, List, Tuple


class ConflictDetector:
    """Detects vertex and edge (swap) conflicts in multi-agent planned paths."""

    def __init__(self, num_rows: int = 8, num_cols: int = 8):
        self.num_rows = num_rows
        self.num_cols = num_cols

    def detect_path_conflicts(
        self,
        agents_paths: Dict[int, List[Tuple[int, int]]],
        current_step: int = 0,
    ) -> Dict:
        """Detect vertex and swap conflicts across all agents' planned paths."""
        conflicts = {
            'has_conflicts': False,
            'conflict_points': [],
            'conflicting_agents': [],
            'conflict_turns': [],
        }

        if not agents_paths:
            return conflicts

        max_path_length = max(len(p) for p in agents_paths.values())

        for turn in range(max_path_length):
            turn_positions: Dict[Tuple[int, int], int] = {}

            for agent_id, path in agents_paths.items():
                pos = path[turn] if turn < len(path) else (path[-1] if path else None)
                if pos is None:
                    continue
                pos = tuple(pos)  # ensure hashable

                if pos in turn_positions:
                    conflicts['has_conflicts'] = True
                    conflicts['conflict_points'].append(pos)
                    conflicts['conflicting_agents'].extend([agent_id, turn_positions[pos]])
                    conflicts['conflict_turns'].append(current_step + turn)
                else:
                    turn_positions[pos] = agent_id

            self._detect_swap_conflicts(agents_paths, turn, conflicts, current_step)

        # Deduplicate
        conflicts['conflict_points'] = list(set(map(tuple, conflicts['conflict_points'])))
        conflicts['conflicting_agents'] = list(set(conflicts['conflicting_agents']))
        conflicts['conflict_turns'] = list(set(conflicts['conflict_turns']))

        return conflicts

    def _detect_swap_conflicts(
        self,
        agents_paths: Dict,
        turn: int,
        conflicts: Dict,
        current_step: int,
    ):
        """Detect when two agents swap positions between consecutive steps."""
        agent_ids = list(agents_paths.keys())

        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                a1, a2 = agent_ids[i], agent_ids[j]
                p1, p2 = agents_paths[a1], agents_paths[a2]

                if turn < len(p1) - 1 and turn < len(p2) - 1:
                    a1_cur, a1_nxt = tuple(p1[turn]), tuple(p1[turn + 1])
                    a2_cur, a2_nxt = tuple(p2[turn]), tuple(p2[turn + 1])

                    if a1_cur == a2_nxt and a2_cur == a1_nxt:
                        conflicts['has_conflicts'] = True
                        conflicts['conflict_points'].extend([a1_cur, a2_cur])
                        conflicts['conflicting_agents'].extend([a1, a2])
                        conflicts['conflict_turns'].append(current_step + turn)
