"""
Conflict Detection and Basic Pathfinding
"""

import heapq
from typing import List, Tuple, Set, Dict, Optional
import numpy as np

class ConflictDetector:
    def __init__(self, map_width: int = 8, map_height: int = 6):
        self.map_width = map_width
        self.map_height = map_height
    
    def detect_path_conflicts(self, agents_paths: Dict[int, List[Tuple[int, int]]], current_turn: int = 0) -> Dict:
        """
        Detect conflicts in planned paths between agents
        
        Args:
            agents_paths: {agent_id: [(x, y), (x, y), ...]}
            current_turn: Current simulation turn
        
        Returns:
            conflict_info: {
                'has_conflicts': bool,
                'conflict_points': [(x, y), ...],
                'conflicting_agents': [agent_id, ...],
                'conflict_turns': [turn, ...]
            }
        """
        
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
    
    def _detect_swap_conflicts(self, agents_paths: Dict, turn: int, conflicts: Dict, current_turn: int):
        """Detect when two agents swap positions (crossing paths)"""
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

class SimplePathfinder:
    def __init__(self, map_width: int = 8, map_height: int = 6):
        self.map_width = map_width
        self.map_height = map_height
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], obstacles: Optional[Set[Tuple[int, int]]] = None) -> List[Tuple[int, int]]:
        """
        Find path using A* algorithm
        
        Args:
            start: Starting position (x, y)
            goal: Goal position (x, y)
            obstacles: Set of obstacle positions
        
        Returns:
            List of positions from start to goal
        """
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
    
    def find_path_avoiding_agents(self, start: Tuple[int, int], goal: Tuple[int, int], 
                                  agent_positions: Dict[int, Tuple[int, int]], 
                                  exclude_agent: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        Find path while avoiding other agents
        
        Args:
            start: Starting position
            goal: Goal position  
            agent_positions: {agent_id: (x, y)}
            exclude_agent: Agent ID to exclude from obstacles (usually the moving agent)
        
        Returns:
            Path avoiding other agents
        """
        obstacles = set()
        
        # Add other agents as obstacles
        for agent_id, pos in agent_positions.items():
            if exclude_agent is None or agent_id != exclude_agent:
                obstacles.add(pos)
        
        return self.find_path(start, goal, obstacles)
    
    def find_path_with_obstacles(self, start: Tuple[int, int], goal: Tuple[int, int],
                                walls: Set[Tuple[int, int]], 
                                agent_positions: Dict[int, Tuple[int, int]], 
                                exclude_agent: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        Find path while avoiding walls and other agents
        
        Args:
            start: Starting position
            goal: Goal position
            walls: Set of wall positions to avoid
            agent_positions: {agent_id: (x, y)}
            exclude_agent: Agent ID to exclude from obstacles
            
        Returns:
            Path avoiding walls and other agents
        """
        obstacles = set(walls)  # Start with walls
        
        # Add other agents as obstacles
        for agent_id, pos in agent_positions.items():
            if exclude_agent is None or agent_id != exclude_agent:
                obstacles.add(pos)
        
        return self.find_path(start, goal, obstacles)
    
    def get_path_cost(self, path: List[Tuple[int, int]]) -> int:
        """Calculate the cost (length) of a path"""
        return len(path) - 1 if len(path) > 1 else 0
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within map bounds"""
        x, y = pos
        return 0 <= x < self.map_width and 0 <= y < self.map_height
