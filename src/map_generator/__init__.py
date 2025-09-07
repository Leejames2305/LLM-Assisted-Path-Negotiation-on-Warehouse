"""
Random Warehouse Map Generator
Generates warehouse maps with agents, boxes, and targets
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from enum import Enum

class CellType(Enum):
    EMPTY = '.'
    WALL = '#'
    AGENT = 'A'
    AGENT_WITH_BOX = '@'
    BOX = 'B'
    TARGET = 'T'

class WarehouseMap:
    def __init__(self, width: int = 8, height: int = 6):
        self.width = width
        self.height = height
        self.grid = np.full((height, width), CellType.EMPTY.value, dtype=str)
        self.agents = {}  # {agent_id: (x, y)}
        self.boxes = {}   # {box_id: (x, y)}
        self.targets = {} # {target_id: (x, y)}
        self.agent_goals = {}  # {agent_id: target_id}
        
    def is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is within bounds and not a wall"""
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[y, x] != CellType.WALL.value
    
    def is_empty_position(self, x: int, y: int) -> bool:
        """Check if position is empty (no agents, boxes, or targets)"""
        return self.is_valid_position(x, y) and self.grid[y, x] == CellType.EMPTY.value
    
    def place_walls(self, wall_density: float = 0.1):
        """Place random walls in the warehouse"""
        num_walls = int(self.width * self.height * wall_density)
        walls_placed = 0
        
        while walls_placed < num_walls:
            x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            if self.is_empty_position(x, y):
                self.grid[y, x] = CellType.WALL.value
                walls_placed += 1
    
    def create_tunnel_layout(self):
        """Create a tunnel-based layout with strategic bottlenecks and single-lane paths"""
        # Fill the entire map with walls first
        self.grid = np.full((self.height, self.width), CellType.WALL.value, dtype=str)
        
        # Create main horizontal tunnel through the middle
        middle_y = self.height // 2
        for x in range(self.width):
            self.grid[middle_y, x] = CellType.EMPTY.value
        
        # Create vertical connecting tunnels (bottlenecks)
        tunnel_positions = []
        
        if self.width >= 6:
            # Create 2-3 vertical tunnels at strategic positions
            tunnel_x_positions = [
                self.width // 4,
                self.width // 2,
                3 * self.width // 4
            ]
            
            for tunnel_x in tunnel_x_positions:
                if tunnel_x < self.width:
                    tunnel_positions.append(tunnel_x)
                    # Create vertical tunnel
                    for y in range(self.height):
                        self.grid[y, tunnel_x] = CellType.EMPTY.value
        else:
            # For smaller maps, create one central vertical tunnel
            tunnel_x = self.width // 2
            tunnel_positions.append(tunnel_x)
            for y in range(self.height):
                self.grid[y, tunnel_x] = CellType.EMPTY.value
        
        # Create small chambers at tunnel intersections
        for tunnel_x in tunnel_positions:
            # Top chamber
            if middle_y > 1:
                chamber_y = middle_y - 2
                if chamber_y >= 0:
                    for dx in [-1, 0, 1]:
                        if 0 <= tunnel_x + dx < self.width:
                            self.grid[chamber_y, tunnel_x + dx] = CellType.EMPTY.value
            
            # Bottom chamber  
            if middle_y < self.height - 2:
                chamber_y = middle_y + 2
                if chamber_y < self.height:
                    for dx in [-1, 0, 1]:
                        if 0 <= tunnel_x + dx < self.width:
                            self.grid[chamber_y, tunnel_x + dx] = CellType.EMPTY.value
        
        # Add some additional single-cell passages to create more complexity
        if self.height >= 4:
            # Top horizontal mini-tunnel
            top_y = 0
            start_x = max(1, tunnel_positions[0] - 1)
            end_x = min(self.width - 1, tunnel_positions[-1] + 1)
            for x in range(start_x, end_x + 1):
                self.grid[top_y, x] = CellType.EMPTY.value
            
            # Bottom horizontal mini-tunnel
            bottom_y = self.height - 1
            for x in range(start_x, end_x + 1):
                self.grid[bottom_y, x] = CellType.EMPTY.value

    def create_extreme_single_corridor(self):
        """Create an EXTREME single corridor layout that forces constant negotiation"""
        # Fill the entire map with walls first
        self.grid = np.full((self.height, self.width), CellType.WALL.value, dtype=str)
        
        # Strategy: Create internal maze with no edge paths
        # All navigation must go through narrow internal corridors
        
        corridor_cells = set()
        
        # Create main internal corridors (avoid all edges completely)
        # Main horizontal corridor in the center
        center_y = self.height // 2
        for x in range(2, self.width - 2):  # Stay away from edges
            corridor_cells.add((x, center_y))
        
        # Default positions for corridors
        left_x = max(1, self.width // 4)
        right_x = min(self.width - 2, 3 * self.width // 4)
        upper_y = max(1, center_y - 1)
        lower_y = min(self.height - 2, center_y + 1)
        
        # Create two vertical corridors that connect areas but avoid edges
        if self.width > 6 and self.height > 4:
            # Left internal corridor
            for y in range(1, self.height - 1):  # Avoid top/bottom edges
                corridor_cells.add((left_x, y))
            
            # Right internal corridor  
            for y in range(1, self.height - 1):  # Avoid top/bottom edges
                corridor_cells.add((right_x, y))
        
        # Create narrow connecting passages between the main corridors
        if self.height > 3:
            # Upper connecting corridor
            for x in range(left_x + 1, right_x):
                if x % 2 == 0:  # Every other cell to create bottlenecks
                    corridor_cells.add((x, upper_y))
            
            # Lower connecting corridor
            for x in range(left_x + 1, right_x):
                if x % 2 == 1:  # Offset pattern for bottlenecks
                    corridor_cells.add((x, lower_y))
        
        # Add critical single-cell bottlenecks at intersections
        critical_points = [
            (left_x, center_y),   # Left intersection
            (right_x, center_y),  # Right intersection
        ]
        
        # Add some internal branching paths (but still no edges)
        if self.width > 8 and self.height > 5:
            mid_x = self.width // 2
            # Short vertical branches from center
            for dy in [-1, 1]:
                branch_y = center_y + dy
                if 1 < branch_y < self.height - 1:
                    corridor_cells.add((mid_x, branch_y))
        
        # Create a few small dead-end chambers to add complexity (only if we have space)
        if self.width > 10 and self.height > 6:
            # Small chambers connected by single cells
            chamber_positions = [
                (left_x - 1, upper_y),
                (left_x - 1, upper_y + 1),
                (right_x + 1, lower_y),
                (right_x + 1, lower_y + 1),
            ]
            for x, y in chamber_positions:
                if 1 <= x < self.width - 1 and 1 <= y < self.height - 1:
                    corridor_cells.add((x, y))
        
        # Apply all corridors to the grid
        all_cells = corridor_cells.union(set(critical_points))
        for x, y in all_cells:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = CellType.EMPTY.value
    
    def create_narrow_bridge_layout(self):
        """Create multiple chambers connected by single-cell bridges"""
        # Fill with walls
        self.grid = np.full((self.height, self.width), CellType.WALL.value, dtype=str)
        
        # Create chambers (2x2 or 3x3 areas)
        chambers = []
        
        # Top-left chamber
        if self.width >= 4 and self.height >= 4:
            for y in range(2):
                for x in range(2):
                    self.grid[y, x] = CellType.EMPTY.value
            chambers.append((0, 0, 2, 2))  # x, y, width, height
        
        # Top-right chamber
        if self.width >= 4:
            start_x = self.width - 2
            for y in range(2):
                for x in range(start_x, self.width):
                    self.grid[y, x] = CellType.EMPTY.value
            chambers.append((start_x, 0, 2, 2))
        
        # Bottom-left chamber
        if self.height >= 4:
            start_y = self.height - 2
            for y in range(start_y, self.height):
                for x in range(2):
                    self.grid[y, x] = CellType.EMPTY.value
            chambers.append((0, start_y, 2, 2))
        
        # Bottom-right chamber
        if self.width >= 4 and self.height >= 4:
            start_x = self.width - 2
            start_y = self.height - 2
            for y in range(start_y, self.height):
                for x in range(start_x, self.width):
                    self.grid[y, x] = CellType.EMPTY.value
            chambers.append((start_x, start_y, 2, 2))
        
        # Create single-cell bridges between chambers
        # Horizontal bridge in the middle
        middle_y = self.height // 2
        for x in range(2, self.width - 2):
            self.grid[middle_y, x] = CellType.EMPTY.value
        
        # Vertical bridges
        middle_x = self.width // 2
        for y in range(2, self.height - 2):
            self.grid[y, middle_x] = CellType.EMPTY.value
        
        # Connect chambers to main corridors
        # Top chambers to horizontal bridge
        self.grid[2, 1] = CellType.EMPTY.value  # Top-left to bridge
        if self.width > 4:
            self.grid[2, self.width - 2] = CellType.EMPTY.value  # Top-right to bridge
        
        # Bottom chambers to horizontal bridge
        if self.height > 4:
            self.grid[self.height - 3, 1] = CellType.EMPTY.value  # Bottom-left to bridge
            if self.width > 4:
                self.grid[self.height - 3, self.width - 2] = CellType.EMPTY.value  # Bottom-right to bridge
    
    def place_agents(self, num_agents: int = 2):
        """Place agents randomly on the map"""
        self.agents = {}
        for agent_id in range(num_agents):
            while True:
                x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
                if self.is_empty_position(x, y):
                    self.grid[y, x] = CellType.AGENT.value
                    self.agents[agent_id] = (x, y)
                    break
    
    def place_boxes(self, num_boxes: Optional[int] = None):
        """Place boxes randomly on the map"""
        if num_boxes is None:
            num_boxes = len(self.agents)  # One box per agent by default
        
        self.boxes = {}
        for box_id in range(num_boxes):
            while True:
                x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
                if self.is_empty_position(x, y):
                    self.grid[y, x] = CellType.BOX.value
                    self.boxes[box_id] = (x, y)
                    break
    
    def place_targets(self, num_targets: Optional[int] = None):
        """Place targets randomly on the map"""
        if num_targets is None:
            num_targets = len(self.boxes)  # One target per box by default
        
        self.targets = {}
        for target_id in range(num_targets):
            while True:
                x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
                if self.is_empty_position(x, y):
                    self.grid[y, x] = CellType.TARGET.value
                    self.targets[target_id] = (x, y)
                    break
    
    def assign_goals(self):
        """Assign each agent a target to reach"""
        target_ids = list(self.targets.keys())
        random.shuffle(target_ids)
        
        for i, agent_id in enumerate(self.agents.keys()):
            if i < len(target_ids):
                self.agent_goals[agent_id] = target_ids[i]
    
    def generate_map(self, num_agents: int = 2, wall_density: float = 0.1, layout_type: str = "tunnel"):
        """
        Generate a complete warehouse map
        
        Args:
            num_agents: Number of agents to place
            wall_density: Density of walls (only used for random layout)
            layout_type: Layout types available:
                - "tunnel": Strategic bottlenecks and chambers
                - "extreme": Single serpentine corridor (MAXIMUM conflicts)
                - "bridge": Chambers connected by single-cell bridges
                - "random": Random wall placement
        """
        # Reset grid
        self.grid = np.full((self.height, self.width), CellType.EMPTY.value, dtype=str)
        
        # Place walls based on layout type
        if layout_type == "extreme":
            self.create_extreme_single_corridor()
        elif layout_type == "bridge":
            self.create_narrow_bridge_layout()
        elif layout_type == "tunnel":
            self.create_tunnel_layout()
        else:  # random
            self.place_walls(wall_density)
        
        # Place elements
        self.place_agents(num_agents)
        self.place_boxes(num_agents)  # Same number of boxes as agents
        self.place_targets(num_agents)  # Same number of targets as boxes
        self.assign_goals()
    
    def move_agent(self, agent_id: int, new_x: int, new_y: int) -> bool:
        """Move an agent to a new position"""
        if agent_id not in self.agents:
            return False
        
        old_x, old_y = self.agents[agent_id]
        
        # Check if new position is valid
        if not self.is_valid_position(new_x, new_y):
            return False
        
        # Check if position is occupied by another agent
        for other_id, (ox, oy) in self.agents.items():
            if other_id != agent_id and ox == new_x and oy == new_y:
                return False
        
        # Update agent position
        self.grid[old_y, old_x] = CellType.EMPTY.value
        self.agents[agent_id] = (new_x, new_y)
        
        # Check if agent is carrying a box
        agent_has_box = self.grid[old_y, old_x] == CellType.AGENT_WITH_BOX.value
        
        if agent_has_box:
            self.grid[new_y, new_x] = CellType.AGENT_WITH_BOX.value
        else:
            self.grid[new_y, new_x] = CellType.AGENT.value
        
        return True
    
    def pickup_box(self, agent_id: int, box_id: int) -> bool:
        """Agent picks up a box"""
        if agent_id not in self.agents or box_id not in self.boxes:
            return False
        
        agent_x, agent_y = self.agents[agent_id]
        box_x, box_y = self.boxes[box_id]
        
        # Check if agent is adjacent to the box
        if abs(agent_x - box_x) + abs(agent_y - box_y) == 1:
            # Remove box from its position
            self.grid[box_y, box_x] = CellType.EMPTY.value
            del self.boxes[box_id]
            
            # Update agent to show it's carrying a box
            self.grid[agent_y, agent_x] = CellType.AGENT_WITH_BOX.value
            return True
        
        return False
    
    def drop_box(self, agent_id: int, target_id: int) -> bool:
        """Agent drops a box at a target"""
        if agent_id not in self.agents or target_id not in self.targets:
            return False
        
        agent_x, agent_y = self.agents[agent_id]
        target_x, target_y = self.targets[target_id]
        
        # Check if agent is at the target and carrying a box
        if (agent_x == target_x and agent_y == target_y and 
            self.grid[agent_y, agent_x] == CellType.AGENT_WITH_BOX.value):
            
            # Update agent to normal state
            self.grid[agent_y, agent_x] = CellType.AGENT.value
            
            # Remove the target (mission completed)
            del self.targets[target_id]
            
            return True
        
        return False
    
    def display(self) -> str:
        """Return a string representation of the map"""
        display_grid = self.grid.copy()
        
        # Add spacing for better visibility
        rows = []
        for row in display_grid:
            rows.append(' '.join(row))
        
        return '\n'.join(rows)
    
    def get_state_dict(self) -> Dict:
        """Get current state as dictionary for logging"""
        return {
            'agents': self.agents.copy(),
            'boxes': self.boxes.copy(),
            'targets': self.targets.copy(),
            'agent_goals': self.agent_goals.copy(),
            'grid': self.grid.tolist()
        }
