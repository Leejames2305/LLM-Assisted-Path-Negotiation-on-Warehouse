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
    AGENT_WITH_BOX = 'A*'
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
    
    def generate_map(self, num_agents: int = 2, wall_density: float = 0.1):
        """Generate a complete warehouse map"""
        # Reset grid
        self.grid = np.full((self.height, self.width), CellType.EMPTY.value, dtype=str)
        
        # Place elements
        self.place_walls(wall_density)
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
