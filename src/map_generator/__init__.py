"""
Warehouse Map - Core data structure for warehouse simulation
Supports loading from layout files and managing map state
"""

import numpy as np
from typing import Dict, Optional

from .constants import CellType

__all__ = ['WarehouseMap', 'CellType']


class WarehouseMap:
    """Warehouse map representation with agents, boxes, targets, and walls"""

    def __init__(self, width: int = 8, height: int = 6):
        """
        Initialize a warehouse map

        Args:
            width: Grid width (will be created empty if no layout provided)
            height: Grid height (will be created empty if no layout provided)
        """
        self.width = width
        self.height = height
        self.grid = np.full((height, width), CellType.EMPTY.value, dtype=str)
        self.agents = {}  # {agent_id: (x, y)}
        self.boxes = {}  # {box_id: (x, y)}
        self.targets = {}  # {target_id: (x, y)}
        self.agent_goals = {}  # {agent_id: target_id}

    @classmethod
    def from_layout(cls, layout: Dict) -> 'WarehouseMap':
        """
        Create a warehouse map from a layout dictionary

        Args:
            layout: Layout dict with structure:
                {
                    "dimensions": {"width": int, "height": int},
                    "grid": [list of strings],
                    "agents": [{"id": int, "x": int, "y": int}, ...],
                    "boxes": [{"id": int, "x": int, "y": int}, ...],
                    "targets": [{"id": int, "x": int, "y": int}, ...],
                    "agent_goals": {str(agent_id): target_id}
                }

        Returns:
            WarehouseMap instance initialized from layout
        """
        width = layout['dimensions']['width']
        height = layout['dimensions']['height']

        # Create new map instance
        warehouse = cls(width, height)

        # Load grid from layout
        grid_data = layout['grid']
        for y, row in enumerate(grid_data):
            for x, cell in enumerate(row):
                warehouse.grid[y, x] = cell

        # Load agents
        for agent in layout.get('agents', []):
            agent_id = agent['id']
            x = agent['x']
            y = agent['y']
            warehouse.agents[agent_id] = (x, y)

        # Load boxes
        for box in layout.get('boxes', []):
            box_id = box['id']
            x = box['x']
            y = box['y']
            warehouse.boxes[box_id] = (x, y)

        # Load targets
        for target in layout.get('targets', []):
            target_id = target['id']
            x = target['x']
            y = target['y']
            warehouse.targets[target_id] = (x, y)

        # Load agent goals
        agent_goals = layout.get('agent_goals', {})
        for agent_id_str, target_id in agent_goals.items():
            agent_id = int(agent_id_str)  # Goals might be stored as strings in JSON
            warehouse.agent_goals[agent_id] = target_id

        return warehouse
        
    def is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is within bounds and not a wall"""
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[y, x] != CellType.WALL.value
    
    def is_empty_position(self, x: int, y: int) -> bool:
        """Check if position is empty (no agents, boxes, or targets)"""
        return self.is_valid_position(x, y) and self.grid[y, x] == CellType.EMPTY.value
    
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
