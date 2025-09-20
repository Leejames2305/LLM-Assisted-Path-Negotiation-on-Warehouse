"""
Robot Agent class representing individual warehouse robots
"""

from typing import Tuple, List, Optional, Dict
from ..llm.agent_validator import AgentValidator
from ..navigation import SimplePathfinder

class RobotAgent:
    def __init__(self, agent_id: int, initial_position: Tuple[int, int], target_position: Optional[Tuple[int, int]] = None):
        self.agent_id = agent_id
        self.position = initial_position
        self.target_position = target_position
        self.carrying_box = False
        self.box_id = None
        self.planned_path = []
        self.current_action = None
        self.validator = AgentValidator()
        self.pathfinder = None  # Will be initialized with map size
        
        # Agent state
        self.is_waiting = False
        self.wait_turns_remaining = 0
        self.priority = 1
        
    def set_target(self, target_position: Tuple[int, int]):
        """Set the agent's target position"""
        self.target_position = target_position
        self.planned_path = []  # Reset path when target changes
    
    def plan_path(self, map_state: Dict) -> List[Tuple[int, int]]:
        """Plan a path to the target using pathfinding"""
        if not self.target_position:
            return []
        
        # Initialize pathfinder with map dimensions if not already done
        grid = map_state.get('grid', [])
        if grid and not self.pathfinder:
            map_height = len(grid)
            map_width = len(grid[0]) if grid else 0
            self.pathfinder = SimplePathfinder(map_width, map_height)
        
        if not self.pathfinder:
            return []  # No valid map state
        
        # Get other agent positions to avoid
        other_agents = {aid: pos for aid, pos in map_state.get('agents', {}).items() 
                       if aid != self.agent_id}
        
        # Get wall positions from the grid
        wall_positions = set()
        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                if cell == '#':  # Wall cell
                    wall_positions.add((x, y))
        
        path = self.pathfinder.find_path_with_obstacles(
            start=self.position,
            goal=self.target_position,
            walls=wall_positions,
            agent_positions=other_agents,
            exclude_agent=self.agent_id
        )
        
        self.planned_path = path
        return path
    
    def get_next_move(self) -> Optional[Tuple[int, int]]:
        """Get the next position to move to"""
        if not self.planned_path or len(self.planned_path) <= 1:
            return None
        
        # Find current position in path
        try:
            current_idx = self.planned_path.index(self.position)
            if current_idx + 1 < len(self.planned_path):
                return self.planned_path[current_idx + 1]
        except ValueError:
            # Current position not in path, return first position
            return self.planned_path[0]
        
        return None
    
    def execute_negotiated_action(self, action_data: Dict, map_state: Dict) -> bool:
        """
        Execute an action received from central negotiator
        
        Args:
            action_data: {'action': 'move'/'wait', 'path': [...], 'priority': int, 'wait_turns': int}
            map_state: Current map state
        
        Returns:
            bool: True if action was executed successfully
        """
        
        # Check if this agent was already validated by HMAS-2
        if getattr(self, '_hmas2_validated', False):
            print(f"✅ Agent {self.agent_id}: Executing HMAS-2 pre-validated action (skipping redundant validation)")
            return self._execute_action(action_data, map_state)
        
        # Otherwise, perform normal validation
        print(f"🔍 Agent {self.agent_id}: Performing game engine validation (no HMAS-2 flag)")
        validation_result = self.validator.validate_negotiated_action(
            self.agent_id, action_data, map_state
        )
        
        if not validation_result['valid']:
            print(f"Agent {self.agent_id}: Action validation failed - {validation_result['reason']}")
            
            # Try alternative if provided
            if validation_result.get('alternative'):
                print(f"Agent {self.agent_id}: Trying alternative action")
                return self._execute_action(validation_result['alternative'], map_state)
            else:
                # Default to waiting
                self.wait(1)
                return False
        
        # Execute the validated action
        return self._execute_action(action_data, map_state)
    
    def _execute_action(self, action_data: Dict, map_state: Dict) -> bool:
        """Internal method to execute an action"""
        action = action_data.get('action', 'wait')
        
        if action == 'wait':
            wait_turns = action_data.get('wait_turns', 1)
            self.wait(wait_turns)
            return True
        
        elif action == 'move':
            path = action_data.get('path', [])
            if path and len(path) > 1:
                next_pos = path[1]  # First position is current, second is next
                return self.move_to(next_pos, map_state)
            else:
                next_move = self.get_next_move()
                if next_move:
                    return self.move_to(next_move, map_state)
        
        return False
    
    def move_to(self, new_position: Tuple[int, int], map_state: Dict) -> bool:
        """
        Move agent to new position with safety checks
        
        Args:
            new_position: Target position (x, y)
            map_state: Current map state
        
        Returns:
            bool: True if move was successful
        """
        
        # Additional safety check using validator
        is_safe = self.validator.check_move_safety(
            self.agent_id, self.position, new_position, map_state
        )
        
        if is_safe:
            self.position = new_position
            
            # Update planned path
            if self.planned_path and new_position in self.planned_path:
                idx = self.planned_path.index(new_position)
                self.planned_path = self.planned_path[idx:]
            
            # Check if reached target
            if self.position == self.target_position:
                print(f"Agent {self.agent_id}: Reached target {self.target_position}")
                self.planned_path = []
            
            return True
        else:
            print(f"Agent {self.agent_id}: Move to {new_position} is not safe")
            return False
    
    def wait(self, turns: int = 1):
        """Make agent wait for specified turns"""
        self.is_waiting = True
        self.wait_turns_remaining = turns
        self.current_action = f"waiting_{turns}_turns"
    
    def update_turn(self):
        """Update agent state for new turn"""
        if self.is_waiting and self.wait_turns_remaining > 0:
            self.wait_turns_remaining -= 1
            if self.wait_turns_remaining <= 0:
                self.is_waiting = False
                self.current_action = None
    
    def pickup_box(self, box_id: int):
        """Agent picks up a box"""
        if not self.carrying_box:
            self.carrying_box = True
            self.box_id = box_id
            return True
        return False
    
    def drop_box(self) -> Optional[int]:
        """Agent drops the box it's carrying"""
        if self.carrying_box:
            dropped_box_id = self.box_id
            self.carrying_box = False
            self.box_id = None
            return dropped_box_id
        return None
    
    def get_status(self) -> Dict:
        """Get agent's current status"""
        return {
            'id': self.agent_id,
            'position': self.position,
            'target': self.target_position,
            'carrying_box': self.carrying_box,
            'box_id': self.box_id,
            'planned_path': self.planned_path,
            'is_waiting': self.is_waiting,
            'wait_turns_remaining': self.wait_turns_remaining,
            'priority': self.priority,
            'current_action': self.current_action
        }
    
    def distance_to_target(self) -> float:
        """Calculate Manhattan distance to target"""
        if not self.target_position:
            return float('inf')
        
        return (abs(self.position[0] - self.target_position[0]) + 
                abs(self.position[1] - self.target_position[1]))
    
    def __str__(self) -> str:
        status = "@" if self.carrying_box else "A"
        return f"Agent{self.agent_id}({status})@{self.position}"
    
    def __repr__(self) -> str:
        return self.__str__()
