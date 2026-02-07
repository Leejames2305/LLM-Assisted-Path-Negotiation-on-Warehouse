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
        self.executed_path = [initial_position]  # Track movement history, starts with initial position
        self.current_action = None
        self.validator = AgentValidator()
        self.pathfinder = None  # Will be initialized with map size
        
        # Agent state
        self.is_waiting = False
        self.wait_turns_remaining = 0
        self.priority = 1
    
    # Set the agent's target position
    def set_target(self, target_position: Tuple[int, int]):
        self.target_position = target_position
        self.planned_path = []  # Reset path when target changes
        
        # Clear negotiated path flag when getting new target
        if hasattr(self, '_has_negotiated_path'):
            self._has_negotiated_path = False
    
    # Plan a path to the target using pathfinding
    def plan_path(self, map_state: Dict) -> List[Tuple[int, int]]:
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
    
    # Directly set the planned path (used for negotiated paths)
    def set_path(self, path: List[Tuple[int, int]]):
        self.planned_path = path
    
    # Get the next position to move to
    def get_next_move(self) -> Optional[Tuple[int, int]]:
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
    
    # Execute an action received from central negotiator, return true if successful
    def execute_negotiated_action(self, action_data: Dict, map_state: Dict) -> bool:
        # Check if this agent was already validated by HMAS-2
        if getattr(self, '_hmas2_validated', False):
            return self._execute_action(action_data, map_state)
        
        # Otherwise, perform normal validation
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
    
    # Internal method to execute an action
    def _execute_action(self, action_data: Dict, map_state: Dict) -> bool:
        action = action_data.get('action', 'wait')
        
        if action == 'wait':
            wait_turns = action_data.get('wait_turns', 1)
            self.wait(wait_turns)
            return True
        
        elif action == 'move':
            path = action_data.get('path', [])
            # Execute the path in order from the beginning
            # If path[0] is current position, agent stays in place (effectively a wait)
            # If path[0] is different, agent moves there
            # This handles both cases where LLM includes/excludes current position
            if path and len(path) > 0:
                next_pos = tuple(path[0])  # Start from the first element
                success, failure_reason = self.move_to(next_pos, map_state)
                if not success and failure_reason:
                    print(f"   ⚠️ Action execution failed: {failure_reason}")
                return success
            else:
                next_move = self.get_next_move()
                if next_move:
                    success, failure_reason = self.move_to(next_move, map_state)
                    if not success and failure_reason:
                        print(f"   ⚠️ Action execution failed: {failure_reason}")
                    return success
        
        return False
    
    # Move agent to new position with safety checks, return (success, reason)
    def move_to(self, new_position: Tuple[int, int], map_state: Dict) -> Tuple[bool, Optional[str]]:
        # Check if position is the same (waiting in place)
        if new_position == self.position:
            print(f"✅ Agent {self.agent_id}: Staying in place (valid wait)")
            return True, None
        
        # Basic adjacency check
        dx = abs(new_position[0] - self.position[0])
        dy = abs(new_position[1] - self.position[1])
        if dx + dy != 1:
            reason = f"not_adjacent (dx={dx}, dy={dy})"
            print(f"❌ Agent {self.agent_id}: Move {reason}")
            return False, reason
        
        # Detailed safety check using validator
        is_safe = self.validator.check_move_safety(
            self.agent_id, self.position, new_position, map_state
        )
        
        if not is_safe:
            print(f"❌ Agent {self.agent_id}: Safety check failed for {new_position}")
            # Get the failure reason and log it
            failure_reason = self._get_move_failure_reason(new_position, map_state)
            print(f"   Reason: {failure_reason}")
            return False, failure_reason
        
        # If safe, make the move
        self.position = new_position
        print(f"✅ Agent {self.agent_id}: Moved to {new_position}")
        
        # Track executed path (movement history)
        self.executed_path.append(new_position)
        
        # Update planned path - but don't interfere with negotiated paths managed by game engine
        if not (hasattr(self, '_has_negotiated_path') and getattr(self, '_has_negotiated_path', False)):
            # Only update path if this is NOT a negotiated path (let game engine handle negotiated paths)
            if self.planned_path and new_position in self.planned_path:
                idx = self.planned_path.index(new_position)
                self.planned_path = self.planned_path[idx:]
        
        # Check if reached target
        if self.position == self.target_position:
            self.planned_path = []
            # Clear negotiated path flag when target is reached
            if hasattr(self, '_has_negotiated_path'):
                self._has_negotiated_path = False
        
        return True, None
    
    # Get detailed reason why a move failed
    def _get_move_failure_reason(self, new_position: Tuple[int, int], map_state: Dict) -> str:
        x, y = new_position
        grid = map_state.get('grid', [])
        
        # Check bounds
        if not grid or y < 0 or y >= len(grid) or x < 0 or x >= len(grid[0]):
            return f"out_of_bounds: ({x}, {y})"
        
        # Check wall collision
        if grid[y][x] == '#':
            return f"wall_collision: ({x}, {y})"
        
        # Check agent collisions
        agents = map_state.get('agents', {})
        for other_id, other_pos in agents.items():
            if other_id != self.agent_id and other_pos == new_position:
                return f"agent_collision: Agent {other_id} at {other_pos}"
        
        # Check box collisions
        boxes = map_state.get('boxes', {})
        for box_id, box_pos in boxes.items():
            if box_pos == new_position:
                return f"box_collision: Box {box_id} at {box_pos}"
        
        return "unknown_safety_check_failure"
    
    # Make agent wait for specified turns
    def wait(self, turns: int = 1):
        self.is_waiting = True
        self.wait_turns_remaining = turns
        self.current_action = f"waiting_{turns}_turns"
    
    # Update agent state for new turn
    def update_turn(self):
        if self.is_waiting and self.wait_turns_remaining > 0:
            self.wait_turns_remaining -= 1
            if self.wait_turns_remaining <= 0:
                self.is_waiting = False
                self.current_action = None
        
        # Only clear negotiated path flag if agent has definitely reached their target
        if hasattr(self, '_has_negotiated_path') and self._has_negotiated_path:
            if self.position == self.target_position:
                self._has_negotiated_path = False
                self.planned_path = []
                print(f"🔄 Agent {self.agent_id}: Cleared negotiated path flag (reached target)")
    
    # Agent picking up a box
    def pickup_box(self, box_id: int):
        if not self.carrying_box:
            self.carrying_box = True
            self.box_id = box_id
            return True
        return False
    
    # Agent dropping the box it's carrying
    def drop_box(self) -> Optional[int]:
        if self.carrying_box:
            dropped_box_id = self.box_id
            self.carrying_box = False
            self.box_id = None
            return dropped_box_id
        return None
    
    # Get current status of the agent
    def get_status(self) -> Dict:
        return {
            'id': self.agent_id,
            'position': self.position,
            'target': self.target_position,
            'carrying_box': self.carrying_box,
            'box_id': self.box_id,
            'planned_path': self.planned_path,
            'executed_path': self.executed_path,
            'is_waiting': self.is_waiting,
            'wait_turns_remaining': self.wait_turns_remaining,
            'priority': self.priority,
            'current_action': self.current_action
        }
    
    # Calculate Manhattan distance to target
    def distance_to_target(self) -> float:
        if not self.target_position:
            return float('inf')
        
        return (abs(self.position[0] - self.target_position[0]) + 
                abs(self.position[1] - self.target_position[1]))
    
    def __str__(self) -> str:
        status = "@" if self.carrying_box else "A"
        return f"Agent{self.agent_id}({status})@{self.position}"
    
    def __repr__(self) -> str:
        return self.__str__()
