"""
Agent LLM Validator for Action Validation
Uses smaller model for quick validation checks
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Union
from ..llm import OpenRouterClient
from .openrouter_config import OpenRouterConfig

class AgentValidator:
    def __init__(self, model: Optional[str] = None):
        self.client = OpenRouterClient()
        self.model = model or os.getenv('AGENT_LLM_MODEL', 'google/gemma-2-9b-it:free')
        
        # Check if we're using a reasoning model for validation
        self.is_reasoning_model = OpenRouterConfig.is_reasoning_model(self.model)
    
    def _is_reasoning_model(self, model: str) -> bool:
        """Check if the model supports reasoning features (delegated to config)"""
        return OpenRouterConfig.is_reasoning_model(model)
    
    def validate_negotiated_action(self, agent_id: int, proposed_action: Dict, current_state: Dict) -> Dict:
        """
        Validate if a negotiated action is safe and executable for an agent
        
        Args:
            agent_id: The agent's ID
            proposed_action: {'action': 'move'/'wait', 'path': [...], 'priority': int}
            current_state: Current map and agent state
        
        Returns:
            validation_result: {'valid': bool, 'reason': str, 'alternative': Dict/None}
        """
        
        # Quick validation for simple wait actions
        if proposed_action.get('action') == 'wait':
            return {
                'valid': True,
                'reason': 'Wait action is always safe'
            }
        
        # Check if agent position is available
        agents = current_state.get('agents', {})
        current_pos = None
        
        # Try multiple approaches to find the agent position
        if agent_id in agents:
            current_pos = agents[agent_id]
        elif str(agent_id) in agents:
            current_pos = agents[str(agent_id)]
        else:
            # Try converting all keys to int and check again
            for key, pos in agents.items():
                try:
                    if int(key) == int(agent_id):
                        current_pos = pos
                        break
                except (ValueError, TypeError):
                    continue
        
        if current_pos is None:
            return {
                'valid': False,
                'reason': f'Agent {agent_id} position not found in map state',
                'alternative': {'action': 'wait', 'reason': 'position_unknown'}
            }
        
        system_prompt = self._create_validation_system_prompt()
        user_prompt = self._create_validation_query(agent_id, proposed_action, current_state)
        
        # Enhanced validation for reasoning models
        if self.is_reasoning_model:
            user_prompt = self._add_validation_reasoning_instructions(user_prompt)
        
        messages = [
            self.client.create_system_message(system_prompt),
            self.client.create_user_message(user_prompt)
        ]
        
        # Adjust parameters for reasoning models - use lower token limits for efficiency
        max_tokens = 15000 if self.is_reasoning_model else 10000
        temperature = 0.05 if self.is_reasoning_model else 0.1  # Very low temperature for consistent validation
        
        response = self.client.send_request(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        if response:
            try:
                result = self._parse_validation_response(response)
                
                # If validation failed and no alternative provided, suggest wait
                if not result.get('valid', False) and 'alternative' not in result:
                    result['alternative'] = {'action': 'wait', 'reason': 'validation_failed'}
                
                return result
            except Exception as e:
                return self._create_permissive_fallback_with_alternative()
        else:
            return self._create_permissive_fallback_with_alternative()
    
    def check_move_safety(self, agent_id: int, from_pos: Tuple[int, int], to_pos: Tuple[int, int], map_state: Dict) -> bool:
        """
        Quick safety check for a single move
        
        Args:
            agent_id: Agent making the move
            from_pos: Current position (x, y)
            to_pos: Target position (x, y)
            map_state: Current map state
        
        Returns:
            bool: True if move is safe, False otherwise
        """
        
        # First do basic validation - this should be deterministic, not LLM-based
        to_x, to_y = to_pos
        from_x, from_y = from_pos
        
        # Check if target position is within map bounds
        grid = map_state.get('grid', [])
        if not grid or to_y < 0 or to_y >= len(grid) or to_x < 0 or to_x >= len(grid[0]):
            return False
        
        # Check if target position is a wall
        if grid[to_y][to_x] == '#':
            return False
        
        # Check if move is to adjacent cell (no diagonal moves)
        dx = abs(to_x - from_x)
        dy = abs(to_y - from_y)
        if dx + dy != 1:  # Not adjacent
            return False
        
        # Check if another agent is at target position
        other_agents = map_state.get('agents', {})
        for other_agent_id, other_pos in other_agents.items():
            if other_agent_id != agent_id and other_pos == to_pos:
                return False
        
        return True
    
    def suggest_alternative_action(self, agent_id: int, failed_action: Dict, current_state: Dict) -> Optional[Dict]:
        """
        Suggest alternative action when original action fails validation
        """
        
        system_prompt = """You are helping a robot find an alternative action when its planned action is invalid.

        Suggest alternatives like:
        - Wait for a turn
        - Move to a different adjacent cell
        - Take a longer but safer path

        Respond with JSON: {"action": "move/wait", "path": [[x,y]...], "reasoning": "explanation"}"""
        
        user_prompt = f"""Agent {agent_id} cannot execute: {failed_action}
        Current state: {current_state}

        Suggest a safe alternative action."""
        
        messages = [
            self.client.create_system_message(system_prompt),
            self.client.create_user_message(user_prompt)
        ]
        
        response = self.client.send_request(
            model=self.model,
            messages=messages,
            max_tokens=20000,
            temperature=0.3
        )
        
        if response:
            try:
                return json.loads(response.strip())
            except:
                return {"action": "wait", "path": [], "reasoning": "Safe fallback: wait"}
        
        return None
    
    def _create_validation_system_prompt(self) -> str:
        """Create system prompt for action validation"""
        return """You are an Agent Validator for warehouse robots. Your job is to perform basic safety validation of negotiated actions from a trusted central coordinator.

        CRITICAL VALIDATION RULES - CHECK THE ENTIRE PATH:
        
        1. NO DIAGONAL MOVES ALLOWED
           - Moves must be orthogonal only: up, down, left, right
           - Each step from (x,y) can ONLY go to: (x±1,y), (x,y±1)
           - Any diagonal move like (x±1,y±1) is INVALID
           - Check ALL moves in the path, not just the first move
        
        2. ZERO-MOVES ARE ACCEPTABLE
           - A step like (x,y) → (x,y) is valid (agent stays in place)
           - This is equivalent to waiting and should NOT be rejected
        
        3. NO WALL COLLISIONS
           - Any position in the path marked as '#' is a wall and BLOCKED
           - The path cannot contain walls
           - Check ALL positions in the provided path
        
        4. WITHIN MAP BOUNDS
           - All path positions must be within (0,0) to (max_x, max_y)
        
        5. TRUST CENTRAL COORDINATOR FOR TIMING
           - Don't second-guess temporal coordination
           - Focus on geometric/physical validity only

        MAP SYMBOLS:
        - '#' = Wall (BLOCKED)
        - '.' = Open space

        RESPONSE FORMAT (JSON):
        {
            "valid": true/false,
            "reason": "specific reason"
        }

        VALIDATION PHILOSOPHY: Reject any path with diagonals or walls. Accept zero-moves (staying in place). Otherwise approve."""
    
    def _create_validation_query(self, agent_id: int, proposed_action: Dict, current_state: Dict) -> str:
        """Create validation query with entire path checking"""
        # Extract agent positions and find this agent's current position
        agents = current_state.get('agents', {})
        current_pos = agents.get(agent_id) or agents.get(str(agent_id))
        
        # If still not found, this is a critical error
        if current_pos is None:
            return f"""VALIDATION ERROR: Agent {agent_id} position not found in map state.
            Available agents: {list(agents.keys())}
            Cannot validate action without current position.
            Respond with: {{"valid": false, "reason": "agent_position_unknown"}}"""
        
        # Get map info
        grid = current_state.get('grid', [])
        if grid:
            height = len(grid)
            width = len(grid[0]) if grid[0] else 0
            max_x, max_y = width - 1, height - 1
        else:
            height, width = 0, 0
            max_x, max_y = 0, 0
        
        # Extract the action details
        action_type = proposed_action.get('action', 'unknown')
        path = proposed_action.get('path', [])
        
        # Convert path to tuples for easier analysis
        path_tuples = [tuple(p) if isinstance(p, (list, tuple)) else p for p in path]
        
        query = f"""Validate negotiated action for Agent {agent_id}:

        AGENT STATE:
        - Current position: {current_pos}
        - Proposed action: {action_type}
        - Full path: {path_tuples}
        - Map size: {width}x{height} (bounds: x:[0-{max_x}], y:[0-{max_y}])

        CRITICAL VALIDATION - CHECK ENTIRE PATH:

        1. DIAGONAL MOVE CHECK:
           For each consecutive pair of positions in path, verify it's orthogonal:
           - (x,y) to (x+1,y) or (x-1,y) or (x,y+1) or (x,y-1) = VALID (no diagonals)
           - (x,y) to (x+1,y+1) or any diagonal = INVALID

        2. WALL CHECK:
           Verify no position in path contains a wall (#):
"""
        
        # Add map grid for reference
        if grid:
            query += "           Map grid:\n"
            for y, row in enumerate(grid):
                query += "           "
                for x, cell in enumerate(row):
                    if (x, y) in path_tuples:
                        query += "P"  # Path position
                    elif (x, y) == tuple(current_pos):
                        query += "@"  # Current agent
                    else:
                        query += cell
                query += f"  (y={y})\n"
            query += "           x coords: " + "".join(str(x % 10) for x in range(width)) + "\n"
        
        query += f"""
        3. BOUNDS CHECK:
           All positions must be within x:[0-{max_x}] and y:[0-{max_y}]

        PATH VALIDATION SUMMARY:
"""
        
        # Analyze path for issues
        issues = []
        for i in range(len(path_tuples) - 1):
            curr = path_tuples[i]
            next_pos = path_tuples[i + 1]
            
            # Check if it's a zero-move (staying in place) - this is valid (equivalent to waiting)
            if curr == next_pos:
                # Zero-move is acceptable, it's like waiting in place
                continue
            
            # Check diagonal
            dx = abs(next_pos[0] - curr[0])
            dy = abs(next_pos[1] - curr[1])
            if dx > 1 or dy > 1 or (dx == 1 and dy == 1):
                issues.append(f"Step {i}-{i+1}: Diagonal move {curr} → {next_pos} (dx={dx}, dy={dy}) - INVALID")
            
            # Check bounds
            if not (0 <= next_pos[0] <= max_x and 0 <= next_pos[1] <= max_y):
                issues.append(f"Step {i+1}: Out of bounds {next_pos}")
            
            # Check walls
            if grid and 0 <= next_pos[1] < height and 0 <= next_pos[0] < width:
                if grid[next_pos[1]][next_pos[0]] == '#':
                    issues.append(f"Step {i+1}: Wall at {next_pos}")
        
        if issues:
            query += "\n".join("       - " + issue for issue in issues)
            query += "\n\n       ❌ PATH INVALID - Contains issues above"
        else:
            query += "       ✓ All moves are orthogonal (no diagonals) or zero-moves (staying in place)\n"
            query += "       ✓ No walls in path\n"
            query += "       ✓ All positions within bounds\n"
            query += "\n       ✅ PATH VALID"
        
        query += f"""

        FINAL VALIDATION:
        - If path contains ANY diagonal moves → REJECT with reason like "diagonal_move_at_step_0: (3,4) → (2,5)"
        - If path contains ANY wall positions → REJECT with reason like "wall_violation_at: (1,4)"
        - If any position out of bounds → REJECT with reason like "out_of_bounds_at: (10,5)"
        - Zero-moves (same position twice like (2,4)→(2,4)) are VALID, do NOT reject them
        - Otherwise → APPROVE with reason "path_valid_orthogonal_moves_no_walls" or "path_valid_includes_zero_moves"

        IMPORTANT: In your reason field, be SPECIFIC with positions and step numbers. Examples:
        - "wall_collision_at_position: (1,4)"
        - "diagonal_move_at_step_2: (3,4)→(2,5) (dx=1, dy=1)"
        - "out_of_bounds_at_step_4: (10,6) exceeds max (9,7)"

        Respond with JSON: {{"valid": true/false, "reason": "specific_reason_with_details"}}"""
        
        return query
    
    def _add_validation_reasoning_instructions(self, base_prompt: str) -> str:
        """Add reasoning instructions for validation"""
        reasoning_instructions = """
        EFFICIENT VALIDATION REASONING:
        1. Focus on immediate safety: Can this agent make this specific move right now?
        2. Check only physical constraints: walls, boundaries, current occupants
        3. Trust central coordinator for temporal coordination and complex routing
        4. Default to APPROVE unless clear immediate danger

        Keep reasoning brief and focused on safety essentials.

        """
        return reasoning_instructions + base_prompt
    
    def _parse_validation_response(self, response: str) -> Dict:
        """Parse validation response"""
        response = response.strip()
        
        # Look for JSON
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            try:
                result = json.loads(json_str)
                # Ensure required fields
                if 'valid' not in result:
                    result['valid'] = True  # Default to trusting central coordinator
                if 'reason' not in result:
                    result['reason'] = "Basic safety validated"
                    
                # If validation failed, add alternative for proper rejection counting
                if not result.get('valid', True) and 'alternative' not in result:
                    result['alternative'] = {'action': 'wait', 'reason': 'llm_rejection'}
                    
                return result
            except json.JSONDecodeError:
                pass
        
        return self._create_permissive_fallback()
    
    def _create_permissive_fallback(self) -> Dict:
        """Create permissive fallback validation result that trusts central coordinator"""
        return {
            "valid": True,  # Trust the central coordinator by default
            "reason": "Validation response unclear, trusting central coordinator decision"
        }
    
    def _create_permissive_fallback_with_alternative(self) -> Dict:
        """Create permissive fallback with alternative for counting rejections properly"""
        return {
            "valid": False,  # Mark as failed for proper counting
            "reason": "Validation failed, suggesting wait as safe alternative",
            "alternative": {"action": "wait", "reason": "validation_error"}
        }
    
    def _create_safe_fallback(self) -> Dict:
        """Create fallback validation result - now more permissive"""
        return {
            "valid": True,  # Trust central coordinator in fallback scenarios too
            "reason": "Using permissive fallback, trusting central coordinator",
        }
    
    def _basic_move_validation(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], map_state: Dict) -> bool:
        """Basic move validation without LLM"""
        try:
            # Check if move is to adjacent cell
            dx = abs(to_pos[0] - from_pos[0])
            dy = abs(to_pos[1] - from_pos[1])
            
            if dx + dy != 1:  # Not adjacent
                return False
            
            # Check if target position is occupied by another agent
            for agent_pos in map_state.get('agents', {}).values():
                if agent_pos == to_pos:
                    return False
            
            # Basic bounds check (assuming 8x6 map)
            if not (0 <= to_pos[0] < 8 and 0 <= to_pos[1] < 6):
                return False
            
            return True
            
        except:
            return False  # If anything goes wrong, be safe
