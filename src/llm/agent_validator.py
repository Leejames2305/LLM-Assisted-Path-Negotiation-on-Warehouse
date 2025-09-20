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
        
        # Debug output to see what we're working with
        print(f"🔍 DEBUG: Validating for agent {agent_id}")
        print(f"🔍 DEBUG: Map state agents: {current_state.get('agents', {})}")
        print(f"🔍 DEBUG: Action: {proposed_action}")
        
        # Quick validation for simple wait actions
        if proposed_action.get('action') == 'wait':
            return {
                'valid': True,
                'reason': 'Wait action is always safe'
            }
        
        # Check if agent position is available - IMPROVED LOGIC
        agents = current_state.get('agents', {})
        current_pos = None
        
        # Try multiple approaches to find the agent position with better debugging
        print(f"🔍 DEBUG: Looking for agent_id {agent_id} (type: {type(agent_id)}) in agents: {agents}")
        print(f"🔍 DEBUG: Agent keys: {list(agents.keys())} (types: {[type(k) for k in agents.keys()]})")
        
        if agent_id in agents:
            current_pos = agents[agent_id]
            print(f"🔍 DEBUG: Found agent {agent_id} directly")
        elif str(agent_id) in agents:
            current_pos = agents[str(agent_id)]
            print(f"🔍 DEBUG: Found agent {agent_id} as string")
        else:
            # Try converting all keys to int and check again
            for key, pos in agents.items():
                try:
                    if int(key) == int(agent_id):
                        current_pos = pos
                        print(f"🔍 DEBUG: Found agent {agent_id} via int conversion from key {key}")
                        break
                except (ValueError, TypeError):
                    continue
        
        print(f"🔍 DEBUG: Agent {agent_id} current position found: {current_pos}")
        
        if current_pos is None:
            print(f"🔍 DEBUG: Agent {agent_id} position not found! Available: {list(agents.keys())}")
            return {
                'valid': False,
                'reason': f'Agent {agent_id} position not found in map state',
                'alternative': {'action': 'wait', 'reason': 'position_unknown'}
            }
        
        print(f"🔍 DEBUG: Agent {agent_id} current position: {current_pos}")
        
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
                print(f"🔍 DEBUG: Validator response: {result}")
                
                # If validation failed and no alternative provided, suggest wait
                if not result.get('valid', False) and 'alternative' not in result:
                    result['alternative'] = {'action': 'wait', 'reason': 'validation_failed'}
                
                return result
            except Exception as e:
                print(f"🔍 DEBUG: Validation parsing error: {e}")
                return self._create_permissive_fallback_with_alternative()
        else:
            print(f"🔍 DEBUG: No response from validator")
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

        TRUST THE CENTRAL COORDINATOR: The central negotiator has already analyzed spatial conflicts and temporal coordination. Your role is focused safety validation only.

        FOCUS ON BASIC SAFETY ONLY:
        - Check if immediate next move is physically possible (adjacent cell, not diagonal)
        - Verify no collision with walls (#) or current agent/box positions
        - Ensure move stays within map boundaries
        - Trust coordination timing - don't second-guess when agents should move

        MAP SYMBOLS:
        - '#' = Wall (BLOCKED)
        - '.' = Open space

        RESPONSE FORMAT (JSON):
        {
            "valid": true/false,
            "reason": "brief safety check result"
        }

        VALIDATION PHILOSOPHY: Approve unless there's a clear immediate safety issue. Trust the central coordinator for complex spatial reasoning."""
    
    def _create_validation_query(self, agent_id: int, proposed_action: Dict, current_state: Dict) -> str:
        """Create validation query with proper position handling"""
        # Extract agent positions and find this agent's current position
        agents = current_state.get('agents', {})
        current_pos = agents.get(agent_id) or agents.get(str(agent_id))
        
        # If still not found, this is a critical error
        if current_pos is None:
            return f"""VALIDATION ERROR: Agent {agent_id} position not found in map state.
            Available agents: {list(agents.keys())}
            Cannot validate action without current position.
            Respond with: {{"valid": false, "reason": "agent_position_unknown"}}"""
        
        # Get basic map info
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
        next_move = None
        
        if path and len(path) > 0:
            # The path should start from current position or next position
            if len(path) > 1 and tuple(path[0]) == tuple(current_pos):
                next_move = tuple(path[1])  # Second position is the next move
            elif len(path) >= 1:
                next_move = tuple(path[0])  # First position is the next move
        
        query = f"""Validate negotiated action for Agent {agent_id}:

        AGENT STATE:
        - Current position: {current_pos}
        - Proposed action: {action_type}
        - Next move: {next_move}
        - Full path preview: {path[:3]}... (showing first 3 steps)

        MAP INFO:
        - Map size: {width}x{height}
        - All agent positions: {agents}

        VALIDATION CHECKS:
        1. Is next move {next_move} adjacent to current position {current_pos}?
        2. Is next move within bounds (0-{max_x}, 0-{max_y})?
        3. Is next move not into a wall?

        Map view around Agent {agent_id} at {current_pos}:"""

        # Add 3x3 local map view
        if grid and current_pos != 'unknown':
            try:
                cx, cy = current_pos
                for dy in range(-1, 2):
                    row = ""
                    for dx in range(-1, 2):
                        x, y = cx + dx, cy + dy
                        if 0 <= y < height and 0 <= x < width:
                            cell = grid[y][x]
                            if (x, y) == tuple(current_pos):
                                row += " @"  # Current agent
                            elif (x, y) == next_move:
                                row += " >"  # Next move target
                            elif (x, y) in [tuple(pos) for pos in agents.values() if tuple(pos) != tuple(current_pos)]:
                                row += " A"  # Other agent
                            elif cell == '#':
                                row += " #"  # Wall
                            else:
                                row += " ."  # Empty
                        else:
                            row += " X"  # Out of bounds
                    query += row + "\n"
            except:
                query += "Map view unavailable\n"
        
        query += f"""
        LEGEND: @ = Agent {agent_id}, > = Next move target, A = Other agents, # = Wall, . = Empty, X = Out of bounds

        REMEMBER: Trust central coordination for timing - focus only on immediate move safety.

        Respond with JSON: {{"valid": true/false, "reason": "explanation"}}"""
        
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
