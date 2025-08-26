"""
Agent LLM Validator for Action Validation
Uses smaller model (gemma-2-9b) for quick validation checks
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Union
from ..llm import OpenRouterClient

class AgentValidator:
    def __init__(self, model: Optional[str] = None):
        self.client = OpenRouterClient()
        self.model = model or os.getenv('AGENT_LLM_MODEL', 'google/gemma-2-9b-it:free')
    
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
        
        system_prompt = self._create_validation_system_prompt()
        user_prompt = self._create_validation_query(agent_id, proposed_action, current_state)
        
        messages = [
            self.client.create_system_message(system_prompt),
            self.client.create_user_message(user_prompt)
        ]
        
        response = self.client.send_request(
            model=self.model,
            messages=messages,
            max_tokens=500,
            temperature=0.1  # Very low temperature for consistent validation
        )
        
        if response:
            try:
                return self._parse_validation_response(response)
            except Exception as e:
                print(f"Error parsing validation response: {e}")
                return self._create_safe_fallback()
        else:
            return self._create_safe_fallback()
    
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
        
        system_prompt = """You are a safety validator for robot movements. Check if a move is valid and safe.

Rules:
- Can only move to adjacent cells (not diagonal)
- Cannot move into walls (#)
- Cannot move into cells occupied by other agents
- Must stay within map bounds

Respond with JSON: {"safe": true/false, "reason": "explanation"}"""
        
        user_prompt = f"""Validate move for Agent {agent_id}:
From: {from_pos}
To: {to_pos}

Map state: {map_state}

Is this move safe?"""
        
        messages = [
            self.client.create_system_message(system_prompt),
            self.client.create_user_message(user_prompt)
        ]
        
        response = self.client.send_request(
            model=self.model,
            messages=messages,
            max_tokens=200,
            temperature=0.0
        )
        
        if response:
            try:
                result = json.loads(response.strip())
                return result.get('safe', False)
            except:
                # If parsing fails, do basic validation
                return self._basic_move_validation(from_pos, to_pos, map_state)
        
        return self._basic_move_validation(from_pos, to_pos, map_state)
    
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
            max_tokens=300,
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
        return """You are an Agent Validator for warehouse robots. Your job is to verify that negotiated actions are safe and executable.

VALIDATION RULES:
- Agents can only move to adjacent cells (not diagonal)
- Cannot move into walls (#) or occupied cells
- Must consider current agent position and map state
- Check if proposed path is physically possible

RESPONSE FORMAT (JSON):
{
    "valid": true/false,
    "reason": "explanation of validation result",
    "alternative": {"action": "move/wait", "path": [...]} or null
}

Be strict about safety. When in doubt, mark as invalid."""
    
    def _create_validation_query(self, agent_id: int, proposed_action: Dict, current_state: Dict) -> str:
        """Create validation query"""
        agent_pos = current_state.get('agents', {}).get(agent_id, 'unknown')
        
        query = f"""VALIDATION REQUEST for Agent {agent_id}:

Current Position: {agent_pos}
Proposed Action: {proposed_action}

Current Map State:
Agents: {current_state.get('agents', {})}
Boxes: {current_state.get('boxes', {})}
Targets: {current_state.get('targets', {})}

Is this action valid and safe to execute?"""
        
        return query
    
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
                    result['valid'] = False
                if 'reason' not in result:
                    result['reason'] = "Validation failed"
                return result
            except json.JSONDecodeError:
                pass
        
        return self._create_safe_fallback()
    
    def _create_safe_fallback(self) -> Dict:
        """Create safe fallback validation result"""
        return {
            "valid": False,
            "reason": "Validation failed, defaulting to safe rejection",
            "alternative": {"action": "wait", "path": [], "reasoning": "Safe fallback"}
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
