"""
Agent LLM Validator for Action Validation (POGEMA port).
Adapted from src/llm/agent_validator.py — box validation removed, (row,col) coordinates used.
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Union
from .openrouter_client import OpenRouterClient
from .openrouter_config import OpenRouterConfig


class AgentValidator:
    def __init__(self, model: Optional[str] = None):
        self.client = OpenRouterClient()
        self.model = model or os.getenv('AGENT_LLM_MODEL', 'nvidia/nemotron-3-nano-30b-a3b:free')
        self.is_reasoning_model = OpenRouterConfig.is_reasoning_model(self.model)

    def _is_reasoning_model(self, model: str) -> bool:
        return OpenRouterConfig.is_reasoning_model(model)

    # Check if a negotiated action is safe and executable
    def validate_negotiated_action(
        self, agent_id: int, proposed_action: Dict, current_state: Dict
    ) -> Dict:

        if proposed_action.get('action') == 'wait':
            return {'valid': True, 'reason': 'Wait action is always safe'}

        agents = current_state.get('agents', {})
        current_pos = None

        if agent_id in agents:
            current_pos = agents[agent_id]
        elif str(agent_id) in agents:
            current_pos = agents[str(agent_id)]
        else:
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
                'alternative': {'action': 'wait', 'reason': 'position_unknown'},
            }

        system_prompt = self._create_validation_system_prompt()
        user_prompt = self._create_validation_query(agent_id, proposed_action, current_state)

        if self.is_reasoning_model:
            user_prompt = self._add_validation_reasoning_instructions(user_prompt)

        messages = [
            self.client.create_system_message(system_prompt),
            self.client.create_user_message(user_prompt),
        ]

        temperature = 0.2 if self.is_reasoning_model else 0.3

        response = self.client.send_request(
            model=self.model,
            messages=messages,
            max_tokens=50000,
            temperature=temperature,
        )

        if response:
            try:
                result = self._parse_validation_response(response)  # type: ignore
                if not result.get('valid', False) and 'alternative' not in result:
                    result['alternative'] = {'action': 'wait', 'reason': 'validation_failed'}
                return result
            except Exception:
                return self._create_permissive_fallback_with_alternative()
        else:
            return self._create_permissive_fallback_with_alternative()

    # Quick safety check for a single move — deterministic, no LLM
    def check_move_safety(
        self,
        agent_id: int,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        map_state: Dict,
    ) -> bool:

        from_r, from_c = from_pos
        to_r, to_c = to_pos

        # Check bounds
        grid = map_state.get('grid', [])
        if not grid or to_r < 0 or to_r >= len(grid) or to_c < 0 or to_c >= len(grid[0]):
            return False

        # Check wall (POGEMA: 1 = wall, '#' in char grid)
        cell = grid[to_r][to_c]
        if cell == 1 or cell == '#':
            return False

        # Check adjacency
        dr = abs(to_r - from_r)
        dc = abs(to_c - from_c)
        if dr + dc != 1:
            return False

        # Check collision with other agents
        other_agents = map_state.get('agents', {})
        for other_id, other_pos in other_agents.items():
            if other_id != agent_id and other_pos == to_pos:
                return False

        return True

    # Suggest alternative action when original action fails validation
    def suggest_alternative_action(
        self, agent_id: int, failed_action: Dict, current_state: Dict
    ) -> Optional[Dict]:

        system_prompt = """You are helping a robot find an alternative action when its planned action is invalid.
        Coordinates use (row, col) format. Origin (0,0) is TOP-LEFT.

        Suggest alternatives like:
        - Wait for a turn
        - Move to a different adjacent cell

        Respond with JSON: {"action": "move/wait", "path": [[row,col]...], "reasoning": "explanation"}
        """

        user_prompt = f"""Agent {agent_id} cannot execute: {failed_action}
        Current state: {current_state}

        Suggest a safe alternative action.
        """

        messages = [
            self.client.create_system_message(system_prompt),
            self.client.create_user_message(user_prompt),
        ]

        response = self.client.send_request(
            model=self.model,
            messages=messages,
            max_tokens=50000,
            temperature=0.3,
        )

        if response:
            try:
                return json.loads(response.strip())  # type: ignore
            except Exception:
                return {"action": "wait", "path": [], "reasoning": "Safe fallback: wait"}

        return None

    def _create_validation_system_prompt(self) -> str:
        return """You are an Agent Validator for warehouse robots navigating to their targets.
        All positions use (row, col) format. Origin (0,0) is TOP-LEFT.

        CORE RULES:
        1. Only orthogonal adjacent moves (up, down, left, right)
           - Each step from (r,c) can ONLY go to: (r±1,c), (r,c±1)
           - Any diagonal move is INVALID

        2. Zero-moves (staying in place) are VALID
           - A step like (r,c) → (r,c) is valid (agent waits in place)

        3. No collisions with walls
           - Any position marked as wall (1 or '#') is BLOCKED

        4. Within map bounds
           - All positions must be within (0,0) to (max_row, max_col)

        5. Trust the central coordinator for timed-spacing

        MAP SYMBOLS:
        - 1 or '#' = Wall (BLOCKED)
        - 0 or '.' = Open space

        RESPONSE FORMAT (JSON):
        {
            "valid": true/false,
            "reason": "specific reason"
        }

        VALIDATION PHILOSOPHY: Reject diagonal or wall collisions. Accept zero-moves. Otherwise approve.
        """

    def _create_validation_query(
        self, agent_id: int, proposed_action: Dict, current_state: Dict
    ) -> str:

        agents = current_state.get('agents', {})
        current_pos = agents.get(agent_id) or agents.get(str(agent_id))

        if current_pos is None:
            return f"""VALIDATION ERROR: Agent {agent_id} position not found.
            Respond with: {{"valid": false, "reason": "agent_position_unknown"}}"""

        grid = current_state.get('grid', [])
        if grid:
            height = len(grid)
            width = len(grid[0]) if grid[0] else 0
            max_r, max_c = height - 1, width - 1
        else:
            height, width = 0, 0
            max_r, max_c = 0, 0

        action_type = proposed_action.get('action', 'unknown')
        path = proposed_action.get('path', [])
        path_tuples = [tuple(p) if isinstance(p, (list, tuple)) else p for p in path]

        query = f"""Validate negotiated action for Agent {agent_id}:

        AGENT STATE:
        - Current position (row, col): {current_pos}
        - Proposed action: {action_type}
        - Full path: {path_tuples}
        - Map size: {height}×{width} (bounds: row:[0-{max_r}], col:[0-{max_c}])

        VALIDATION - check ENTIRE PATH using CORE RULES from system prompt

        BOUNDS CHECK:
            - All positions must be within row:[0-{max_r}] and col:[0-{max_c}]

        VALIDATION DECISION LOGIC:
        - If path contains ANY diagonal moves → REJECT
        - If path contains ANY wall positions → REJECT
        - If any position is out of bounds → REJECT
        - Zero-moves (same position twice) are VALID
        - Otherwise, approve

        Respond with JSON: {{"valid": true/false, "reason": "specific_reason_with_details"}}
        """

        return query

    def _add_validation_reasoning_instructions(self, base_prompt: str) -> str:
        return """
        EFFICIENT VALIDATION REASONING:
        0. Strict adherence to core rules
        1. Focus on safety of ENTIRE PATH
        2. Check physical constraints: walls, boundaries, current occupants
        3. Trust central coordinator for timing-based spacing
        4. Default to approve if no clear violations found, else reject with specifics

        Please think through this step-by-step before providing your JSON response.
        """ + base_prompt

    def _parse_validation_response(self, response: str) -> Dict:
        response = response.strip()

        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            try:
                result = json.loads(json_str)
                if 'valid' not in result:
                    result['valid'] = True
                if 'reason' not in result:
                    result['reason'] = "Basic safety validated"
                if not result.get('valid', True) and 'alternative' not in result:
                    result['alternative'] = {'action': 'wait', 'reason': 'llm_rejection'}
                return result
            except json.JSONDecodeError:
                pass

        return self._create_permissive_fallback()

    def _create_permissive_fallback(self) -> Dict:
        return {
            "valid": True,
            "reason": "Validation response unclear, trusting central coordinator decision",
        }

    def _create_permissive_fallback_with_alternative(self) -> Dict:
        return {
            "valid": False,
            "reason": "Validation failed, suggesting wait as safe alternative",
            "alternative": {"action": "wait", "reason": "validation_error"},
        }

    def _create_safe_fallback(self) -> Dict:
        return {
            "valid": True,
            "reason": "Using permissive fallback, trusting central coordinator",
        }
