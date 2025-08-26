"""
Central LLM Negotiator for Multi-Agent Conflict Resolution
Uses powerful model (glm-4.5-air) for complex reasoning and negotiation
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from ..llm import OpenRouterClient

class CentralNegotiator:
    def __init__(self, model: Optional[str] = None):
        self.client = OpenRouterClient()
        self.model = model or os.getenv('CENTRAL_LLM_MODEL', 'zai/glm-4.5-air:free')
        
    def negotiate_path_conflict(self, conflict_data: Dict) -> Dict:
        """
        Negotiate path conflicts between agents
        
        Args:
            conflict_data: {
                'agents': [{'id': 0, 'current_pos': (x, y), 'target_pos': (x, y), 'planned_path': [(x,y), ...]}, ...],
                'conflict_points': [(x, y), ...],
                'map_state': {...},
                'turn': int
            }
        
        Returns:
            negotiation_result: {
                'resolution': 'priority'/'reroute'/'wait',
                'agent_actions': {agent_id: {'action': 'move'/'wait', 'path': [...], 'priority': int}},
                'reasoning': str
            }
        """
        
        system_prompt = self._create_negotiation_system_prompt()
        user_prompt = self._create_conflict_description(conflict_data)
        
        messages = [
            self.client.create_system_message(system_prompt),
            self.client.create_user_message(user_prompt)
        ]
        
        response = self.client.send_request(
            model=self.model,
            messages=messages,
            max_tokens=1500,
            temperature=0.3  # Lower temperature for more consistent reasoning
        )
        
        if response:
            try:
                return self._parse_negotiation_response(response)
            except Exception as e:
                print(f"Error parsing negotiation response: {e}")
                return self._create_fallback_resolution(conflict_data)
        else:
            return self._create_fallback_resolution(conflict_data)
    
    def _create_negotiation_system_prompt(self) -> str:
        """Create system prompt for negotiation"""
        return """You are a Central Negotiator for a multi-robot warehouse navigation system. Your role is to resolve path conflicts between robots efficiently and fairly.

WAREHOUSE RULES:
- Robots must deliver boxes to targets
- Only one robot can occupy a cell at a time
- Robots can carry one box at a time
- Minimize total time and distance for all robots

CONFLICT RESOLUTION STRATEGIES:
1. PRIORITY: Assign movement priority based on urgency/distance to goal
2. REROUTE: Find alternative paths for conflicting robots
3. WAIT: Have some robots wait while others pass

RESPONSE FORMAT (JSON):
{
    "resolution": "priority|reroute|wait",
    "agent_actions": {
        "agent_id": {
            "action": "move|wait",
            "path": [[x, y], [x, y], ...],
            "priority": 1-10,
            "wait_turns": 0-3
        }
    },
    "reasoning": "Brief explanation of decision"
}

Always respond with valid JSON. Consider efficiency, fairness, and deadlock prevention."""
    
    def _create_conflict_description(self, conflict_data: Dict) -> str:
        """Create human-readable conflict description"""
        description = f"TURN {conflict_data.get('turn', 0)} - PATH CONFLICT DETECTED\n\n"
        
        description += "AGENTS IN CONFLICT:\n"
        for agent in conflict_data.get('agents', []):
            agent_id = agent['id']
            current = agent['current_pos']
            target = agent.get('target_pos', 'unknown')
            path = agent.get('planned_path', [])
            
            description += f"- Agent {agent_id}: At {current}, going to {target}\n"
            description += f"  Planned path: {path}\n"
        
        description += f"\nCONFLICT POINTS: {conflict_data.get('conflict_points', [])}\n"
        
        # Add map context if available
        if 'map_state' in conflict_data:
            description += "\nMAP CONTEXT:\n"
            agents = conflict_data['map_state'].get('agents', {})
            boxes = conflict_data['map_state'].get('boxes', {})
            targets = conflict_data['map_state'].get('targets', {})
            
            description += f"All agents: {agents}\n"
            description += f"Remaining boxes: {boxes}\n"
            description += f"Remaining targets: {targets}\n"
        
        description += "\nPlease provide a negotiation solution in JSON format."
        return description
    
    def _parse_negotiation_response(self, response: str) -> Dict:
        """Parse LLM response into structured format"""
        # Try to extract JSON from response
        response = response.strip()
        
        # Look for JSON in the response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # If JSON parsing fails, create a simple resolution
        return {
            "resolution": "priority",
            "agent_actions": {},
            "reasoning": "Failed to parse LLM response, using default priority resolution"
        }
    
    def _create_fallback_resolution(self, conflict_data: Dict) -> Dict:
        """Create a simple fallback resolution when LLM fails"""
        agents = conflict_data.get('agents', [])
        
        # Simple priority-based resolution
        agent_actions = {}
        for i, agent in enumerate(agents):
            agent_id = agent['id']
            agent_actions[agent_id] = {
                "action": "move" if i == 0 else "wait",
                "path": agent.get('planned_path', []),
                "priority": len(agents) - i,
                "wait_turns": 0 if i == 0 else 1
            }
        
        return {
            "resolution": "priority",
            "agent_actions": agent_actions,
            "reasoning": "Fallback resolution: First agent moves, others wait"
        }
    
    def get_path_guidance(self, agent_data: Dict, map_state: Dict) -> Optional[List[Tuple[int, int]]]:
        """
        Get path guidance from LLM for pathfinding
        
        Args:
            agent_data: {'id': int, 'current_pos': (x, y), 'target_pos': (x, y), 'carrying_box': bool}
            map_state: Full map state
        
        Returns:
            List of positions representing suggested path
        """
        system_prompt = """You are a pathfinding assistant for warehouse robots. Provide efficient paths avoiding obstacles and other robots.

RESPONSE FORMAT (JSON):
{
    "path": [[x, y], [x, y], ...],
    "reasoning": "Brief explanation"
}

Consider: walls (#), other agents (A), boxes (B), targets (T)."""
        
        user_prompt = f"""Find path for Agent {agent_data['id']}:
Current: {agent_data['current_pos']}
Target: {agent_data['target_pos']}
Carrying box: {agent_data.get('carrying_box', False)}

Map state: {map_state}

Provide optimal path as JSON."""
        
        messages = [
            self.client.create_system_message(system_prompt),
            self.client.create_user_message(user_prompt)
        ]
        
        response = self.client.send_request(
            model=self.model,
            messages=messages,
            max_tokens=800,
            temperature=0.2
        )
        
        if response:
            try:
                result = json.loads(response.strip())
                return result.get('path', [])
            except:
                return None
        
        return None
