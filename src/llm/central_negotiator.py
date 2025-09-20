"""
Central LLM Negotiator for Multi-Agent Conflict Resolution
Uses powerful model for complex reasoning and negotiation
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from ..llm import OpenRouterClient
from .openrouter_config import OpenRouterConfig

class CentralNegotiator:
    def __init__(self, model: Optional[str] = None):
        self.client = OpenRouterClient()
        self.model = model or os.getenv('CENTRAL_LLM_MODEL', 'zai/glm-4.5-air:free')
        
        # Use dynamic reasoning detection
        self.is_reasoning_model = OpenRouterConfig.is_reasoning_model(self.model)
        
    def _is_reasoning_model(self, model: str) -> bool:
        """Check if the model supports reasoning features (delegated to config)"""
        return OpenRouterConfig.is_reasoning_model(model)
        
    def negotiate_path_conflict(self, conflict_data: Dict) -> Dict:
        """
        Negotiate path conflicts between agents with enhanced reasoning
        
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
        
        # Add reasoning-specific instructions if using reasoning model
        if self.is_reasoning_model:
            user_prompt = self._add_reasoning_instructions(user_prompt)
        
        messages = [
            self.client.create_system_message(system_prompt),
            self.client.create_user_message(user_prompt)
        ]
        
        # Adjust parameters for reasoning models
        max_tokens = 30000 if self.is_reasoning_model else 20000
        temperature = 0.1 if self.is_reasoning_model else 0.3
        
        response = self.client.send_request(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        if response:
            try:
                return self._parse_negotiation_response(response)
            except Exception as e:
                print(f"❌ Error parsing negotiation response: {e}")
                print(f"🔍 Raw response: {response}")
                return self._create_fallback_resolution(conflict_data)
        else:
            print("❌ No response from LLM API")
            return self._create_fallback_resolution(conflict_data)
    
    def _create_negotiation_system_prompt(self) -> str:
        """Create system prompt for negotiation"""
        return """You are a robot conflict resolver. Respond ONLY with valid JSON.

RULES: Robots deliver boxes, one per cell, avoid collisions.

RESPONSE FORMAT:
{
    "resolution": "priority|reroute|wait",
    "agent_actions": {
        "0": {"action": "move|wait", "path": [[x,y]...], "priority": 1}
    },
    "reasoning": "Brief explanation"
}

Keep reasoning under 50 words. Always respond with complete JSON."""
    
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
    
    def _add_reasoning_instructions(self, base_prompt: str) -> str:
        """Add specific instructions for reasoning models"""
        reasoning_instructions = """
REASONING APPROACH:
1. Analyze the spatial configuration and movement constraints
2. Consider each agent's priority and current objective
3. Evaluate potential collision points and timing
4. Reason through multiple resolution strategies
5. Select the most efficient solution

Please think through this step-by-step before providing your JSON response.

"""
        return reasoning_instructions + base_prompt
    
    def _parse_negotiation_response(self, response: str) -> Dict:
        """Parse LLM response into structured format with truncation handling"""
        # Try to extract JSON from response
        response = response.strip()
        
        print(f"🔍 DEBUG: Response length: {len(response)} chars")
        print(f"🔍 DEBUG: Response preview: {response[:200]}...")
        
        # Look for JSON in the response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            try:
                result = json.loads(json_str)
                print("✅ Successfully parsed JSON response")
                return result
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                print(f"🔧 Attempting truncation recovery...")
                
                # Try to recover from truncated JSON
                recovered_json = self._attempt_json_recovery(json_str)
                if recovered_json:
                    print("✅ Successfully recovered from truncated JSON!")
                    return recovered_json
                else:
                    print("❌ Could not recover truncated JSON")
        
        # If all parsing fails, create a simple resolution
        print("🔄 Using fallback JSON structure")
        return {
            "resolution": "priority",
            "agent_actions": {},
            "reasoning": "Failed to parse LLM response, using default priority resolution"
        }
    
    def _attempt_json_recovery(self, json_str: str) -> Optional[Dict]:
        """Attempt to recover from truncated JSON"""
        try:
            # Common truncation fixes
            fixed_json = json_str.rstrip()
            
            # If it ends with a comma, remove it and add closing brace
            if fixed_json.endswith(','):
                fixed_json = fixed_json.rstrip(',') + '}'
            
            # If it doesn't end with closing brace, add one
            elif not fixed_json.endswith('}'):
                fixed_json += '}'
            
            # Try to fix incomplete string values
            if fixed_json.count('"') % 2 != 0:
                fixed_json += '"'
                if not fixed_json.endswith('}'):
                    fixed_json += '}'
            
            print(f"🔧 Recovery attempt: {fixed_json[:100]}...")
            return json.loads(fixed_json)
            
        except json.JSONDecodeError:
            return None
    
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
            max_tokens=20000,
            temperature=0.2
        )
        
        if response:
            try:
                result = json.loads(response.strip())
                return result.get('path', [])
            except:
                return None
        
        return None
