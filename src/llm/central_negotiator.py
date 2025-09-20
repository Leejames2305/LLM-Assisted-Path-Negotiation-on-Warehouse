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
    def __init__(self, model: Optional[str] = None, enable_spatial_hints: bool = True):
        self.client = OpenRouterClient()
        self.model = model or os.getenv('CENTRAL_LLM_MODEL', 'zai/glm-4.5-air:free')
        
        # Use dynamic reasoning detection
        self.is_reasoning_model = OpenRouterConfig.is_reasoning_model(self.model)
        
        # Spatial hints configuration (toggleable for benchmarking)
        self.enable_spatial_hints = enable_spatial_hints
        
        if self.enable_spatial_hints:
            print("🎯 Spatial hints ENABLED - LLM will receive wiggle room guidance")
        else:
            print("🚫 Spatial hints DISABLED - Baseline negotiation mode")
        
    def _is_reasoning_model(self, model: str) -> bool:
        """Check if the model supports reasoning features (delegated to config)"""
        return OpenRouterConfig.is_reasoning_model(model)
    
    def set_spatial_hints(self, enabled: bool):
        """Toggle spatial hints on/off for benchmarking"""
        self.enable_spatial_hints = enabled
        status = "ENABLED" if enabled else "DISABLED"
        print(f"🔧 Spatial hints {status}")
        
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
        """Create system prompt for negotiation with rerouting always available"""
        return """You are an expert robot conflict resolver. Respond ONLY with valid JSON.

        RULES: Robots deliver boxes, one per cell, avoid collisions.

        RESOLUTION STRATEGIES:
        1. "priority": Assign movement priorities
           - Higher priority agents move first
           - Others wait in current position
        
        2. "reroute": Use empty spaces for temporary positioning
           - Move agents to strategic waiting positions
           - Create paths that use available space efficiently
           - Consider "stepping aside" into empty cells
           - Look for alternative routes around conflicts
        
        3. "wait": Conservative pause
           - All agents pause movement
           - Use for complex deadlocks

        RESPONSE FORMAT:
        {
            "resolution": "priority|reroute|wait",
            "agent_actions": {
                "0": {"action": "move|wait", "path": [[x,y]...], "priority": 1}
            },
            "reasoning": "Brief explanation of chosen strategy"
        }

        PREFER rerouting solutions when empty spaces are available. Analyze the map layout to find creative positioning opportunities!"""
    
    def _create_conflict_description(self, conflict_data: Dict) -> str:
        """Create human-readable conflict description with optional spatial hints"""
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
            
            # Add spatial hints only if enabled
            if self.enable_spatial_hints:
                wiggle_rooms = self._analyze_wiggle_rooms(conflict_data)
                if wiggle_rooms:
                    description += "\n🎯 STRATEGIC REROUTING OPTIONS (Wiggle Rooms):\n"
                    for wr in wiggle_rooms:
                        pos = wr['position']
                        wr_type = wr['type']
                        value = wr['strategic_value']
                        nearby = wr['nearby_agents']
                        
                        description += f"- Position {pos}: {wr_type} (value: {value})\n"
                        if nearby:
                            description += f"  └─ Near agents: {nearby}\n"
                    
                    description += "\n💡 REROUTING STRATEGIES:\n"
                    description += "- 'reroute': Use wiggle rooms for temporary waiting/bypassing\n"
                    description += "- 'priority': Assign movement priority + others wait in place\n"
                    description += "- 'wait': All agents pause (conservative approach)\n"
            
            # Add map visualization (always show, but wiggle rooms only if hints enabled)
            grid = conflict_data['map_state'].get('grid', [])
            if grid:
                description += "\nMAP LAYOUT:\n"
                wiggle_rooms = self._analyze_wiggle_rooms(conflict_data) if self.enable_spatial_hints else []
                description += self._create_map_visualization(grid, agents, boxes, targets, wiggle_rooms)
        
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
    
    def _analyze_wiggle_rooms(self, conflict_data: Dict) -> List[Dict]:
        """Identify potential wiggle rooms/waiting areas for rerouting"""
        # Only analyze if spatial hints are enabled
        if not self.enable_spatial_hints:
            return []
            
        map_state = conflict_data.get('map_state', {})
        grid = map_state.get('grid', [])
        agents = map_state.get('agents', {})
        boxes = map_state.get('boxes', {})
        
        if not grid:
            return []
        
        height, width = len(grid), len(grid[0]) if grid else 0
        
        # Get all planned paths to avoid overlap
        all_planned_positions = set()
        agent_paths = {}
        for agent in conflict_data.get('agents', []):
            agent_id = agent.get('id')
            path = agent.get('planned_path', [])
            agent_paths[agent_id] = path
            # Add all positions in the path (except current position)
            for pos in path[1:]:  # Skip current position
                if isinstance(pos, (list, tuple)) and len(pos) == 2:
                    all_planned_positions.add(tuple(pos))
        
        print(f"🔍 DEBUG: All planned path positions to avoid: {all_planned_positions}")
        print(f"🔍 DEBUG: Agent current positions: {agents}")
        print(f"🔍 DEBUG: Box positions: {boxes}")
        
        wiggle_rooms = []
        
        # Find empty cells that are NOT on any planned path
        for y in range(height):
            for x in range(width):
                pos = (x, y)
                
                # Must be empty floor
                if grid[y][x] != '.':
                    continue
                
                # Must not be occupied by agents or boxes
                if pos in agents.values() or pos in boxes.values():
                    continue
                
                # CRITICAL: Must not be on any agent's planned path
                if pos in all_planned_positions:
                    continue
                
                # Check if this cell is a potential wiggle room
                wiggle_info = self._evaluate_wiggle_potential(x, y, grid, conflict_data, agent_paths)
                if wiggle_info['is_wiggle_room']:
                    wiggle_rooms.append({
                        'position': pos,
                        'type': wiggle_info['type'],
                        'strategic_value': wiggle_info['strategic_value'],
                        'nearby_agents': wiggle_info['nearby_agents'],
                        'distance_to_conflict': wiggle_info['distance_to_conflict'],
                        'connectivity': wiggle_info['connectivity']
                    })
        
        # Sort by strategic value
        wiggle_rooms.sort(key=lambda w: w['strategic_value'], reverse=True)
        
        print(f"🎯 DEBUG: Found {len(wiggle_rooms)} potential wiggle rooms:")
        for wr in wiggle_rooms:
            print(f"   {wr['position']}: {wr['type']} (value: {wr['strategic_value']}, connectivity: {wr['connectivity']})")
        
        return wiggle_rooms[:5]  # Return top 5 wiggle rooms
    
    def _evaluate_wiggle_potential(self, x: int, y: int, grid: List[List[str]], 
                                 conflict_data: Dict, agent_paths: Dict) -> Dict:
        """Evaluate if a position is a good wiggle room"""
        height, width = len(grid), len(grid[0])
        pos = (x, y)
        
        # Count adjacent open spaces (connectivity)
        adjacent_open = 0
        adjacent_positions = [(x-1,y), (x+1,y), (x,y-1), (x,y+1)]
        
        for ax, ay in adjacent_positions:
            if 0 <= ax < width and 0 <= ay < height and grid[ay][ax] == '.':
                adjacent_open += 1
        
        # Get conflict points and calculate minimum distance
        conflict_points = conflict_data.get('conflict_points', [])
        min_conflict_distance = float('inf')
        
        if conflict_points:
            for cx, cy in conflict_points:
                distance = abs(x - cx) + abs(y - cy)  # Manhattan distance
                min_conflict_distance = min(min_conflict_distance, distance)
        else:
            # If no specific conflict points, use agent positions
            agents = conflict_data.get('agents', [])
            for agent in agents:
                current_pos = agent.get('current_pos', (0, 0))
                if isinstance(current_pos, (list, tuple)) and len(current_pos) == 2:
                    cx, cy = current_pos
                    distance = abs(x - cx) + abs(y - cy)
                    min_conflict_distance = min(min_conflict_distance, distance)
        
        # Calculate distance to any agent's path (proximity without overlap)
        min_path_distance = float('inf')
        nearby_agents = []
        
        for agent_id, path in agent_paths.items():
            path_distances = []
            for px, py in path:
                if isinstance((px, py), (tuple, list)) and len((px, py)) == 2:
                    distance = abs(x - px) + abs(y - py)
                    path_distances.append(distance)
            
            if path_distances:
                min_agent_path_distance = min(path_distances)
                min_path_distance = min(min_path_distance, min_agent_path_distance)
                
                # Agent is "nearby" if wiggle room is within 3 cells of their path
                if min_agent_path_distance <= 3:
                    nearby_agents.append(agent_id)
        
        # Determine wiggle room type and strategic value
        is_wiggle_room = False
        wiggle_type = "none"
        strategic_value = 0
        
        # Must have at least 2 adjacent open spaces to be useful
        if adjacent_open >= 2:
            is_wiggle_room = True
            
            # Higher connectivity is better
            connectivity_bonus = adjacent_open
            
            # Distance scoring: nearby but not too close
            if 2 <= min_conflict_distance <= 4:
                # Perfect range - close enough to be useful, far enough to be safe
                distance_score = 15 - min_conflict_distance
                wiggle_type = "strategic_bypass"
            elif 1 <= min_conflict_distance <= 1:
                # Very close to conflict - risky but might be useful
                distance_score = 8
                wiggle_type = "emergency_dodge"
            elif 5 <= min_conflict_distance <= 8:
                # Further away - good for longer-term repositioning
                distance_score = 12 - min_conflict_distance
                wiggle_type = "staging_area"
            else:
                # Too far away to be immediately useful
                distance_score = 3
                wiggle_type = "distant_refuge"
            
            # Bonus for being near multiple agents (coordination potential)
            multi_agent_bonus = len(nearby_agents) * 2
            
            # Special bonus for corner positions (often good for stepping aside)
            corner_bonus = 0
            if adjacent_open == 2:
                # Check if it's a corner (L-shaped connectivity)
                horizontal_open = (0 <= x-1 < width and grid[y][x-1] == '.') + (0 <= x+1 < width and grid[y][x+1] == '.')
                vertical_open = (0 <= y-1 < height and grid[y-1][x] == '.') + (0 <= y+1 < height and grid[y+1][x] == '.')
                if horizontal_open == 1 and vertical_open == 1:
                    corner_bonus = 3
                    wiggle_type = "corner_refuge"
            
            strategic_value = distance_score + connectivity_bonus + multi_agent_bonus + corner_bonus
        
        return {
            'is_wiggle_room': is_wiggle_room,
            'type': wiggle_type,
            'strategic_value': strategic_value,
            'nearby_agents': nearby_agents,
            'distance_to_conflict': min_conflict_distance,
            'connectivity': adjacent_open
        }
    
    def _create_map_visualization(self, grid: List[List[str]], agents: Dict, boxes: Dict, 
                                targets: Dict, wiggle_rooms: List[Dict]) -> str:
        """Create ASCII map with current positions and optionally wiggle rooms highlighted"""
        if not grid:
            return "No grid available\n"
        
        height, width = len(grid), len(grid[0])
        viz_grid = [row.copy() for row in grid]
        
        # Mark agent positions
        for agent_id, (x, y) in agents.items():
            if 0 <= y < height and 0 <= x < width:
                viz_grid[y][x] = f'A{agent_id}'
        
        # Mark box positions  
        for box_id, (x, y) in boxes.items():
            if 0 <= y < height and 0 <= x < width:
                viz_grid[y][x] = f'B{box_id}'
        
        # Mark target positions
        for target_id, (x, y) in targets.items():
            if 0 <= y < height and 0 <= x < width:
                if viz_grid[y][x] == '.':  # Only if empty
                    viz_grid[y][x] = f'T{target_id}'
        
        # Mark wiggle rooms (only if spatial hints enabled)
        if self.enable_spatial_hints and wiggle_rooms:
            for wr in wiggle_rooms[:3]:  # Show top 3 wiggle rooms
                x, y = wr['position']
                if 0 <= y < height and 0 <= x < width and viz_grid[y][x] == '.':
                    viz_grid[y][x] = 'W'  # Wiggle room marker
        
        # Convert to string
        map_str = ""
        for row in viz_grid:
            map_str += " ".join(f"{cell:>2}" for cell in row) + "\n"
        
        # Legend changes based on whether spatial hints are enabled
        legend = "\nLEGEND: # = Wall, . = Empty, A# = Agent, B# = Box, T# = Target"
        if self.enable_spatial_hints and wiggle_rooms:
            legend += ", W = Wiggle Room"
        legend += "\n"
        
        map_str += legend
        return map_str
