"""
Central LLM Negotiator for Multi-Agent Conflict Resolution
Uses powerful model for complex reasoning and negotiation with iterative refinement
"""

import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime
from ..llm import OpenRouterClient
from .openrouter_config import OpenRouterConfig

logger = logging.getLogger(__name__)

class CentralNegotiator:
    def __init__(self, model: Optional[str] = None, enable_spatial_hints: bool = True):
        self.client = OpenRouterClient()
        self.model = model or os.getenv('CENTRAL_LLM_MODEL', 'zai/glm-4.5-air:free')
        
        # Use dynamic reasoning detection
        self.is_reasoning_model = OpenRouterConfig.is_reasoning_model(self.model)
        
        # Spatial hints configuration (toggleable for benchmarking)
        self.enable_spatial_hints = enable_spatial_hints
        
        # Refinement loop configuration
        self.max_refinement_iterations = 5
        self.refinement_history = []
        
        if self.enable_spatial_hints:
            print("🎯 Spatial hints ENABLED - LLM will receive wiggle room guidance")
        else:
            print("🚫 Spatial hints DISABLED - Baseline negotiation mode")
        
        print(f"🔄 Refinement loop ENABLED - Max iterations: {self.max_refinement_iterations}")
    
    # Check if model supports reasoning features
    def _is_reasoning_model(self, model: str) -> bool:
        return OpenRouterConfig.is_reasoning_model(model)
    
    # Toggle spatial hints on/off
    def set_spatial_hints(self, enabled: bool):
        self.enable_spatial_hints = enabled
        status = "ENABLED" if enabled else "DISABLED"
    
    # Main negotiation method, get plan from central LLM, validate, refine 5 times if needed
    def negotiate_path_conflict(
        self, 
        conflict_data: Dict, 
        agent_validators: Optional[Dict[int, Callable]] = None
        ) -> Tuple[Dict, List[Dict], Dict]:
        
        self.refinement_history = []
        current_plan = None
        rejected_agents = set()
        iteration = 0
        
        # Track prompts for logging
        prompts_data = {
            'system_prompt': '',
            'user_prompt': '',
            'model_used': self.model
        }
        
        logger.info(f"Starting conflict negotiation for agents: {[a.get('id') for a in conflict_data.get('agents', [])]}")
        print(f"🎯 Starting conflict negotiation (max {self.max_refinement_iterations} refinement iterations)")
        
        # Check if this is a deadlock situation requiring special handling
        if conflict_data.get('deadlock_breaking', False) or conflict_data.get('conflict_type') in ['deadlock', 'stagnation']:
            print("🔧 DEADLOCK BREAKING MODE: Using specialized resolution")
            result = self._create_deadlock_breaking_resolution(conflict_data)
            return result, [], prompts_data
        
        # === STAGE 1: INITIAL NEGOTIATION ===
        logger.debug("Stage 1: Getting initial plan from central LLM")
        print(f"\n📋 STAGE 1: INITIAL NEGOTIATION")
        current_plan, captured_prompts = self._get_initial_central_plan_with_prompts(conflict_data)
        
        # Store captured prompts
        prompts_data['system_prompt'] = captured_prompts.get('system_prompt', '')
        prompts_data['user_prompt'] = captured_prompts.get('user_prompt', '')
        
        self.refinement_history.append({
            "iteration": 0,
            "stage": "initial_negotiation",
            "timestamp": datetime.now().isoformat(),
            "llm_response": current_plan,
            "validation_results": None,
            "rejected_by": [],
            "feedback_provided": None,
            "refined_plan": None
        })
        
        # If no validators provided, return initial plan without refinement
        if not agent_validators:
            logger.debug("No validators provided, returning initial plan")
            return current_plan, self.refinement_history, prompts_data
        
        # === STAGE 2-6: REFINEMENT LOOP (max 5 iterations) ===
        while iteration < self.max_refinement_iterations:
            iteration += 1
            logger.debug(f"Refinement iteration {iteration}/{self.max_refinement_iterations}")
            print(f"\n🔄 STAGE 2: VALIDATION (Iteration {iteration}/{self.max_refinement_iterations})")
            
            # Validate current plan
            validation_results = self._validate_plan(
                current_plan, 
                conflict_data, 
                agent_validators
            )
            
            rejected_agents = {
                agent_id 
                for agent_id, result in validation_results.items() 
                if not result["valid"]
            }
            
            logger.debug(f"Validation complete. Rejected by: {rejected_agents}")
            print(f"   Results: {len(validation_results)} agents, {len(rejected_agents)} rejections")
            
            # Log validation stage
            self.refinement_history.append({
                "iteration": iteration,
                "stage": "validation",
                "timestamp": datetime.now().isoformat(),
                "llm_response": None,
                "validation_results": validation_results,
                "rejected_by": list(rejected_agents),
                "feedback_provided": None,
                "refined_plan": None
            })
            
            # Check if all agents approved (unanimous approval required)
            if not rejected_agents:
                logger.info(f"✓ Plan approved unanimously after iteration {iteration}")
                print(f"   ✅ All agents approved! Plan accepted.")
                return current_plan, self.refinement_history, prompts_data
            
            # If we've exhausted iterations, break before refining again
            if iteration >= self.max_refinement_iterations:
                logger.warning(f"Max refinement iterations ({self.max_refinement_iterations}) reached")
                print(f"   ⚠️  Max iterations reached, attempting final validation")
                break
            
            # === REFINEMENT REQUEST ===
            logger.debug(f"Requesting refinement for rejected agents: {rejected_agents}")
            print(f"   📞 Requesting LLM refinement for {len(rejected_agents)} rejected agent(s)")
            
            feedback_summary = self._build_feedback_summary(
                validation_results, 
                rejected_agents
            )
            
            refined_plan = self._refine_plan(
                current_plan,
                feedback_summary,
                conflict_data
            )
            
            # Log refinement stage
            self.refinement_history.append({
                "iteration": iteration,
                "stage": "refinement",
                "timestamp": datetime.now().isoformat(),
                "llm_response": refined_plan,
                "validation_results": None,
                "rejected_by": list(rejected_agents),
                "feedback_provided": feedback_summary,
                "refined_plan": refined_plan
            })
            
            current_plan = refined_plan
        
        # === STAGE 7: FINAL VALIDATION ===
        logger.debug("Stage 7: Final validation after max iterations")
        print(f"\n📋 STAGE 7: FINAL VALIDATION")
        validation_results = self._validate_plan(
            current_plan, 
            conflict_data, 
            agent_validators
        )
        
        rejected_agents = {
            agent_id 
            for agent_id, result in validation_results.items() 
            if not result["valid"]
        }
        
        self.refinement_history.append({
            "iteration": iteration,
            "stage": "final_validation",
            "timestamp": datetime.now().isoformat(),
            "llm_response": None,
            "validation_results": validation_results,
            "rejected_by": list(rejected_agents),
            "feedback_provided": None,
            "refined_plan": None,
            "final_status": "approved" if not rejected_agents else "deadlock_skipped"
        })
        
        if rejected_agents:
            logger.warning(
                f"✗ Negotiation deadlock: Still rejected by {rejected_agents} "
                f"after {self.max_refinement_iterations} refinement iterations. "
                f"Skipping turn (no movement)."
            )
            print(f"   ❌ DEADLOCK: {len(rejected_agents)} agent(s) still rejecting after refinement")
            print(f"   ⏭️  Skipping this turn (no movement executed)")
            return {}, self.refinement_history, prompts_data
        
        logger.info(f"✓ Plan approved after final validation")
        print(f"   ✅ Final validation passed! Plan accepted.")
        return current_plan, self.refinement_history, prompts_data
    
    # Get initial plan from central LLM
    def _get_initial_central_plan(self, conflict_data: Dict) -> Dict:
        plan, _ = self._get_initial_central_plan_with_prompts(conflict_data)
        return plan
    
    # Get initial plan with prompt capture
    def _get_initial_central_plan_with_prompts(self, conflict_data: Dict) -> Tuple[Dict, Dict]:
        system_prompt = self._create_negotiation_system_prompt()
        user_prompt = self._create_conflict_description(conflict_data)
        
        # Add reasoning-specific instructions if using reasoning model
        if self.is_reasoning_model:
            user_prompt = self._add_reasoning_instructions(user_prompt)
        
        # Capture prompts for logging
        prompts_data = {
            'system_prompt': system_prompt,
            'user_prompt': user_prompt
        }
        
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
                result = self._parse_negotiation_response(response)
                
                # Handle different response formats:
                # 1. If result has 'agent_actions' key, use it
                # 2. If result has agent IDs as top-level keys (0, 1, etc.), use result as is
                # 3. Otherwise return empty
                if 'agent_actions' in result:
                    return result.get('agent_actions', {}), prompts_data
                elif any(str(i) in result for i in range(10)):  # Check for agent IDs as keys
                    return result, prompts_data
                else:
                    return {}, prompts_data
            except Exception as e:
                logger.error(f"Error parsing initial negotiation response: {e}")
                return self._create_fallback_resolution(conflict_data).get('agent_actions', {}), prompts_data
        else:
            logger.error("No response from LLM API")
            return self._create_fallback_resolution(conflict_data).get('agent_actions', {}), prompts_data
    
    # Validate plan with all involved agents
    def _validate_plan(
        self, 
        plan: Dict, 
        conflict_data: Dict,
        agent_validators: Dict[int, Callable]
        ) -> Dict[int, Dict[str, any]]:  # type: ignore

        validation_results = {}
        map_state = conflict_data.get('map_state', {})
        
        for agent_id, validator_func in agent_validators.items():
            agent_id_str = str(agent_id)
            agent_action = plan.get(agent_id_str)
            
            if not agent_action:
                validation_results[agent_id] = {
                    "valid": False,
                    "reason": "No action provided for this agent",
                    "alternative": None
                }
                continue
            
            try:
                # Call validator with correct parameter names: agent_id, proposed_action, current_state (map_state)
                result = validator_func(
                    agent_id=agent_id,
                    proposed_action=agent_action,
                    current_state=map_state  # map_state is the current_state parameter name expected by validator
                )
                
                validation_results[agent_id] = {
                    "valid": result.get("valid", False),
                    "reason": result.get("reason", "Validation failed"),
                    "alternative": result.get("alternative")
                }
            except Exception as e:
                logger.error(f"Validation error for agent {agent_id}: {str(e)}")
                validation_results[agent_id] = {
                    "valid": False,
                    "reason": f"Validation error: {str(e)}",
                    "alternative": None
                }
        
        return validation_results
    
    # Build feedback summary from rejected agents for refinement
    def _build_feedback_summary(
        self, 
        validation_results: Dict[int, Dict[str, any]],  # type: ignore
        rejected_agents: set
        ) -> Dict[str, any]:  # type: ignore

        feedback = {
            "total_rejected": len(rejected_agents),
            "rejection_count": len(rejected_agents),
            "rejections": []
        }
        
        for agent_id in rejected_agents:
            result = validation_results.get(agent_id, {})
            feedback["rejections"].append({
                "agent_id": agent_id,
                "rejection_reason": result.get("reason", "Unknown reason"),
                "suggested_alternative_action": result.get("alternative")
            })
        
        return feedback
    
    # Send refinement request to LLM based on feedback
    def _refine_plan(
        self, 
        current_plan: Dict, 
        feedback_summary: Dict[str, any],  # type: ignore
        conflict_data: Dict
        ) -> Dict:

        refinement_prompt = self._build_refinement_prompt(
            current_plan,
            feedback_summary,
            conflict_data
        )
        
        system_prompt = self._get_refinement_system_prompt()
        
        logger.debug("Sending refinement request to LLM")
        
        try:
            response = self.client.send_request(
                model=self.model,
                messages=[
                    self.client.create_system_message(system_prompt),
                    self.client.create_user_message(refinement_prompt)
                ],
                max_tokens=20000,
                temperature=0.7
            )
            
            if response:
                refined_response = self._parse_negotiation_response(response)
                
                # Handle different response formats (same as initial plan):
                # 1. If response has 'agent_actions' key, use it
                # 2. If response has agent IDs as top-level keys (0, 1, etc.), use response as is
                # 3. Otherwise return current plan as fallback
                if 'agent_actions' in refined_response:
                    refined_plan = refined_response.get("agent_actions", current_plan)
                elif any(str(i) in refined_response for i in range(10)):  # Check for agent IDs as keys
                    refined_plan = refined_response
                else:
                    refined_plan = current_plan
                
                logger.debug(f"Refinement response received: {len(refined_plan)} agents")
                return refined_plan
            else:
                return current_plan
            
        except Exception as e:
            logger.error(f"Error during plan refinement: {str(e)}")
            return current_plan
    
    # Build refinement prompt with detailed feedback
    def _build_refinement_prompt(
        self, 
        current_plan: Dict, 
        feedback_summary: Dict[str, any],  # type: ignore
        conflict_data: Dict
        ) -> str:
        
        # Format rejections with full details
        rejections_text = ""
        for rejection in feedback_summary["rejections"]:
            agent_id = rejection["agent_id"]
            reason = rejection["rejection_reason"]
            alternative = rejection["suggested_alternative_action"]
            
            rejections_text += f"\n- Agent {agent_id}:\n"
            rejections_text += f"  Reason: {reason}\n"
            
            if alternative:
                rejections_text += f"  Suggested alternative: {json.dumps(alternative, indent=4)}\n"
        
        # Format current plan for clarity
        plan_text = json.dumps(current_plan, indent=2)
        
        # Extract agent info from conflict data
        agents_info = ""
        if "agents" in conflict_data:
            for agent in conflict_data["agents"]:
                agent_id = agent.get("id")
                current_pos = agent.get("current_pos")
                target_pos = agent.get("target_pos")
                planned_path = agent.get("planned_path")
                agents_info += (
                    f"\nAgent {agent_id}:\n"
                    f"  Current Position: {current_pos}\n"
                    f"  Target: {target_pos}\n"
                    f"  Original Planned Path: {planned_path}\n"
                )
                
                # Add failure history context
                failed_history = agent.get('failed_move_history', [])
                if failed_history:
                    agents_info += f"  Recent Move Failures:\n"
                    for failure in failed_history[-3:]:  # Last 3 failures
                        turn = failure.get('turn', '?')
                        attempted = failure.get('attempted_move', '?')
                        reason = failure.get('failure_reason', 'unknown')
                        agents_info += f"    • Turn {turn}: {attempted} - {reason}\n"
        
        prompt = f"""REFINEMENT REQUEST - Your previous conflict resolution plan was rejected.

        REJECTION SUMMARY:
        Total rejected: {feedback_summary['total_rejected']} agent(s)

        DETAILED REJECTIONS:{rejections_text}

        AGENT STATUS:{agents_info}

        PREVIOUSLY PROPOSED PLAN (REJECTED):
        {plan_text}

        CONFLICT INFORMATION:
        {json.dumps(conflict_data, indent=2)}

        REFINEMENT TASK:
        Please provide a refined conflict resolution plan that addresses the validator feedback.

        REQUIREMENTS:
        1. Analyze each rejection reason carefully
        2. Consider suggested alternatives as guidance (not mandatory)
        3. Ensure all moves are orthogonally adjacent (up, down, left, right only)
        4. No two agents can occupy the same cell at any point
        5. Provide full paths for all agents

        RESPONSE FORMAT:
        {{
            "resolution": "reroute|priority|wait",
            "agent_actions": {{
                "0": {{"action": "move|wait", "path": [[x,y], [x,y], ...], "priority": 1}},
                "1": {{"action": "move|wait", "path": [[x,y], [x,y], ...], "priority": 2}}
            }},
            "reasoning": "Detailed explanation of how rejections were addressed"
        }}

        Return ONLY valid JSON, no additional text.
        """
        
        return prompt
    
    def _get_refinement_system_prompt(self) -> str:
        return """You are an expert robot conflict resolver specializing in plan refinement.

        Your task is to refine a previously rejected multi-agent conflict resolution plan based on specific validator feedback.

        CORE RULES:
        1. Only orthogonal adjacent moves (up, down, left, right)
        2. No two agents can occupy the same cell simultaneously
        3. Each agent must have a complete path from current position to target
        4. Provide full paths in all refined actions
        5. Prioritize addressing the specific rejection reasons provided

        REFINEMENT STRATEGY:
        - Analyze each rejection reason in depth
        - Identify the root cause of validation failures
        - Propose alternative routing or timing strategies
        - Ensure the refined plan is fundamentally different from rejected version
        - Maintain safety constraints: no collisions, valid moves only

        OUTPUT REQUIREMENT:
        Return ONLY valid JSON with no markdown formatting or text outside the JSON structure.
        """
    
    # Get refinement history from last negotiation
    def get_refinement_history(self) -> List[Dict]:
        return self.refinement_history
    
    # Create system prompt for negotiation
    def _create_negotiation_system_prompt(self) -> str:
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
           - IMPORTANT: Avoid positions that previously caused failures
        
        3. "wait": Conservative pause
           - All agents pause movement
           - Use for complex deadlocks

        ANALYZING FAILURE HISTORY:
        - When agents have recent move failures, analyze the failure reasons
        - wall_collision: Agent tried to move into a wall - avoid that position
        - agent_collision: Agent collided with another agent - coordinate timing
        - out_of_bounds: Move exceeded map boundaries - stay within bounds
        - not_adjacent: Move was diagonal or too far - ensure orthogonal moves only
        - Use failure patterns to inform better path planning

        RESPONSE FORMAT:
        {
            "resolution": "priority|reroute|wait",
            "agent_actions": {
                "0": {"action": "move|wait", "path": [[x,y]...], "priority": 1}
            },
            "reasoning": "Brief explanation of chosen strategy"
        }

        PREFER rerouting solutions when empty spaces are available. Analyze the map layout to find creative positioning opportunities!
        """
    
    # Create conflict description for user prompt
    def _create_conflict_description(self, conflict_data: Dict) -> str:
        description = f"TURN {conflict_data.get('turn', 0)} - PATH CONFLICT DETECTED\n\n"
        
        description += "AGENTS IN CONFLICT:\n"
        for agent in conflict_data.get('agents', []):
            agent_id = agent['id']
            current = agent['current_pos']
            target = agent.get('target_pos', 'unknown')
            path = agent.get('planned_path', [])
            
            description += f"- Agent {agent_id}: At {current}, going to {target}\n"
            description += f"  Planned path: {path}\n"
            
            # Add failure history if available
            failed_history = agent.get('failed_move_history', [])
            if failed_history:
                description += f"  Recent move failures ({len(failed_history)}):\n"
                for failure in failed_history[-3:]:  # Show last 3 failures
                    turn = failure.get('turn', '?')
                    attempted = failure.get('attempted_move', '?')
                    reason = failure.get('failure_reason', 'unknown')
                    description += f"    • Turn {turn}: Tried to move to {attempted} - Failed: {reason}\n"
        
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
    
    # Add reasoning instructions for reasoning-capable models
    def _add_reasoning_instructions(self, base_prompt: str) -> str:
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
    
    # Parse LLM response with truncation handling
    def _parse_negotiation_response(self, response: str) -> Dict:
        # Try to extract JSON from response
        response = response.strip()
        
        # Look for JSON in the response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError as e:
                # Try to recover from truncated JSON
                recovered_json = self._attempt_json_recovery(json_str)
                if recovered_json:
                    return recovered_json
        
        # If all parsing fails, create a simple resolution
        return {
            "resolution": "priority",
            "agent_actions": {},
            "reasoning": "Failed to parse LLM response, using default priority resolution"
        }
    
    # Attempt to recover from truncated JSON
    def _attempt_json_recovery(self, json_str: str) -> Optional[Dict]:
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
    
    # Create resolution specifically for deadlock breaking
    def _create_deadlock_breaking_resolution(self, conflict_data: Dict) -> Dict:
        agents = conflict_data.get('agents', [])
        
        print("🔧 Creating deadlock-breaking resolution")
        
        # Strategy: Force one agent to wait, others to try alternative moves
        agent_actions = {}
        
        for i, agent in enumerate(agents):
            agent_id = agent['id']
            current_pos = agent.get('current_pos', [0, 0])
            
            if i == 0:
                # First agent gets priority - try original move
                planned_path = agent.get('planned_path', [])
                if len(planned_path) > 1:
                    agent_actions[agent_id] = {
                        "action": "move",
                        "path": planned_path[:3],  # Only next few steps
                        "priority": 1
                    }
                else:
                    agent_actions[agent_id] = {
                        "action": "wait",
                        "wait_turns": 1,
                        "priority": 1
                    }
            else:
                # Other agents try to step aside or wait
                alternative_pos = self._find_safe_step_aside(current_pos, conflict_data)
                
                if alternative_pos:
                    # Step aside temporarily
                    agent_actions[agent_id] = {
                        "action": "move", 
                        "path": [current_pos, alternative_pos, current_pos],  # Step aside and back
                        "priority": 2
                    }
                else:
                    # Wait if no safe step-aside available
                    agent_actions[agent_id] = {
                        "action": "wait",
                        "wait_turns": 2,
                        "priority": 3
                    }
        
        return {
            "resolution": "deadlock_breaking",
            "agent_actions": agent_actions,
            "reasoning": "Deadlock detected - forcing priority movement and step-aside maneuvers to break the deadlock"
        }
    
    # Find safe adjacent position to step aside
    def _find_safe_step_aside(self, current_pos: List[int], conflict_data: Dict) -> Optional[List[int]]:
        x, y = current_pos
        adjacent_positions = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        
        map_state = conflict_data.get('map_state', {})
        grid = map_state.get('grid', [])
        
        for adj_x, adj_y in adjacent_positions:
            if self._is_valid_coordinate(adj_x, adj_y, grid):
                # Check if it's free of other agents
                agents = map_state.get('agents', {})
                if not any(pos == (adj_x, adj_y) for pos in agents.values()):
                    return [adj_x, adj_y]
        
        return None
    
    # Create simple fallback resolution
    def _create_fallback_resolution(self, conflict_data: Dict) -> Dict:
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
    
    # Get path guidance from LLM for pathfinding
    def get_path_guidance(self, agent_data: Dict, map_state: Dict) -> Optional[List[Tuple[int, int]]]:
        system_prompt = """You are a pathfinding assistant for warehouse robots. Provide efficient paths avoiding obstacles and other robots.

        RESPONSE FORMAT (JSON):
        {
            "path": [[x, y], [x, y], ...],
            "reasoning": "Brief explanation"
        }

        Consider: walls (#), other agents (A), boxes (B), targets (T).
        """
        
        user_prompt = f"""Find path for Agent {agent_data['id']}:
        Current: {agent_data['current_pos']}
        Target: {agent_data['target_pos']}
        Carrying box: {agent_data.get('carrying_box', False)}

        Map state: {map_state}

        Provide optimal path as JSON.
        """
        
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
    
    # Validate coordinate within bounds and accessibility
    def _is_valid_coordinate(self, x: int, y: int, grid: List[List[str]]) -> bool:
        if not grid:
            return False
            
        height, width = len(grid), len(grid[0]) if grid else 0
        
        # Check bounds
        if not (0 <= x < width and 0 <= y < height):
            return False
            
        # Check if position is accessible (not a wall)
        cell = grid[y][x]
        return cell == '.'  # Only empty floor is valid for wiggle rooms
    
    # Analyze wiggle rooms in the map for rerouting options
    def _analyze_wiggle_rooms(self, conflict_data: Dict) -> List[Dict]:
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
        
        wiggle_rooms = []
        
        # Find empty cells that are NOT on any planned path
        for y in range(height):
            for x in range(width):
                pos = (x, y)
                
                # Enhanced validation using coordinate validator
                if not self._is_valid_coordinate(x, y, grid):
                    continue
                
                # Must not be occupied by agents or boxes
                if pos in agents.values() or pos in boxes.values():
                    continue
                
                # MUST NOT be on any agent's planned path
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
        
        return wiggle_rooms[:5]  # Return top 5 wiggle rooms
    
    # Evaluate if a position is a good wiggle room
    def _evaluate_wiggle_potential(self, x: int, y: int, grid: List[List[str]], 
                                 conflict_data: Dict, agent_paths: Dict) -> Dict:
        height, width = len(grid), len(grid[0])
        pos = (x, y)
        
        # Count adjacent open spaces (connectivity) with enhanced validation
        adjacent_open = 0
        adjacent_positions = [(x-1,y), (x+1,y), (x,y-1), (x,y+1)]
        
        for ax, ay in adjacent_positions:
            if self._is_valid_coordinate(ax, ay, grid):
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
    
    # Create ASCII map visualization
    def _create_map_visualization(self, grid: List[List[str]], agents: Dict, boxes: Dict, 
                                targets: Dict, wiggle_rooms: List[Dict]) -> str:
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
