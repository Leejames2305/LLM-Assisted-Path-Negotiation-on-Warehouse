"""
Central LLM Negotiator for Multi-Agent Conflict Resolution (POGEMA port).
Uses powerful model for complex reasoning and negotiation with iterative refinement.
Adapted from src/llm/central_negotiator.py — box references removed, (row,col) coordinates used.
"""

import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime
from .openrouter_client import OpenRouterClient
from .openrouter_config import OpenRouterConfig

logger = logging.getLogger(__name__)


class CentralNegotiator:
    def __init__(self, model: Optional[str] = None, enable_spatial_hints: bool = True):
        self.client = OpenRouterClient()
        self.model = model or os.getenv('CENTRAL_LLM_MODEL', 'zai/glm-4.5-air:free')

        self.is_reasoning_model = OpenRouterConfig.is_reasoning_model(self.model)
        self.enable_spatial_hints = enable_spatial_hints
        self.max_refinement_iterations = 5
        self.refinement_history = []

        if self.enable_spatial_hints:
            print("🎯 Spatial hints ENABLED - LLM will receive wiggle room guidance")
        else:
            print("🚫 Spatial hints DISABLED - Baseline negotiation mode")

        print(f"🔄 Refinement loop ENABLED - Max iterations: {self.max_refinement_iterations}")

    def _is_reasoning_model(self, model: str) -> bool:
        return OpenRouterConfig.is_reasoning_model(model)

    def set_spatial_hints(self, enabled: bool):
        self.enable_spatial_hints = enabled

    # Main negotiation entry point
    def negotiate_path_conflict(
        self,
        conflict_data: Dict,
        agent_validators: Optional[Dict[int, Callable]] = None,
    ) -> Tuple[Dict, List[Dict], Dict]:

        self.refinement_history = []
        current_plan = None
        rejected_agents = set()
        iteration = 0

        prompts_data = {
            'system_prompt': '',
            'user_prompt': '',
            'model_used': self.model,
        }

        logger.info(f"Starting conflict negotiation for agents: {[a.get('id') for a in conflict_data.get('agents', [])]}")
        print(f"🎯 Starting conflict negotiation (max {self.max_refinement_iterations} refinement iterations)")

        if conflict_data.get('deadlock_breaking', False) or conflict_data.get('conflict_type') in ['deadlock', 'stagnation']:
            print("🔧 DEADLOCK BREAKING MODE: Using specialized resolution")
            result = self._create_deadlock_breaking_resolution(conflict_data)
            return result, [], prompts_data

        # === STAGE 1: INITIAL NEGOTIATION ===
        print(f"\n📋 STAGE 1: INITIAL NEGOTIATION")
        current_plan, captured_prompts = self._get_initial_central_plan_with_prompts(conflict_data)

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
            "refined_plan": None,
        })

        if not agent_validators:
            return current_plan, self.refinement_history, prompts_data

        # === REFINEMENT LOOP ===
        while iteration < self.max_refinement_iterations:
            iteration += 1
            print(f"\n🔄 STAGE 2: VALIDATION (Iteration {iteration}/{self.max_refinement_iterations})")

            validation_results = self._validate_plan(current_plan, conflict_data, agent_validators)
            rejected_agents = {
                agent_id
                for agent_id, result in validation_results.items()
                if not result["valid"]
            }

            print(f"   Results: {len(validation_results)} agents, {len(rejected_agents)} rejections")

            self.refinement_history.append({
                "iteration": iteration,
                "stage": "validation",
                "timestamp": datetime.now().isoformat(),
                "llm_response": None,
                "validation_results": validation_results,
                "rejected_by": list(rejected_agents),
                "feedback_provided": None,
                "refined_plan": None,
            })

            if not rejected_agents:
                print(f"   ✅ All agents approved! Plan accepted.")
                return current_plan, self.refinement_history, prompts_data

            if iteration >= self.max_refinement_iterations:
                print(f"   ⚠️  Max iterations reached, attempting final validation")
                break

            print(f"   📞 Requesting LLM refinement for {len(rejected_agents)} rejected agent(s)")

            feedback_summary = self._build_feedback_summary(validation_results, rejected_agents)
            refined_plan = self._refine_plan(current_plan, feedback_summary, conflict_data)

            self.refinement_history.append({
                "iteration": iteration,
                "stage": "refinement",
                "timestamp": datetime.now().isoformat(),
                "llm_response": refined_plan,
                "validation_results": None,
                "rejected_by": list(rejected_agents),
                "feedback_provided": feedback_summary,
                "refined_plan": refined_plan,
            })

            current_plan = refined_plan

        # === FINAL VALIDATION ===
        print(f"\n📋 STAGE 7: FINAL VALIDATION")
        validation_results = self._validate_plan(current_plan, conflict_data, agent_validators)
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
            "final_status": "approved" if not rejected_agents else "deadlock_skipped",
        })

        if rejected_agents:
            print(f"   ❌ DEADLOCK: {len(rejected_agents)} agent(s) still rejecting after refinement")
            print(f"   ⏭️  Skipping this turn (no movement executed)")
            return {}, self.refinement_history, prompts_data

        print(f"   ✅ Final validation passed! Plan accepted.")
        return current_plan, self.refinement_history, prompts_data

    def _get_initial_central_plan(self, conflict_data: Dict) -> Dict:
        plan, _ = self._get_initial_central_plan_with_prompts(conflict_data)
        return plan

    def _get_initial_central_plan_with_prompts(self, conflict_data: Dict) -> Tuple[Dict, Dict]:
        system_prompt = self._create_negotiation_system_prompt()
        user_prompt = self._create_conflict_description(conflict_data)

        if self.is_reasoning_model:
            user_prompt = self._add_reasoning_instructions(user_prompt)

        prompts_data = {
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
        }

        messages = [
            self.client.create_system_message(system_prompt),
            self.client.create_user_message(user_prompt),
        ]

        max_tokens = 50000
        temperature = 0.2 if self.is_reasoning_model else 0.3

        response = self.client.send_request(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if response:
            try:
                result = self._parse_negotiation_response(response)  # type: ignore
                if 'agent_actions' in result:
                    return result.get('agent_actions', {}), prompts_data
                elif any(str(i) in result for i in range(10)):
                    return result, prompts_data
                else:
                    return {}, prompts_data
            except Exception as e:
                logger.error(f"Error parsing initial negotiation response: {e}")
                return self._create_fallback_resolution(conflict_data).get('agent_actions', {}), prompts_data
        else:
            logger.error("No response from LLM API")
            return self._create_fallback_resolution(conflict_data).get('agent_actions', {}), prompts_data

    def _validate_plan(
        self,
        plan: Dict,
        conflict_data: Dict,
        agent_validators: Dict[int, Callable],
    ) -> Dict[int, Dict]:

        validation_results = {}
        map_state = conflict_data.get('map_state', {})

        for agent_id, validator_func in agent_validators.items():
            agent_id_str = str(agent_id)
            agent_action = plan.get(agent_id_str)

            if not agent_action:
                validation_results[agent_id] = {
                    "valid": False,
                    "reason": "No action provided for this agent",
                    "alternative": None,
                }
                continue

            try:
                result = validator_func(
                    agent_id=agent_id,
                    proposed_action=agent_action,
                    current_state=map_state,
                )
                validation_results[agent_id] = {
                    "valid": result.get("valid", False),
                    "reason": result.get("reason", "Validation failed"),
                    "alternative": result.get("alternative"),
                }
            except Exception as e:
                logger.error(f"Validation error for agent {agent_id}: {str(e)}")
                validation_results[agent_id] = {
                    "valid": False,
                    "reason": f"Validation error: {str(e)}",
                    "alternative": None,
                }

        return validation_results

    def _build_feedback_summary(
        self,
        validation_results: Dict[int, Dict],
        rejected_agents: set,
    ) -> Dict:

        feedback = {
            "total_rejected": len(rejected_agents),
            "rejection_count": len(rejected_agents),
            "rejections": [],
        }

        for agent_id in rejected_agents:
            result = validation_results.get(agent_id, {})
            feedback["rejections"].append({
                "agent_id": agent_id,
                "rejection_reason": result.get("reason", "Unknown reason"),
                "suggested_alternative_action": result.get("alternative"),
            })

        return feedback

    def _refine_plan(
        self,
        current_plan: Dict,
        feedback_summary: Dict,
        conflict_data: Dict,
    ) -> Dict:

        refinement_prompt = self._build_refinement_prompt(current_plan, feedback_summary, conflict_data)
        system_prompt = self._get_refinement_system_prompt()

        try:
            response = self.client.send_request(
                model=self.model,
                messages=[
                    self.client.create_system_message(system_prompt),
                    self.client.create_user_message(refinement_prompt),
                ],
                max_tokens=50000,
                temperature=0.3,
            )

            if response:
                refined_response = self._parse_negotiation_response(response)  # type: ignore
                if 'agent_actions' in refined_response:
                    return refined_response.get("agent_actions", current_plan)
                elif any(str(i) in refined_response for i in range(10)):
                    return refined_response
                else:
                    return current_plan
            else:
                return current_plan

        except Exception as e:
            logger.error(f"Error during plan refinement: {str(e)}")
            return current_plan

    def _build_refinement_prompt(
        self,
        current_plan: Dict,
        feedback_summary: Dict,
        conflict_data: Dict,
    ) -> str:

        rejections_text = ""
        for rejection in feedback_summary["rejections"]:
            agent_id = rejection["agent_id"]
            reason = rejection["rejection_reason"]
            alternative = rejection["suggested_alternative_action"]
            rejections_text += f"\n- Agent {agent_id}:\n"
            rejections_text += f"  Reason: {reason}\n"
            if alternative:
                rejections_text += f"  Suggested alternative: {json.dumps(alternative, indent=4)}\n"

        plan_text = json.dumps(current_plan, indent=2)

        agents_info = ""
        if "agents" in conflict_data:
            for agent in conflict_data["agents"]:
                agent_id = agent.get("id")
                current_pos = agent.get("current_pos")
                target_pos = agent.get("target_pos")
                planned_path = agent.get("planned_path")
                agents_info += (
                    f"\nAgent {agent_id}:\n"
                    f"  Current Position (row, col): {current_pos}\n"
                    f"  Target (row, col): {target_pos}\n"
                    f"  Original Planned Path: {planned_path}\n"
                )
                failed_history = agent.get('failed_move_history', [])
                if failed_history:
                    agents_info += "  Recent Move Failures:\n"
                    for failure in failed_history[-3:]:
                        turn = failure.get('turn', '?')
                        attempted = failure.get('attempted_move', '?')
                        reason = failure.get('failure_reason', 'unknown')
                        agents_info += f"    • Step {turn}: {attempted} - {reason}\n"

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
        6. Use (row, col) coordinate format for all positions

        RESPONSE FORMAT:
        {{
            "resolution": "reroute|priority|wait",
            "agent_actions": {{
                "0": {{"action": "move|wait", "path": [[row,col], ...], "priority": 1}},
                "1": {{"action": "move|wait", "path": [[row,col], ...], "priority": 2}}
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
        0. Coordinates are (row, col) — origin (0,0) is TOP-LEFT
        1. Only orthogonal adjacent moves (up, down, left, right)
        2. No two agents can occupy the same cell simultaneously
        3. Each agent must have a complete path from current position to target
        4. Provide full paths in all refined actions
        5. Prioritize addressing the specific rejection reasons provided

        REFINEMENT STRATEGY:
        - Analyze each rejection reason in depth
        - Identify the root cause of validation failures
        - Propose alternative routing or timed-waiting strategies
        - Maintain safety constraints: no collisions, valid moves only

        OUTPUT REQUIREMENT:
        Return ONLY valid JSON with no markdown formatting or text outside the JSON structure.

        RESPONSE FORMAT:
        {
            "resolution": "priority|reroute|wait",
            "agent_actions": {
                "0": {"action": "move|wait", "path": [[row,col]...], "priority": 1},
                "1": {"action": "move|wait", "path": [[row,col], ...], "priority": 2}
            },
            "reasoning": "Brief explanation of chosen strategy"
        }
        """

    def get_refinement_history(self) -> List[Dict]:
        return self.refinement_history

    def _create_negotiation_system_prompt(self) -> str:
        return """You are an expert robot conflict resolver. Respond ONLY with valid JSON.

        COORDINATE SYSTEM: All positions use (row, col) format. Origin (0,0) is TOP-LEFT.

        CORE RULES:
        1. Only orthogonal adjacent moves (up, down, left, right)
        2. No two agents can occupy the same cell simultaneously
        3. Each agent must have a complete path from current position to target
        4. Provide full paths in all actions
        5. Prioritize addressing the specific rejection reasons provided

        RESOLUTION STRATEGIES:
        1. "priority": Assign movement priorities
           Use when: Agents have non-overlapping goals but conflicting immediate paths
           How it works:
           - Higher priority agents move first
           - Lower priority agents wait in current position
           - Example: Agent 0 moves through narrow passage first, while Agent 1 waits

        2. "reroute": Use empty spaces for temporary positioning
            Use when: The map has available space for strategic repositioning/waiting
            How it works:
           - Guide agents to temporary safe spots, then resume original path
           - Example: Agent 0 moves to nearby empty cell to let Agent 1 pass, then continues
           - Prefer this strategy when wiggle spaces are available

        3. "wait": Conservative pause
            Use when: Pause the agent due to complex conflicts or lack of space
            How it works:
           - The selected agent waits at original position until next turn

        RESPONSE FORMAT:
        {
            "resolution": "priority|reroute|wait",
            "agent_actions": {
                "0": {"action": "move|wait", "path": [[row,col]...], "priority": 1}
            },
            "reasoning": "Brief explanation of chosen strategy"
        }
        """

    def _create_conflict_description(self, conflict_data: Dict) -> str:
        description = f"STEP {conflict_data.get('turn', 0)} - PATH CONFLICT DETECTED\n\n"

        description += "AGENTS IN CONFLICT:\n"
        for agent in conflict_data.get('agents', []):
            agent_id = agent['id']
            current = agent['current_pos']
            target = agent.get('target_pos', 'unknown')
            path = agent.get('planned_path', [])

            description += f"- Agent {agent_id}: At (row,col)={current}, navigating to (row,col)={target}\n"
            description += f"  Planned path: {path}\n"

            failed_history = agent.get('failed_move_history', [])
            if failed_history:
                description += f"  Recent move failures ({len(failed_history)}):\n"
                for failure in failed_history[-3:]:
                    turn = failure.get('turn', '?')
                    attempted = failure.get('attempted_move', '?')
                    reason = failure.get('failure_reason', 'unknown')
                    description += f"    • Step {turn}: Tried to move to {attempted} - Failed: {reason}\n"

        description += f"\nCONFLICT POINTS: {conflict_data.get('conflict_points', [])}\n"

        if 'map_state' in conflict_data:
            description += "\nMAP CONTEXT:\n"
            agents = conflict_data['map_state'].get('agents', {})
            targets = conflict_data['map_state'].get('targets', {})

            description += f"All agent positions (row,col): {agents}\n"
            description += f"Agent targets (row,col): {targets}\n"

            if self.enable_spatial_hints:
                wiggle_rooms = self._analyze_wiggle_rooms(conflict_data)
                if wiggle_rooms:
                    description += "\n🎯 STRATEGIC REROUTING OPTIONS (Wiggle Rooms):\n"
                    for wr in wiggle_rooms:
                        pos = wr['position']
                        wr_type = wr['type']
                        value = wr['strategic_value']
                        nearby = wr['nearby_agents']
                        description += f"- Position (row,col)={pos}: {wr_type} (value: {value})\n"
                        if nearby:
                            description += f"  └─ Near agents: {nearby}\n"

                    description += "\n💡 REROUTING STRATEGIES:\n"
                    description += "- 'reroute': Use wiggle rooms for temporary waiting/bypassing\n"
                    description += "- 'priority': Assign movement priority + others wait in place\n"
                    description += "- 'wait': Pause the agent due to complex conflicts or lack of space\n"

            grid = conflict_data['map_state'].get('grid', [])
            if grid:
                description += "\nMAP LAYOUT (0=free, 1=wall):\n"
                wiggle_rooms = self._analyze_wiggle_rooms(conflict_data) if self.enable_spatial_hints else []
                description += self._create_map_visualization(grid, agents, targets, wiggle_rooms)

        description += "\nPlease provide a negotiation solution in JSON format."
        return description

    def _add_reasoning_instructions(self, base_prompt: str) -> str:
        reasoning_instructions = """
        REASONING APPROACH:
        0. Strict adherence to core rules
        1. Analyze the spatial configuration and movement constraints
        2. Consider each agent's priority and current objective
        3. Evaluate potential collision points and timing
        4. Reason through the most efficient solution

        Please think through this step-by-step before providing your JSON response.

        """
        return reasoning_instructions + base_prompt

    def _parse_negotiation_response(self, response: str) -> Dict:
        response = response.strip()

        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                recovered = self._attempt_json_recovery(json_str)
                if recovered:
                    return recovered

        return {
            "resolution": "priority",
            "agent_actions": {},
            "reasoning": "Failed to parse LLM response, using default priority resolution",
        }

    def _attempt_json_recovery(self, json_str: str) -> Optional[Dict]:
        try:
            fixed_json = json_str.rstrip()
            if fixed_json.endswith(','):
                fixed_json = fixed_json.rstrip(',') + '}'
            elif not fixed_json.endswith('}'):
                fixed_json += '}'
            if fixed_json.count('"') % 2 != 0:
                fixed_json += '"'
                if not fixed_json.endswith('}'):
                    fixed_json += '}'
            return json.loads(fixed_json)
        except json.JSONDecodeError:
            return None

    def _create_deadlock_breaking_resolution(self, conflict_data: Dict) -> Dict:
        agents = conflict_data.get('agents', [])
        print("🔧 Creating deadlock-breaking resolution")

        agent_actions = {}
        for i, agent in enumerate(agents):
            agent_id = agent['id']
            current_pos = agent.get('current_pos', [0, 0])

            if i == 0:
                planned_path = agent.get('planned_path', [])
                if len(planned_path) > 1:
                    agent_actions[agent_id] = {
                        "action": "move",
                        "path": planned_path[:3],
                        "priority": 1,
                    }
                else:
                    agent_actions[agent_id] = {
                        "action": "wait",
                        "wait_turns": 1,
                        "priority": 1,
                    }
            else:
                alternative_pos = self._find_safe_step_aside(current_pos, conflict_data)
                if alternative_pos:
                    agent_actions[agent_id] = {
                        "action": "move",
                        "path": [current_pos, alternative_pos, current_pos],
                        "priority": 2,
                    }
                else:
                    agent_actions[agent_id] = {
                        "action": "wait",
                        "wait_turns": 2,
                        "priority": 3,
                    }

        return {
            "resolution": "deadlock_breaking",
            "agent_actions": agent_actions,
            "reasoning": "Deadlock detected — forcing priority movement and step-aside maneuvers",
        }

    def _find_safe_step_aside(self, current_pos: List[int], conflict_data: Dict) -> Optional[List[int]]:
        r, c = current_pos
        # Adjacent positions in (row, col)
        adjacent = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]

        map_state = conflict_data.get('map_state', {})
        grid = map_state.get('grid', [])

        for nr, nc in adjacent:
            if self._is_valid_rc(nr, nc, grid):
                agents = map_state.get('agents', {})
                if not any(pos == [nr, nc] or pos == (nr, nc) for pos in agents.values()):
                    return [nr, nc]

        return None

    def _create_fallback_resolution(self, conflict_data: Dict) -> Dict:
        agents = conflict_data.get('agents', [])
        agent_actions = {}
        for i, agent in enumerate(agents):
            agent_id = agent['id']
            agent_actions[agent_id] = {
                "action": "move" if i == 0 else "wait",
                "path": agent.get('planned_path', []),
                "priority": len(agents) - i,
                "wait_turns": 0 if i == 0 else 1,
            }
        return {
            "resolution": "priority",
            "agent_actions": agent_actions,
            "reasoning": "Fallback resolution: First agent moves, others wait",
        }

    def get_path_guidance(self, agent_data: Dict, map_state: Dict) -> Optional[List[Tuple[int, int]]]:
        system_prompt = """You are a pathfinding assistant for warehouse robots. Provide efficient paths avoiding obstacles.
        All positions use (row, col) format. Origin (0,0) is TOP-LEFT.

        RESPONSE FORMAT (JSON):
        {
            "path": [[row, col], [row, col], ...],
            "reasoning": "Brief explanation"
        }
        """

        user_prompt = f"""Find path for Agent {agent_data['id']}:
        Current (row,col): {agent_data['current_pos']}
        Target (row,col): {agent_data['target_pos']}

        Map state: {map_state}

        Provide optimal path as JSON.
        """

        messages = [
            self.client.create_system_message(system_prompt),
            self.client.create_user_message(user_prompt),
        ]

        response = self.client.send_request(
            model=self.model,
            messages=messages,
            max_tokens=50000,
            temperature=0.2,
        )

        if response:
            try:
                result = json.loads(response.strip())  # type: ignore
                return result.get('path', [])
            except Exception:
                return None

        return None

    def _is_valid_rc(self, row: int, col: int, grid: List) -> bool:
        """Check if (row, col) is within bounds and not a wall (POGEMA: 1=wall)."""
        if not grid:
            return False
        height = len(grid)
        width = len(grid[0]) if grid else 0
        if not (0 <= row < height and 0 <= col < width):
            return False
        return grid[row][col] == 0  # POGEMA: 0=free, 1=wall

    def _analyze_wiggle_rooms(self, conflict_data: Dict) -> List[Dict]:
        if not self.enable_spatial_hints:
            return []

        map_state = conflict_data.get('map_state', {})
        grid = map_state.get('grid', [])
        agents = map_state.get('agents', {})

        if not grid:
            return []

        height = len(grid)
        width = len(grid[0]) if grid else 0

        all_planned_positions = set()
        agent_paths = {}
        for agent in conflict_data.get('agents', []):
            agent_id = agent.get('id')
            path = agent.get('planned_path', [])
            agent_paths[agent_id] = path
            for pos in path[1:]:
                if isinstance(pos, (list, tuple)) and len(pos) == 2:
                    all_planned_positions.add(tuple(pos))

        wiggle_rooms = []

        for row in range(height):
            for col in range(width):
                pos = (row, col)

                if not self._is_valid_rc(row, col, grid):
                    continue

                if pos in all_planned_positions:
                    continue

                # Check if occupied by any agent
                if any(
                    (isinstance(p, list) and p == [row, col]) or p == pos
                    for p in agents.values()
                ):
                    continue

                wiggle_info = self._evaluate_wiggle_potential(row, col, grid, conflict_data, agent_paths)
                if wiggle_info['is_wiggle_room']:
                    wiggle_rooms.append({
                        'position': pos,
                        'type': wiggle_info['type'],
                        'strategic_value': wiggle_info['strategic_value'],
                        'nearby_agents': wiggle_info['nearby_agents'],
                        'distance_to_conflict': wiggle_info['distance_to_conflict'],
                        'connectivity': wiggle_info['connectivity'],
                    })

        wiggle_rooms.sort(key=lambda w: w['strategic_value'], reverse=True)
        return wiggle_rooms[:5]

    def _evaluate_wiggle_potential(
        self, row: int, col: int, grid: List, conflict_data: Dict, agent_paths: Dict
    ) -> Dict:
        height = len(grid)
        width = len(grid[0]) if grid else 0

        adjacent_open = sum(
            1
            for nr, nc in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
            if 0 <= nr < height and 0 <= nc < width and grid[nr][nc] == 0
        )

        conflict_points = conflict_data.get('conflict_points', [])
        min_conflict_distance = float('inf')
        if conflict_points:
            for cr, cc in conflict_points:
                min_conflict_distance = min(min_conflict_distance, abs(row - cr) + abs(col - cc))
        else:
            for agent in conflict_data.get('agents', []):
                cp = agent.get('current_pos', (0, 0))
                if isinstance(cp, (list, tuple)) and len(cp) == 2:
                    min_conflict_distance = min(min_conflict_distance, abs(row - cp[0]) + abs(col - cp[1]))

        nearby_agents = []
        min_path_distance = float('inf')
        for agent_id, path in agent_paths.items():
            path_dists = [abs(row - p[0]) + abs(col - p[1]) for p in path if isinstance(p, (list, tuple)) and len(p) == 2]
            if path_dists:
                min_d = min(path_dists)
                min_path_distance = min(min_path_distance, min_d)
                if min_d <= 3:
                    nearby_agents.append(agent_id)

        is_wiggle_room = False
        wiggle_type = "none"
        strategic_value = 0

        if adjacent_open >= 2:
            is_wiggle_room = True
            connectivity_bonus = adjacent_open

            if 2 <= min_conflict_distance <= 4:
                distance_score = 15 - min_conflict_distance
                wiggle_type = "strategic_bypass"
            elif min_conflict_distance == 1:
                distance_score = 8
                wiggle_type = "emergency_dodge"
            elif 5 <= min_conflict_distance <= 8:
                distance_score = 12 - min_conflict_distance
                wiggle_type = "staging_area"
            else:
                distance_score = 3
                wiggle_type = "distant_refuge"

            multi_agent_bonus = len(nearby_agents) * 2
            strategic_value = distance_score + connectivity_bonus + multi_agent_bonus

        return {
            'is_wiggle_room': is_wiggle_room,
            'type': wiggle_type,
            'strategic_value': strategic_value,
            'nearby_agents': nearby_agents,
            'distance_to_conflict': min_conflict_distance,
            'connectivity': adjacent_open,
        }

    def _create_map_visualization(
        self, grid: List, agents: Dict, targets: Dict, wiggle_rooms: List[Dict]
    ) -> str:
        if not grid:
            return "No grid available\n"

        height = len(grid)
        width = len(grid[0]) if grid else 0

        # Build char grid: 0→'.', 1→'#'
        viz = [['#' if grid[r][c] else '.' for c in range(width)] for r in range(height)]

        for agent_id, pos in agents.items():
            r, c = (pos[0], pos[1]) if isinstance(pos, (list, tuple)) else pos
            if 0 <= r < height and 0 <= c < width:
                viz[r][c] = f'A{agent_id}'

        for target_id, pos in targets.items():
            r, c = (pos[0], pos[1]) if isinstance(pos, (list, tuple)) else pos
            if 0 <= r < height and 0 <= c < width and viz[r][c] == '.':
                viz[r][c] = f'T{target_id}'

        if self.enable_spatial_hints and wiggle_rooms:
            for wr in wiggle_rooms[:3]:
                r, c = wr['position']
                if 0 <= r < height and 0 <= c < width and viz[r][c] == '.':
                    viz[r][c] = 'W'

        map_str = ""
        for row in viz:
            map_str += " ".join(f"{cell:>3}" for cell in row) + "\n"

        legend = "\nLEGEND: # = Wall, . = Empty, A# = Agent, T# = Target"
        if self.enable_spatial_hints and wiggle_rooms:
            legend += ", W = Wiggle Room"
        legend += "\n"
        map_str += legend
        return map_str
