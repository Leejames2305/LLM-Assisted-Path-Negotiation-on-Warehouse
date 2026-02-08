---
name: Feature request
about: Improve LLM Prompts in Negotiators and Validators
title: "[REQUEST] Improve LLM Prompts for Better Conflict Resolution and Validation"
labels: enhancement, llm, prompt-engineering
assignees: ''

---

## Describe the feature

Enhance the LLM prompts used in the Central Negotiator and Agent Validators to improve multi-agent conflict resolution accuracy, reduce validation failures, and optimize reasoning efficiency.

## Affected current module & features

**Modules:**
- `src/llm/central_negotiator.py` - Central conflict negotiator
- `src/llm/agent_validator.py` - Agent action validator

**Features impacted:**
- Multi-agent path conflict resolution
- Action validation and safety checks
- Refinement loop efficiency
- Deadlock breaking strategies
- Response parsing reliability
- Reasoning model optimization

## Current Issues with Existing Prompts

### Central Negotiator Issues

1. **Vague Resolution Strategy Descriptions**
   - Current prompts use brief descriptions like "priority", "reroute", "wait"
   - Missing concrete examples of when to use each strategy
   - Insufficient guidance on strategy selection criteria

2. **Incomplete Spatial Context**
   - Map visualization exists but lacks clarity on coordinate systems
   - Wiggle room hints could be more actionable
   - Missing distance/proximity calculations to help LLM reason

3. **Weak Failure History Analysis**
   - Shows recent failures but doesn't provide pattern analysis guidance
   - No explicit instructions on how to learn from failure types
   - Missing correlation between failure types and resolution strategies

4. **Generic Reasoning Instructions**
   - Reasoning model instructions are too generic
   - Doesn't leverage specific reasoning capabilities effectively
   - Could benefit from structured thinking prompts (e.g., "Think step by step")

5. **Refinement Prompt Limitations**
   - Doesn't explicitly ask LLM to explain WHY previous plan failed
   - Missing guidance on generating fundamentally different solutions
   - No encouragement to explore multiple alternatives before settling

### Agent Validator Issues

1. **Overly Verbose Validation Prompt**
   - System prompt is extremely long (200+ lines)
   - Contains redundant information
   - May confuse smaller validation models

2. **Inconsistent Error Message Format**
   - Some errors use technical terms, others use natural language
   - No standardized error code system
   - Makes parsing and debugging difficult

3. **Weak Zero-Move Handling**
   - Zero-moves are mentioned but not prominently featured
   - Could lead to unnecessary rejections
   - Needs clearer emphasis on "staying in place = valid"

4. **Limited Alternative Suggestion Quality**
   - Alternative action prompts are too brief
   - Doesn't provide enough context for good alternatives
   - Missing examples of good vs. bad alternatives

## Proposed Improvements

### 1. Central Negotiator Prompt Enhancements

#### A. Enhanced Resolution Strategy Descriptions

**Current:**
```
RESOLUTION STRATEGIES:
1. "priority": Assign movement priorities
   - Higher priority agents move first
   - Others wait in current position
```

**Proposed:**
```
RESOLUTION STRATEGIES:

1. "priority" - Sequential Movement
   Use when: Agents have non-overlapping goals but conflicting immediate paths
   How it works:
   - Assign priority based on: urgency, distance to goal, blocking potential
   - Higher priority agents complete their moves first
   - Lower priority agents wait in current position
   - Example: Agent A moves through narrow corridor while Agent B waits
   
   Success criteria: Conflict can be resolved by simple time-based sequencing
```

#### B. Structured Reasoning Framework

**Proposed addition to reasoning instructions:**
```
STRUCTURED REASONING PROCESS:

Step 1: ANALYZE THE SITUATION
- What is the exact nature of the conflict? (head-on, crossing paths, bottleneck)
- How many agents are involved? (2-agent vs. multi-agent conflicts)
- What are the spatial constraints? (corridor width, available wiggle rooms)

Step 2: EVALUATE FAILURE HISTORY
- Have these agents failed similar moves recently?
- What were the specific failure reasons? (wall_collision, agent_collision, etc.)
- Are there patterns suggesting a deadlock cycle?

Step 3: CONSIDER RESOLUTION OPTIONS
- Priority: Can sequential movement work? Who should go first?
- Reroute: Are there alternative paths using wiggle rooms?
- Wait: Is the conflict too complex for immediate resolution?

Step 4: VERIFY YOUR SOLUTION
- Do all paths consist of orthogonal moves only?
- Are all positions valid (no walls, within bounds)?
- Does this solution avoid repeating failed patterns?

Step 5: PROVIDE SOLUTION
- Return your JSON response with clear reasoning
```

#### C. Enhanced Failure Pattern Analysis

**Proposed addition:**
```
FAILURE PATTERN RECOGNITION:

When analyzing move failures, identify patterns:

1. wall_collision pattern:
   - Indicates agent doesn't understand map boundaries
   - Solution: Verify grid and provide explicit boundary reminders
   - Avoid: Positions marked as '#' in grid

2. agent_collision pattern:
   - Indicates timing coordination issue
   - Solution: Use priority strategy or reroute one agent
   - Avoid: Having agents target same cell in same timestep

3. diagonal_move pattern:
   - Indicates path planning error
   - Solution: Break diagonal moves into two orthogonal moves
   - Avoid: Moves where both x AND y change simultaneously

4. out_of_bounds pattern:
   - Indicates coordinate system confusion
   - Solution: Remind agent of grid size (0 to max_x, 0 to max_y)
   - Avoid: Any position with negative coords or exceeding limits

If you see repeated failures of same type, use a DIFFERENT strategy.
```

#### D. Improved Map Visualization Context

**Proposed enhancement:**
```
MAP COORDINATE SYSTEM:
- Origin (0,0) is at TOP-LEFT corner
- X increases going RIGHT (horizontal)
- Y increases going DOWN (vertical)
- Grid size: {width}x{height} means X:[0-{width-1}], Y:[0-{height-1}]

SPATIAL ANALYSIS:
For each agent in conflict:
- Agent {id}: Currently at ({x}, {y})
- Manhattan distance to goal: {distance}
- Obstacles in direct path: {count}
- Available adjacent cells: {list of free neighbors}
```

#### E. Refinement Prompt Improvements

**Proposed addition to refinement prompt:**
```
REFINEMENT ANALYSIS REQUIRED:

Before providing a refined plan, analyze:

1. ROOT CAUSE: Why did the previous plan fail validation?
   - Was it a geometric issue (diagonal moves, walls)?
   - Was it a coordination issue (agent collisions)?
   - Was it a boundary issue (out of bounds)?

2. ALTERNATIVE APPROACHES:
   - List 2-3 different resolution strategies you could use
   - For each, briefly explain pros and cons
   - Select the most promising approach

3. DIFFERENTIATION:
   - How is your refined plan fundamentally different from the rejected plan?
   - What specific changes address the validation failures?
   - Have you avoided simply tweaking coordinates without changing strategy?

4. VERIFICATION CHECKLIST:
   ✓ All moves are orthogonal (no diagonals)
   ✓ All positions are within bounds
   ✓ No wall collisions
   ✓ No agent-agent collisions
   ✓ Paths are complete (start to goal)
   ✓ Solution doesn't repeat failed patterns

Then provide your refined JSON response.
```

### 2. Agent Validator Prompt Enhancements

#### A. Simplified and Structured Validation Prompt

**Current issue:** 200+ line validation prompt
**Proposed:** Restructure into clear sections

```
You are an Agent Validator for warehouse robots.

CORE MISSION: Perform fast, focused safety validation of negotiated actions.

=== VALIDATION RULES (Check ALL positions in path) ===

1. ORTHOGONAL MOVES ONLY
   ✓ Valid: (x,y) → (x±1,y) or (x,y±1)
   ✗ Invalid: (x,y) → (x±1,y±1) [diagonal]
   ✓ Valid: (x,y) → (x,y) [zero-move/wait]

2. NO WALLS
   ✗ Invalid: Any position marked '#' in grid

3. WITHIN BOUNDS
   ✓ Valid: All positions in range x:[0-{max_x}], y:[0-{max_y}]

4. TRUST COORDINATOR
   - Don't second-guess timing/coordination
   - Focus ONLY on geometric validity

=== RESPONSE FORMAT ===

Success: {"valid": true, "reason": "all_checks_passed"}
Failure: {"valid": false, "reason": "diagonal_move_at_step_2: (3,4)→(2,5)"}

Be SPECIFIC in failure reasons: include step numbers and coordinates.
```

#### B. Standardized Error Codes

**Proposed error code system:**
```
STANDARD ERROR CODES (use these in 'reason' field):

Physical Violations:
- diagonal_move_at_step_{N}: {pos1}→{pos2}
- wall_collision_at: {position}
- out_of_bounds_at: {position} (max:{max_x},{max_y})

Path Issues:
- empty_path_provided
- path_too_long: {length} steps exceeds maximum
- discontinuous_path: gap between step {N} and {N+1}

Agent Issues:
- agent_position_unknown
- invalid_action_type: {action}

Success Codes:
- all_checks_passed
- zero_move_valid (when path contains staying in place)
- orthogonal_path_valid
```

#### C. Enhanced Alternative Suggestion Prompt

**Current:**
```python
system_prompt = """You are helping a robot find an alternative action when its planned action is invalid.

Suggest alternatives like:
- Wait for a turn
- Move to a different adjacent cell
- Take a longer but safer path
```

**Proposed:**
```python
system_prompt = """You are a robot path planning assistant specializing in alternative action suggestions.

CONTEXT: Agent {agent_id} had an action rejected due to: {rejection_reason}

YOUR TASK: Suggest a SAFE alternative action.

ALTERNATIVE STRATEGY GUIDE:

If rejection was due to:

1. diagonal_move → Suggest: Break into two orthogonal moves
   Example: (2,3)→(3,4) becomes (2,3)→(3,3)→(3,4)

2. wall_collision → Suggest: Route around wall using adjacent free cells
   Example: If (2,4) is wall, try (2,3)→(3,3)→(3,4) instead

3. agent_collision → Suggest: Wait one turn, then proceed
   Example: {"action": "wait", "reason": "let_other_agent_pass"}

4. out_of_bounds → Suggest: Trim path to stay within bounds
   Example: If max_x=7 but trying (8,3), suggest (7,3) instead

5. Complex issues → Suggest: Wait for coordinator's next plan
   Example: {"action": "wait", "reason": "defer_to_next_negotiation"}

RESPONSE FORMAT:
{
    "action": "move" or "wait",
    "path": [[x,y], ...] or [],
    "reasoning": "Specific explanation tied to rejection reason"
}

CRITICAL: Your alternative must be SAFE and SIMPLE. When in doubt, suggest "wait".
```

#### D. Explicit Zero-Move Validation

**Proposed addition to validation query:**
```
ZERO-MOVE CLARIFICATION:

A zero-move is when an agent stays at the same position:
- Example: (x,y) → (x,y)
- This is EQUIVALENT to waiting in place
- This is ALWAYS VALID (no movement = no collision risk)

When you see a zero-move in the path:
✓ DO: Accept it as valid
✗ DON'T: Reject it as "not moving" or "invalid move"

Zero-moves are a legitimate conflict resolution strategy!
```

### 3. Cross-Cutting Improvements

#### A. Temperature Optimization

**Current:**
- Central Negotiator: 0.3 (initial), 0.7 (refinement)
- Agent Validator: 0.1

**Proposed:**
- Central Negotiator: 0.2 (initial) - More consistent
- Refinement: 0.5 (refinement) - More creative but not wild
- Agent Validator: 0.05 - Maximum consistency for validation

**Rationale:** Lower temperature for initial negotiation ensures consistent, safe plans. Medium temperature for refinement allows creativity while maintaining coherence.

#### B. Token Limit Optimization

**Current:**
- Central: 30000 (reasoning), 20000 (standard)
- Validator: 15000 (reasoning), 10000 (standard)

**Proposed:**
- Central: 25000 (reasoning), 15000 (standard)
- Validator: 10000 (reasoning), 5000 (standard)

**Rationale:** Validators don't need long responses. Reducing token limits can:
- Reduce API costs
- Force more concise responses
- Improve response time

#### C. Response Parsing Robustness

**Proposed addition to parsing logic:**
```python
def _parse_with_retry(self, response: str, expected_schema: Dict) -> Dict:
    """
    Parse LLM response with schema validation and retry logic
    
    Args:
        response: Raw LLM response
        expected_schema: Dict describing expected JSON structure
    
    Returns:
        Parsed and validated response, or fallback
    """
    # Try standard parsing
    parsed = self._parse_json(response)
    
    # Validate against schema
    if self._validate_schema(parsed, expected_schema):
        return parsed
    
    # If failed, try recovery
    recovered = self._attempt_json_recovery(response)
    if recovered and self._validate_schema(recovered, expected_schema):
        return recovered
    
    # Log the failure for debugging
    logger.warning(f"Failed to parse response: {response[:100]}...")
    
    # Return safe fallback
    return self._create_safe_fallback(expected_schema)
```

### 4. Model-Specific Optimizations

#### A. Reasoning Model Enhancements

For models like o1, o3, DeepSeek, Gemini with reasoning capabilities:

**Proposed:**
```python
if self.is_reasoning_model:
    system_prompt += """
    
    REASONING MODEL INSTRUCTIONS:
    
    You have advanced reasoning capabilities. Use them to:
    
    1. THINK DEEPLY about the problem before responding
       - Consider multiple solution paths
       - Evaluate trade-offs explicitly
       - Reason about long-term consequences
    
    2. SHOW YOUR WORK (in your reasoning, not in final JSON)
       - Walk through your analysis step-by-step
       - Explain why you rejected certain approaches
       - Justify your final solution choice
    
    3. SELF-VERIFY your solution
       - Check your JSON against all constraints
       - Confirm no diagonal moves, walls, or collisions
       - Ensure paths are complete and valid
    
    Your <thinking> will be used to improve future prompts, so be thorough!
    """
```

#### B. Smaller Model Optimizations

For smaller models (Gemma, Llama, etc.):

**Proposed:**
```python
if not self.is_reasoning_model:
    # Use simpler, more directive language
    # Break complex instructions into numbered steps
    # Provide explicit examples
    # Avoid abstract concepts
    
    system_prompt += """
    
    VALIDATION STEPS (Follow in order):
    
    1. Check each step in path:
       For step i to i+1:
       a) If x changes AND y changes → REJECT (diagonal)
       b) If position is '#' → REJECT (wall)
       c) If x < 0 or x > max_x → REJECT (out of bounds)
       d) If y < 0 or y > max_y → REJECT (out of bounds)
    
    2. If all checks pass → APPROVE
    
    3. If any check fails → REJECT with specific reason
    
    EXAMPLE OUTPUTS:
    Good: {"valid": true, "reason": "all_checks_passed"}
    Bad: {"valid": false, "reason": "diagonal_move_at_step_1: (2,3)→(3,4)"}
    """
```

## Implementation Priority

### Phase 1 - High Impact (Week 1-2)
- [ ] Enhance Central Negotiator system prompt with resolution strategy details
- [ ] Add structured reasoning framework for reasoning models
- [ ] Implement standardized error codes in Agent Validator
- [ ] Simplify Agent Validator system prompt

### Phase 2 - Medium Impact (Week 3-4)
- [ ] Add failure pattern recognition to Central Negotiator
- [ ] Improve map visualization context
- [ ] Enhance alternative suggestion prompts
- [ ] Optimize temperature settings

### Phase 3 - Low Impact / Optimization (Week 5-6)
- [ ] Implement robust response parsing with retries
- [ ] Add model-specific optimizations
- [ ] Reduce token limits for efficiency
- [ ] Add comprehensive prompt versioning/logging

## Success Metrics

After implementing these improvements, we should see:

1. **Validation Success Rate**: Increase from current baseline by 15-25%
2. **Refinement Iterations**: Decrease average iterations from ~3 to ~1.5
3. **Parsing Failures**: Reduce JSON parsing errors by 50%
4. **Deadlock Resolution**: Improve deadlock breaking success rate by 20%
5. **Response Quality**: More specific, actionable error messages

## Testing Strategy

1. **A/B Testing**: Run simulations with old vs. new prompts on same scenarios
2. **Benchmark Suite**: Create set of challenging conflict scenarios
3. **Model Compatibility**: Test with multiple LLM providers (OpenRouter models)
4. **Failure Analysis**: Track specific failure types before/after
5. **Human Evaluation**: Manual review of reasoning quality

## Additional Context

**Related Files:**
- `src/llm/central_negotiator.py` - Lines 511-633 (prompts)
- `src/llm/agent_validator.py` - Lines 163-326 (prompts)

**References:**
- OpenRouter model capabilities: https://openrouter.ai/models
- Prompt engineering best practices: Chain-of-thought, few-shot examples
- Warehouse simulation constraints: docs/Basics.md

**Considerations:**
- Keep prompts concise while being comprehensive
- Balance between specificity and flexibility
- Ensure backward compatibility with existing model configurations
- Monitor API costs (token usage) during rollout

## Example Improvements in Action

### Before (Current Negotiator Prompt):
```
RULES: Robots deliver boxes, one per cell, avoid collisions.
RESOLUTION STRATEGIES:
1. "priority": Assign movement priorities
```

### After (Improved Negotiator Prompt):
```
WAREHOUSE RULES:
- Each cell holds max 1 robot or 1 box
- Robots must avoid collisions
- Goal: Deliver all boxes to targets efficiently

RESOLUTION STRATEGIES (Choose the best fit):

1. "priority" - Sequential Movement
   WHEN TO USE: Simple conflicts, non-overlapping goals
   HOW: Assign priority (higher goes first), others wait
   EXAMPLE: Two robots at (2,3) and (4,3) both heading right
   → Higher priority moves through, lower waits
   SUCCESS INDICATOR: Clear sequencing order exists

2. "reroute" - Alternative Paths  
   WHEN TO USE: Wiggle rooms available, complex paths
   HOW: Use empty spaces for temporary positioning
   EXAMPLE: Robot blocked by another, moves to (2,4) then (3,4) instead of direct (2,3)→(3,3)
   SUCCESS INDICATOR: Multiple valid paths exist

3. "wait" - Pause & Reassess
   WHEN TO USE: Deadlock, insufficient information
   HOW: All agents pause for one turn
   EXAMPLE: Three robots in circular dependency
   → Wait, then coordinator will provide new plan
   SUCCESS INDICATOR: Situation too complex for immediate resolution
```

This improved version gives the LLM:
- Clear decision criteria for strategy selection
- Concrete examples of each strategy
- Success indicators to validate choice
- Structured format that's easier to parse

---

**Priority:** High
**Effort:** Medium (2-3 weeks)
**Impact:** High (directly improves core functionality)
