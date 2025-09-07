# LLM Negotiation Testing System



### 1. `test_negotiation.py` - Main Test Suite
**Purpose**: Comprehensive negotiation testing with detailed logging

**Features**:
- ✅ **Forces 100% conflict rate** with artificial scenarios
- ✅ **Captures both prompts and responses** from LLM negotiations  
- ✅ **Saves all data to logs/** folder with timestamps
- ✅ **Interactive testing** with step-by-step control
- ✅ **Multiple conflict scenarios** to test different situations


**Scenarios Available**:
1. **Single Corridor**: Two agents must cross paths in narrow corridor
2. **Bottleneck Chamber**: Three agents must pass through single bottleneck  
3. **Triple Intersection**: Three agents converge at single intersection

### 2. `test_negotiation_auto.py` - Automated Demo
**Purpose**: Quick automatic test without manual input



## What Gets Logged

Each negotiation captures:

```json
{
  "turn": 3,
  "timestamp": "2025-09-07T15:30:45",
  "system_prompt": "You are a central negotiator for multi-agent path planning...",
  "user_prompt": "CONFLICT DETECTED: Agents 0 and 1 at positions...",
  "llm_response": {
    "resolution": "priority",
    "agent_actions": {
      "0": {"action": "wait", "priority": 1},
      "1": {"action": "move", "path": [...], "priority": 2}
    },
    "reasoning": "Agent 1 has shorter path, should proceed first..."
  },
  "conflict_data": {
    "agents": [...],
    "conflict_points": [(2,2)],
    "map_state": {...}
  }
}
```

## Log Files Structure

Saved to `logs/negotiation_test_YYYYMMDD_HHMMSS.json`:

```json
{
  "test_info": {
    "timestamp": "2025-09-07T15:30:45",
    "test_type": "forced_conflict_negotiation",
    "description": "Artificial scenarios designed to force 100% conflict rate"
  },
  "negotiations": [/* All negotiation entries */],
  "conflict_scenarios": [/* Scenario details */],
  "summary": {
    "total_scenarios": 1,
    "total_negotiations": 3,
    "total_conflicts": 3,
    "success_rate": 1.0
  }
}
```
## Example Output

```
🤖 CONFLICT DETECTED! Initiating LLM Negotiation...
   Conflicting agents: [0, 1]
   Conflict points: [(3, 1)]

📋 SYSTEM PROMPT:
==================================================
You are a central negotiator for multi-agent path planning...
==================================================

📝 USER PROMPT TO LLM:
==================================================
CONFLICT DETECTED at turn 2:

Agent 0: Currently at (1, 1), targeting (6, 1)
Planned path: [(1,1), (2,1), (3,1), (4,1), (5,1), (6,1)]

Agent 1: Currently at (6, 1), targeting (1, 1)  
Planned path: [(6,1), (5,1), (4,1), (3,1), (2,1), (1,1)]

CONFLICT POINTS: [(3,1)]
==================================================

💬 LLM RESPONSE:
----------------------------------------
{
  "resolution": "priority",
  "agent_actions": {
    "0": {"action": "move", "priority": 1},
    "1": {"action": "wait", "priority": 2, "wait_turns": 3}
  },
  "reasoning": "Agent 0 should proceed first as it started from the left..."
}
----------------------------------------
```
