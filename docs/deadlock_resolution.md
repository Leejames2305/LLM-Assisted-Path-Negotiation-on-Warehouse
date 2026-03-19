# Deadlock Resolution Mechanism

## Overview

The deadlock resolution mechanism prevents agents from getting stuck due to repeated failed moves. When agents fail to execute moves consecutively (e.g., hitting walls, colliding with other agents), the system detects this as stagnation and attempts to resolve it.

## Resolution Strategy

### Two-Tier Approach

The system uses a **two-tier resolution strategy** to minimize LLM API calls and improve resolution time:

1. **Tier 1: A\* Pathfinding (Fast & Cost-Free)**
   - When an agent is stuck (3 consecutive failed moves)
   - System first attempts A\* pathfinding to find an alternative path
   - Checks if the new path conflicts with other agents' planned paths
   - If successful and conflict-free, the agent uses this path

2. **Tier 2: LLM Negotiation (Smart but Slower)**
   - Only triggered if A\* pathfinding fails OR generates a conflicting path
   - LLM negotiates paths between multiple agents
   - Handles complex scenarios requiring coordination

## Key Benefits

- **Reduced Token Cost**: A\* pathfinding is free, reducing unnecessary LLM API calls
- **Faster Resolution**: A\* is deterministic and instant, no network latency
- **Better Path Quality**: Addresses LLM hallucination issues where generated paths may be imperfect
- **Fallback Safety**: LLM negotiation still available for complex scenarios

## Implementation Details

### Stagnation Detection

Located in `src/simulation/game_engine.py::detect_stagnation_conflicts()`

```python
# Agent is marked as stagnant after 3 consecutive failed moves
if failed_move_count >= self.stagnation_turns:  # default: 3
    stagnant_agents.append(agent_id)
```

### Resolution Flow

```
Agent Stuck (3 failed moves)
    ↓
Try A* Pathfinding
    ↓
Path Found? ──NO──→ Trigger LLM Negotiation
    ↓ YES
Check Path Conflicts
    ↓
Conflicts? ──YES──→ Trigger LLM Negotiation
    ↓ NO
Use A* Path & Clear Failed Move History
```

### Conflict Detection

The `_check_path_conflicts_with_others()` method checks if a proposed path:
- Collides with other agents at any time step
- Creates swap conflicts (agents exchanging positions)
- Uses the existing `ConflictDetector` for consistency

### Example Scenario

**Problem**: Agent 8 got an LLM-negotiated path that hit a wall due to hallucination

**Old Behavior**:
1. Agent fails to move (wall collision)
2. After 3 failures → LLM negotiation again
3. Potentially get another imperfect path

**New Behavior**:
1. Agent fails to move (wall collision)
2. After 3 failures → A\* pathfinding first
3. A\* finds valid path around wall
4. Path is conflict-free → Use it immediately
5. LLM negotiation avoided entirely

## Configuration

```python
# In GameEngine.__init__()
self.stagnation_turns = 3      # Threshold for stagnation detection
self.max_failed_moves = 3      # Threshold for deadlock detection
```

## Metrics Impact

This improvement should show:
- **Reduced LLM API calls**: Fewer negotiation rounds
- **Lower token costs**: Less prompt/completion tokens used
- **Faster resolution**: A\* is instantaneous vs. network + LLM time
- **Better path quality**: Deterministic A\* paths vs. potentially hallucinated LLM paths

## Related Files

- `src/simulation/game_engine.py`: Main implementation
- `src/navigation/__init__.py`: A\* pathfinding and conflict detection
- `src/llm/central_negotiator.py`: LLM negotiation (Tier 2)
