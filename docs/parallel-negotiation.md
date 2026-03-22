# Parallel Central Negotiation Framework

## Overview

The parallel central negotiation framework enables the system to handle multiple independent conflict groups simultaneously, significantly improving scalability when many agents are present on the map.

## Architecture

### Conflict Grouping

When conflicts are detected, the system uses graph connectivity analysis to identify independent conflict groups:

1. **Conflict Detection**: Standard path conflict detection identifies all conflicting agents
2. **Graph Construction**: Build adjacency graph where edges represent direct conflicts between agents
3. **Connected Components**: Use DFS to find connected components (independent groups)
4. **Parallel Resolution**: Each group is resolved by its own central negotiator instance

### Components

#### 1. ConflictDetector.group_conflicts()
**Location**: `src/navigation/__init__.py`

Groups conflicts into independent sets based on agent connectivity.

```python
def group_conflicts(self, agents_paths: Dict, current_turn: int) -> List[Dict]:
    """
    Returns list of conflict groups, each with:
    - has_conflicts: bool
    - conflicting_agents: List[int]
    - conflict_points: List[Tuple[int, int]]
    - conflict_turns: List[int]
    """
```

**Algorithm**:
- Builds agent conflict graph (agents that collide/swap are connected)
- Finds connected components using DFS
- Returns one conflict dict per independent group

**Example**:
```
Agents 1,3 conflict at [3,5]
Agents 5,6,7 conflict at [7,7],[7,8],[7,9]
Agent 2 conflicts at [9,9]

Result: 3 independent groups
```

#### 2. ParallelNegotiatorManager
**Location**: `src/llm/parallel_negotiator_manager.py`

Coordinates multiple central negotiators running in parallel.

**Key Methods**:

- `negotiate_parallel_conflicts()`: Main entry point
  - Creates dedicated negotiator per group
  - Runs negotiations in parallel (ThreadPoolExecutor)
  - Collects results from all groups

- `merge_resolutions()`: Combines group resolutions
  - Merges agent_actions from all groups
  - Tracks per-group status (resolved/deadlock)

**Configuration**:
```python
manager = ParallelNegotiatorManager(
    model=None,  # Inherits from env var
    enable_spatial_hints=True
)
```

#### 3. GameEngine Integration
**Location**: `src/simulation/game_engine.py`

**New Method**: `_negotiate_parallel_conflicts()`
- Groups conflicts using ConflictDetector
- Falls back to single negotiation if only 1 group
- Prepares conflict data for each group
- Invokes ParallelNegotiatorManager
- Returns merged resolution and list of log data

**Toggle Behavior**:
```python
if self.use_parallel_negotiation:
    resolution, neg_data_list = self._negotiate_parallel_conflicts(conflict_info, moves)
else:
    resolution, neg_data = self._negotiate_conflicts(conflict_info, moves)
```

#### 4. UnifiedLogger Updates
**Location**: `src/logging/unified_logger.py`

**Enhanced log_turn()**:
- Accepts `negotiation_data` as Dict (single) or List[Dict] (parallel)
- Creates structured log with `negotiation_mode: 'parallel'`
- Groups metadata includes group_index and total_groups

**Log Structure** (parallel mode):
```json
{
  "type": "negotiation_parallel",
  "negotiation": {
    "negotiation_mode": "parallel",
    "total_groups": 3,
    "groups": [
      {
        "group_index": 0,
        "conflict_data": {...},
        "hmas2_stages": {...},
        ...
      },
      ...
    ]
  }
}
```

**Metrics Calculation**:
- `_calculate_hmas2_metrics()` iterates through all groups
- Aggregates validations, approvals, rejections across groups
- Summary includes `parallel_negotiation_turns` count

## Configuration

### Environment Variable

```bash
# Enable parallel negotiation (default)
USE_PARALLEL_NEGOTIATION=true

# Disable to use legacy single negotiation
USE_PARALLEL_NEGOTIATION=false
```

Add to `.env` file or set in environment.

### Performance Tuning

**ParallelNegotiatorManager**:
```python
self.max_parallel_negotiations = 32  # Max concurrent negotiations
```

Adjust based on system resources and API rate limits.

## Usage Example

### Scenario
- 10 agents with multiple conflicts
- Groups: {1,2,3}, {4,5}, {6,7,8,9}

### Execution Flow

1. **Conflict Detection**:
   ```python
   conflict_info = self.conflict_detector.detect_path_conflicts(planned_moves)
   # conflicting_agents: [1,2,3,4,5,6,7,8,9]
   ```

2. **Conflict Grouping**:
   ```python
   groups = self.conflict_detector.group_conflicts(planned_moves)
   # Group 1: agents [1,2,3]
   # Group 2: agents [4,5]
   # Group 3: agents [6,7,8,9]
   ```

3. **Parallel Negotiation**:
   ```python
   resolutions, logs = manager.negotiate_parallel_conflicts(groups, validators)
   # All 3 negotiations run simultaneously
   ```

4. **Merge Results**:
   ```python
   merged = manager.merge_resolutions(resolutions)
   # Combined agent_actions for all 9 agents
   ```

## Performance Characteristics

### Scalability Benefits

**Without Parallel Negotiation**:
- 1 group of 10 agents: ~30s LLM call
- Linear scaling with agent count

**With Parallel Negotiation**:
- 3 groups of [3,2,5] agents: 3 parallel ~10-15s LLM calls
- Wall-clock time: max(group times) ≈ 15s
- **2x speedup** in this example

### When Parallel Helps Most

✅ **Best cases**:
- Multiple spatially-separated conflicts
- Large maps with independent regions
- High agent counts (6+ agents)

⚠️ **Limited benefit**:
- Single large conflict group
- All agents in same region
- Few agents (2-4 agents)

## Fallback Behavior

The system gracefully falls back to single negotiation when:

1. No conflict groups identified → Use `_negotiate_conflicts()`
2. Only 1 conflict group → Use `_negotiate_conflicts()`
3. `USE_PARALLEL_NEGOTIATION=false` → Always use `_negotiate_conflicts()`

## Testing

### Manual Test Scenario

1. Load `s_shaped` map (forces complex conflicts)
2. Enable parallel negotiation:
   ```bash
   export USE_PARALLEL_NEGOTIATION=true
   ```
3. Run simulation:
   ```bash
   python main.py
   ```
4. Observe console output:
   ```
   🔄 Initiating PARALLEL conflict negotiation...
      Total conflict groups: 2
      Group 1: 3 agent(s) at 2 conflict point(s)
      Group 2: 2 agent(s) at 1 conflict point(s)

   ⚡ Starting 2 parallel negotiation(s) with 2 worker(s)...
   ```

### Verification

Check simulation logs:
```bash
cat logs/sim_log_*.json | jq '.summary'
```

Should show:
```json
{
  "parallel_negotiation_turns": 5,
  "single_negotiation_turns": 0,
  ...
}
```

## Visualization Compatibility

The visualization tools (`visualization.py`) should work without changes:
- Logs contain all necessary agent paths and states
- Parallel negotiations are stored in structured format
- Replay functionality unchanged

Verify by running:
```bash
python visualization.py logs/sim_log_*.json
```

## Troubleshooting

### Issue: All negotiations are single-mode

**Cause**: Conflicts not splitting into groups

**Solution**:
1. Check conflict points are spatially separated
2. Verify no shared agents between conflict regions
3. Increase agent count to trigger multiple groups

### Issue: Slower than single negotiation

**Cause**: API overhead or small groups

**Solution**:
1. Groups must be sufficiently large to benefit from parallelism
2. Check API rate limits aren't being hit
3. Monitor `max_parallel_negotiations` setting

### Issue: Logs not showing parallel data

**Cause**: Legacy mode active or fallback triggered

**Solution**:
1. Verify `USE_PARALLEL_NEGOTIATION=true` in environment
2. Check console output for "PARALLEL conflict negotiation"
3. Ensure multiple conflict groups exist

## Future Enhancements

Potential improvements:
- **Dynamic grouping**: Re-group conflicts after partial resolution
- **Priority-based scheduling**: High-priority groups first
- **Adaptive parallelism**: Adjust workers based on group sizes
- **Cross-group coordination**: Handle edge cases where groups interact

## References

- Central Negotiator: `src/llm/central_negotiator.py`
- Conflict Detection: `src/navigation/__init__.py`
- Game Engine: `src/simulation/game_engine.py`
- Unified Logger: `src/logging/unified_logger.py`
