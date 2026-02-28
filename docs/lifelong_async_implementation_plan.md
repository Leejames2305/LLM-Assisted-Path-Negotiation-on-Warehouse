# 🗺️ Implementation Plan: Async Engine & LifeLong Benchmarking

This plan is broken into **3 major epics**, each with sub-tasks, touching specific files you already have. Every decision is grounded in what the existing code actually does.

---

## Overview of Changes

```
benchmark_tool.py              → Add LifelongBenchmarkConfig, LifelongRoundResult, run_lifelong_benchmark()
src/simulation/game_engine.py  → Add simulation_mode flag, async step logic, task re-assignment, lifelong termination
src/logging/unified_logger.py  → Add lifelong-aware log_task_completion(), lifelong summary metrics
```

---

## Epic 1 — Refactor `GameEngine` for Dual-Mode (Turn-Based + Async)

### Background & Problem

The current `run_simulation_step()` is a **blocking, synchronous turn gate**: every agent is resolved in a single step, and any conflict triggers a **global stop** while LLM negotiation completes. This is fine for turn-based benchmarks because all agents move atomically per turn — but it breaks Lifelong mode, where agents should continue working independently while a conflict-pair is being resolved.

The key insight: **async mode doesn't throw away turns** — it still advances a discrete turn counter. What changes is that **non-conflicting agents are not frozen** while conflicting agents wait for LLM resolution.

---

### 1.1 — Add `simulation_mode` flag to `GameEngine.__init__`

**File:** `src/simulation/game_engine.py`

Add to `__init__`:

```python
# Simulation mode: 'turn_based' (default) or 'async'
self.simulation_mode = 'turn_based'

# Async mode: agents blocked waiting for LLM resolution
self._pending_resolution_agents: set = set()

# Async mode: per-agent last-known safe position (before conflict)
self._conflict_hold_positions: Dict[int, Tuple[int, int]] = {}
```

**Why:** This single flag gates all async-specific behaviour, so turn-based runs stay completely untouched and backward-compatible.

---

### 1.2 — Split `run_simulation_step()` into mode-aware dispatch

Replace the single `run_simulation_step()` with:

```python
def run_simulation_step(self) -> bool:
    if self.simulation_mode == 'async':
        return self._run_async_step()
    else:
        return self._run_turn_based_step()
```

Rename the **current body** of `run_simulation_step()` → `_run_turn_based_step()` (zero logic change, just a rename). This keeps all existing benchmark behaviour intact.

---

### 1.3 — Implement `_run_async_step()`

The async step logic:

```
1. Advance turn counter & call agent.update_turn() for all agents
2. Check deliveries (unchanged)
3. Partition agents into two groups:
   - "free agents"   → not in self._pending_resolution_agents
   - "held agents"   → in self._pending_resolution_agents (awaiting LLM)
4. For FREE agents:
   a. Run stagnation check (same as Phase 0)
   b. Get forced moves and check for conflicts
   c. If conflict detected:
      - Add conflicting agents to _pending_resolution_agents
      - Store their hold positions in _conflict_hold_positions
      - Fire _negotiate_conflicts() [this is the async LLM call]
        → In this plan we keep it synchronous per-conflict-group,
          but non-conflicting agents still move this turn
   d. If no conflict: execute normal planned moves
5. For HELD agents:
   - If their LLM resolution has completed → execute resolution & remove from pending
   - If still waiting → hold in place (move to _conflict_hold_positions[agent_id])
6. Update map state, log turn, display
7. Check lifelong/turn-based termination
```

**Key difference from turn-based:**
- In turn-based: ALL agents are frozen when ANY conflict fires.
- In async: only the conflicting subset is frozen; others keep moving.

> **Note on true concurrency:** Python's GIL and the synchronous OpenRouter API calls mean we can't fire multiple LLM calls in truly parallel threads safely. The "async" here means **agent-level isolation of conflict resolution**, not OS-level threading. If you later want real concurrency, you can wrap `_negotiate_conflicts()` with `asyncio` + `await`, but that requires the OpenRouter client to support async — worth a separate PR.

---

### 1.4 — Lifelong task re-assignment after delivery

Currently `_check_box_delivery()` calls `agent.set_target(None)` and that's the end. In Lifelong mode, a completed agent should immediately get a new task.

Add a new method:

```python
def _assign_next_task_lifelong(self, agent_id: int):
    """Assign a new random box+target pair to agent after delivery."""
    # Pick from available (unassigned) boxes in warehouse_map.boxes
    # Assign agent_goal mapping for the new box/target pair
    # Call agent.set_target(box_pos) to start the new pickup phase
    # Log the new task assignment (for throughput metrics)
```

Modify `_check_box_delivery()`:

```python
if success:
    # ... existing delivery logic ...
    if self.simulation_mode == 'lifelong':
        self._assign_next_task_lifelong(agent_id)
```

> For Lifelong mode, boxes/targets are conceptually **infinite** — when the pool is exhausted, regenerate new random positions (similar to what `generate_random_positions()` already does in `benchmark_tool.py`).

---

### 1.5 — Add `simulation_mode = 'lifelong'` and time-based termination

Lifelong mode is a variant of async mode. Add a third mode value:

```python
# 'turn_based' | 'async' | 'lifelong'
self.simulation_mode = 'turn_based'
```

In `run_simulation_step()`:

```python
def run_simulation_step(self) -> bool:
    if self.simulation_mode in ('async', 'lifelong'):
        return self._run_async_step()
    else:
        return self._run_turn_based_step()
```

The **termination condition** changes:
- Turn-based: ends when `_check_completion()` is `True` OR `max_turns` hit.
- Lifelong: ends **only** when `timeout_seconds` is hit (never on completion, since tasks keep coming).

Modify the timeout/completion check at the top of `_run_async_step()`:

```python
if self.simulation_mode == 'lifelong':
    # Never terminate on completion — only on timeout
    if self.timeout_seconds > 0 and elapsed >= self.timeout_seconds:
        return False
else:
    # Async but not lifelong: original turn/completion checks still apply
    if self.simulation_complete or self.current_turn >= self.max_turns:
        return False
```

---

### 1.6 — Track throughput metrics in `GameEngine`

Add new counters to `__init__`:

```python
# Lifelong mode metrics
self.total_tasks_completed = 0      # total deliveries across entire run
self.task_completion_timestamps: List[float] = []  # wall-clock time of each delivery
self.agent_task_counts: Dict[int, int] = {}        # per-agent deliveries
```

Update `_check_box_delivery()` to populate these.

Update `calculate_performance_metrics()` to include:

```python
if self.simulation_mode == 'lifelong':
    metrics['throughput_tasks_per_second'] = ...
    metrics['per_agent_task_counts'] = self.agent_task_counts
    metrics['total_tasks_completed'] = self.total_tasks_completed
```

---

## Epic 2 — LifeLong Benchmark Mode in `benchmark_tool.py`

### 2.1 — Add `LifelongBenchmarkConfig` dataclass

```python
@dataclass
class LifelongBenchmarkConfig:
    num_agents: int
    num_rounds: int
    time_limit_seconds: int   # duration of each lifelong run
    seed: int
    spatial_hints_enabled: bool
    task_pool_size: int        # how many box/target pairs to pre-generate per run
    
    @classmethod
    def from_env(cls) -> 'LifelongBenchmarkConfig':
        return cls(
            num_agents=int(os.getenv('BENCHMARK_NUM_AGENTS', '2')),
            num_rounds=int(os.getenv('BENCHMARK_NUM_ROUNDS', '3')),
            time_limit_seconds=int(os.getenv('BENCHMARK_TIME_LIMIT_SECONDS', '300')),
            seed=int(os.getenv('BENCHMARK_SEED', '42')),
            spatial_hints_enabled=os.getenv('BENCHMARK_SPATIAL_HINTS_ENABLED', 'true').lower() == 'true',
            task_pool_size=int(os.getenv('BENCHMARK_LIFELONG_TASK_POOL', '50'))
        )
```

New `.env` variable: `BENCHMARK_LIFELONG_TASK_POOL=50`

---

### 2.2 — Add `LifelongRoundResult` dataclass

The metrics are fundamentally different from `RoundResult`:

```python
@dataclass
class LifelongRoundResult:
    round_num: int
    status: str                        # 'completed' (hit time limit, which is always the goal)
    
    # Throughput (primary metric for lifelong)
    total_tasks_completed: int
    throughput_tasks_per_second: float  # = total_tasks_completed / time_limit_seconds
    throughput_tasks_per_turn: float    # = total_tasks_completed / total_turns
    
    # Per-agent fairness
    per_agent_task_counts: Dict[int, int]  # how many tasks each agent completed
    agent_utilization: Dict[int, float]    # % of turns agent was actively working
    
    # Conflict metrics (same as before, still relevant)
    total_negotiations: int
    total_collisions: int
    avg_conflict_resolution_time_ms: float
    
    # LLM cost
    total_tokens_used: int
    tokens_per_task: float             # = total_tokens / total_tasks_completed
    
    # Time
    total_turns: int
    wall_clock_seconds: float
```

---

### 2.3 — Add `run_lifelong_round()` function

Mirrors `run_single_round()` but:
1. Sets `game_engine.simulation_mode = 'lifelong'`
2. Pre-generates a task pool using `generate_random_positions()` (already exists)
3. Passes the task pool to `GameEngine` so `_assign_next_task_lifelong()` can draw from it
4. Runs `while game_engine.run_simulation_step(): pass` — ends only when timeout fires
5. Collects `LifelongRoundResult` from `game_engine.calculate_performance_metrics()`

---

### 2.4 — Add `run_lifelong_benchmark()` function

Mirrors `run_benchmark()` but:
- Uses `LifelongBenchmarkConfig`
- Calls `run_lifelong_round()` for each round
- Saves results with `save_lifelong_csv()` and `save_lifelong_summary()`

---

### 2.5 — Update `main()` with mode selection

```python
def main():
    # Ask user to choose benchmark mode
    print("Select benchmark mode:")
    print("  1. Standard Benchmark (turn-based, task completion)")
    print("  2. Lifelong Benchmark (async, throughput over time)")
    
    mode = input("Enter 1 or 2: ").strip()
    
    if mode == '2':
        config = LifelongBenchmarkConfig.from_env()
        # ... select layout, confirm, run_lifelong_benchmark()
    else:
        config = BenchmarkConfig.from_env()
        # ... existing flow (unchanged)
```

---

### 2.6 — Add `.env.example` entries

```
# Lifelong Benchmark Settings
BENCHMARK_LIFELONG_TASK_POOL=50
```

---

## Epic 3 — `UnifiedLogger` Updates for Dual-Mode Logging

### 3.1 — Add `mode` to `initialize()`

```python
def initialize(self, scenario_data: Dict) -> None:
    self.log_data['scenario'] = {
        'type': scenario_data.get('type', 'simulation'),
        'simulation_mode': scenario_data.get('simulation_mode', 'turn_based'),  # NEW
        # ... existing fields
    }
    self._simulation_mode = scenario_data.get('simulation_mode', 'turn_based')  # NEW
```

**Why:** `finalize()` needs to know which summary to compute.

---

### 3.2 — Add `log_task_completion()` for lifelong events

```python
def log_task_completion(
    self,
    turn_num: int,
    agent_id: int,
    task_num: int,              # cumulative task count for this agent
    box_id: int,
    target_id: int,
    pickup_turn: int,           # when agent picked up this box
    wall_clock_time: float      # time.time() at delivery
) -> None:
    """Log a single task completion event in lifelong mode."""
    if 'task_completions' not in self.log_data:
        self.log_data['task_completions'] = []
    
    self.log_data['task_completions'].append({
        'turn': turn_num,
        'timestamp': datetime.now().isoformat(),
        'agent_id': agent_id,
        'task_num': task_num,
        'box_id': box_id,
        'target_id': target_id,
        'pickup_turn': pickup_turn,
        'task_duration_turns': turn_num - pickup_turn,
        'wall_clock_time': wall_clock_time
    })
    self._unsaved_data = True
```

---

### 3.3 — Mode-aware `finalize()` summary

```python
def finalize(self, emergency: bool = False, performance_metrics: Optional[Dict] = None) -> Optional[str]:
    # ... existing summary computation for turn-based ...
    
    if self._simulation_mode == 'lifelong':
        self.log_data['summary'] = self._compute_lifelong_summary(performance_metrics)
    else:
        self.log_data['summary'] = self._compute_turnbased_summary(performance_metrics)
    
    # ... filename generation and file save (unchanged) ...
```

Extract the **existing summary logic** into `_compute_turnbased_summary()` — zero behavioural change for existing users.

---

### 3.4 — Add `_compute_lifelong_summary()`

```python
def _compute_lifelong_summary(self, performance_metrics: Optional[Dict]) -> Dict:
    task_completions = self.log_data.get('task_completions', [])
    turns = self.log_data.get('turns', [])
    negotiation_turns = [t for t in turns if t.get('type') == 'negotiation']
    
    # Throughput over time (tasks per second at each 30s window)
    throughput_timeline = self._compute_throughput_timeline(task_completions)
    
    # Per-agent breakdown
    per_agent = {}
    for tc in task_completions:
        aid = tc['agent_id']
        per_agent.setdefault(aid, {'tasks': 0, 'avg_task_turns': []})
        per_agent[aid]['tasks'] += 1
        per_agent[aid]['avg_task_turns'].append(tc['task_duration_turns'])
    
    return {
        'simulation_mode': 'lifelong',
        'total_turns': len(turns),
        'total_tasks_completed': len(task_completions),
        'throughput_tasks_per_second': performance_metrics.get('throughput_tasks_per_second', 0),
        'throughput_tasks_per_turn': len(task_completions) / max(len(turns), 1),
        'throughput_timeline': throughput_timeline,
        'per_agent_stats': per_agent,
        'negotiation_metrics': {
            'total_negotiations': len(negotiation_turns),
            'hmas2_metrics': self._calculate_hmas2_metrics(negotiation_turns)
        },
        'llm_cost': {
            'total_tokens': performance_metrics.get('total_tokens_used', 0),
            'tokens_per_task': (
                performance_metrics.get('total_tokens_used', 0) / max(len(task_completions), 1)
            )
        },
        'completion_timestamp': datetime.now().isoformat()
    }
```

---

### 3.5 — Update `initialize_simulation()` in `GameEngine` to pass mode

In `game_engine.py`'s `initialize_simulation()`:

```python
self.logger.initialize({
    'type': 'simulation',
    'simulation_mode': self.simulation_mode,   # NEW
    # ... existing fields
})
```

---

## Summary Table

| Task | File | Scope |
|---|---|---|
| 1.1 Add `simulation_mode` flag | `src/simulation/game_engine.py` | `__init__` |
| 1.2 Mode-dispatch in `run_simulation_step()` | `src/simulation/game_engine.py` | Method rename + wrapper |
| 1.3 Implement `_run_async_step()` | `src/simulation/game_engine.py` | New method (~80 lines) |
| 1.4 `_assign_next_task_lifelong()` | `src/simulation/game_engine.py` | New method + hook in `_check_box_delivery` |
| 1.5 Lifelong termination condition | `src/simulation/game_engine.py` | Inside `_run_async_step()` |
| 1.6 Throughput metric counters | `src/simulation/game_engine.py` | `__init__` + `calculate_performance_metrics()` |
| 2.1 `LifelongBenchmarkConfig` | `benchmark_tool.py` | New dataclass |
| 2.2 `LifelongRoundResult` | `benchmark_tool.py` | New dataclass |
| 2.3 `run_lifelong_round()` | `benchmark_tool.py` | New function |
| 2.4 `run_lifelong_benchmark()` | `benchmark_tool.py` | New function |
| 2.5 Mode selection in `main()` | `benchmark_tool.py` | Updated `main()` |
| 2.6 New `.env` variables | `.env.example` | Config |
| 3.1 `mode` in `initialize()` | `src/logging/unified_logger.py` | `initialize()` |
| 3.2 `log_task_completion()` | `src/logging/unified_logger.py` | New method |
| 3.3 Mode-aware `finalize()` | `src/logging/unified_logger.py` | `finalize()` refactor |
| 3.4 `_compute_lifelong_summary()` | `src/logging/unified_logger.py` | New method |
| 3.5 Pass `simulation_mode` from engine | `src/simulation/game_engine.py` | `initialize_simulation()` |

---

## Metrics Comparison

| Metric | Turn-Based Benchmark | Lifelong Benchmark |
|---|---|---|
| **Primary success signal** | `cooperative_success_rate` (CSR) | `throughput_tasks_per_second` |
| **Makespan** | Time to complete all tasks | N/A (open-ended) |
| **Throughput** | N/A | Tasks/second, Tasks/turn |
| **Per-agent fairness** | N/A | `per_agent_task_counts`, utilization % |
| **Conflict metrics** | `collision_rate`, negotiation count | Same (still tracked) |
| **LLM cost** | `total_tokens_used` | `total_tokens_used` + `tokens_per_task` |
| **Termination** | All tasks done OR timeout | Timeout only |
| **Round status** | `success` / `timeout` / `failed` | Always `completed` (timeout = success) |

---

## Recommended Implementation Order

1. **Start with Epic 1.1 + 1.2** — the mode flag + step dispatch. This is the safest change and is purely additive. Run existing benchmarks to confirm nothing broke.
2. **Epic 1.3** — `_run_async_step()`. Test using `s_shaped` map to confirm non-conflicting agents keep moving while one pair negotiates.
3. **Epic 1.4 + 1.5 + 1.6** — lifelong task re-assignment and termination. Test manually by watching agents cycle through tasks.
4. **Epic 3.1 → 3.5** — logger updates next, so logging is ready before the benchmark tool runs.
5. **Epic 2.1 → 2.5** — benchmark tool last, wiring everything together.