# POGEMA Porting Plan — Stage-by-Stage TODO List
## LLM-Assisted Path Negotiation on Warehouse → POGEMA

### Target Architecture

```
pogema_port/
├── __init__.py
├── main.py                    → New POGEMA entrypoint
├── config.py                  → GridConfig builder + .env loading + map save/load
├── pathfinding/
│   ├── __init__.py
│   └── astar.py               → Ported SimplePathfinder (global A* on POGEMA grid)
├── negotiation/
│   ├── __init__.py
│   ├── conflict_detector.py   → Adapted conflict detection for POGEMA
│   ├── central_negotiator.py  → Reused from src/llm/ (minimal changes)
│   ├── agent_validator.py     → Reused from src/llm/ (minimal changes)
│   ├── openrouter_client.py   → Reused from src/llm/ (as-is)
│   └── openrouter_config.py   → Reused from src/llm/ (as-is)
├── agent_controller.py        → Per-agent wrapper: A* + negotiation → POGEMA actions
└── requirements.txt           → pogema, pogema-toolbox, requests, python-dotenv, numpy
```

### Trade-offs Accepted
1. ❌ Picking boxes → ✅ Agent straight pathfind to target
2. ⏳ Wiggle space → Implement later, not in this port
3. ❌ visualization.py, GameEngine → ✅ POGEMA env + pogema-toolbox rendering/metrics
4. ⚠️ LLM negotiated multi-step paths → Fed to POGEMA one action at a time per step
5. ❌ Custom JSON layout format → ✅ POGEMA GridConfig (with agents_xy/targets_xy)
6. ❌ UnifiedLogger → ✅ POGEMA + pogema-toolbox built-in metrics/logging/visualization

### Critical: Coordinate Convention
- Current codebase uses (x, y) everywhere
- POGEMA uses (row, col) which is (y, x)
- Create ONE utility function and use it everywhere

---

## Stage 0: Branch Setup & POGEMA Installation

Goal: Create a clean branch and install POGEMA + toolbox.

- [ ] 0.1 — Create new branch `feat/pogema-port` from `main`
- [ ] 0.2 — Create `pogema_port/` directory with `__init__.py`
- [ ] 0.3 — Create `pogema_port/requirements.txt`:
  ```
  pogema>=1.4.0
  pogema-toolbox>=0.2.0
  requests>=2.31.0
  python-dotenv>=1.0.0
  numpy>=1.24.0
  colorama>=0.4.6
  ```
  (matplotlib removed — POGEMA handles visualization)
- [ ] 0.4 — Run `pip install -r pogema_port/requirements.txt` to verify installation
- [ ] 0.5 — Write smoke test script `pogema_port/smoke_test.py`:
  - Creates a basic POGEMA env with `GridConfig(num_agents=2, size=8, density=0.3)`
  - Runs 10 random steps via `env.step(env.sample_actions())`
  - Prints observations and confirms env works
  - Exits cleanly

---

## Stage 1: POGEMA GridConfig Map System

Goal: Build the map configuration layer using POGEMA's native GridConfig.
Trash the old JSON layout format entirely. Use GridConfig's `map`, `agents_xy`,
`targets_xy` as the source of truth. Create a simple save/load utility for
persisting GridConfig setups as JSON for reproducibility.

- [ ] 1.1 — Create `pogema_port/config.py` with:

  - `create_grid_config(map_grid, agents_xy, targets_xy, **kwargs) -> GridConfig`
    - `map_grid`: list of strings (POGEMA format, '.' = free, '#' = obstacle)
    - `agents_xy`: list of (row, col) start positions
    - `targets_xy`: list of (row, col) goal positions
    - `**kwargs`: obs_radius, max_episode_steps, seed (loaded from .env with defaults)
    - Returns a fully configured `GridConfig`

  - `save_grid_config(config_dict: dict, filepath: str)`
    - Saves a dict with keys: map, agents_xy, targets_xy, obs_radius,
      max_episode_steps, seed, name, description
    - Format: plain JSON file

  - `load_grid_config(filepath: str) -> GridConfig`
    - Loads the JSON, returns a `GridConfig` instance

  - `load_env_defaults() -> dict`
    - Reads from .env: OBS_RADIUS (default 5), MAX_EPISODE_STEPS (default 256),
      POGEMA_SEED (default 42)
    - Also reads all OPENROUTER_* variables for LLM config

- [ ] 1.2 — Create 2-3 example map config files under `pogema_port/maps/`:
  - `corridor.json` — narrow corridor, 2 agents head-on (guaranteed conflict)
  - `open_warehouse.json` — 8x8 open grid with some obstacles, 3 agents
  - `bottleneck.json` — map with a chokepoint forcing negotiation

  Each JSON should contain:
  ```json
  {
    "name": "corridor",
    "description": "Narrow corridor forcing head-on conflict",
    "map": ["........", ".######.", "........"],
    "agents_xy": [[0, 0], [0, 7]],
    "targets_xy": [[0, 7], [0, 0]],
    "obs_radius": 5,
    "max_episode_steps": 128,
    "seed": 42
  }
  ```

- [ ] 1.3 — Coordinate helper in `config.py`:
  - `xy_to_rc(x, y) -> (row, col)` — converts old (x,y) to POGEMA (row,col)
  - `rc_to_xy(row, col) -> (x, y)` — converts POGEMA (row,col) to (x,y)
  - All new code should use (row, col) natively; these are only for
    reference/debugging if needed

- [ ] 1.4 — Write test: load `corridor.json` → create GridConfig → create env →
  `env.reset()` → verify agents at expected positions

---

## Stage 2: Port A* Pathfinder to Work with POGEMA Grid

Goal: Reuse SimplePathfinder from src/navigation/__init__.py but adapted to
POGEMA's (row, col) coordinates and action space.

Current A* uses 4-directional movement + Manhattan distance heuristic, which
maps directly to POGEMA's 5-action space (idle + 4 cardinal).

- [ ] 2.1 — Create `pogema_port/pathfinding/astar.py`
  - Copy `SimplePathfinder` class from src/navigation/__init__.py
  - Modify to use (row, col) internally instead of (x, y)
  - Keep: `find_path()`, `find_path_with_obstacles()`, `find_path_avoiding_agents()`
  - Rename map_width/map_height to num_rows/num_cols for clarity

- [ ] 2.2 — Add action converter in same file:
  - `path_to_pogema_actions(path: List[Tuple[int,int]]) -> List[int]`
    Converts a sequence of (row, col) positions into POGEMA action integers:
      Same position → 0 (idle/stay)
      Row decreases → 1 (up)
      Row increases → 2 (down)
      Col decreases → 3 (left)
      Col increases → 4 (right)

  - `pogema_action_to_delta(action: int) -> Tuple[int,int]`
    Returns the (d_row, d_col) for a given action int.
    Useful for predicting next position.

- [ ] 2.3 — Create `pogema_port/negotiation/conflict_detector.py`
  - Copy `ConflictDetector` class from src/navigation/__init__.py
  - Adapt to (row, col) format
  - Keep `detect_path_conflicts()` and `_detect_swap_conflicts()`

- [ ] 2.4 — Write test:
  - Create POGEMA env with `corridor.json`
  - Extract full grid from GridConfig
  - Run A* from agent 0 start to agent 0 target
  - Convert path → actions
  - Verify actions are valid (all in range 0-4)
  - Verify path ends at target

---

## Stage 3: Port LLM Negotiation Layer

Goal: Bring over the LLM modules with minimal changes. They are decoupled
from the environment. Use mock/fake OpenRouter calls for smoke tests since
.env API keys are not available during development.

- [ ] 3.1 — Copy files into `pogema_port/negotiation/`:
  - src/llm/__init__.py → pogema_port/negotiation/openrouter_client.py
    (extract OpenRouterClient class)
  - src/llm/openrouter_config.py → pogema_port/negotiation/openrouter_config.py
    (as-is)
  - src/llm/central_negotiator.py → pogema_port/negotiation/central_negotiator.py
  - src/llm/agent_validator.py → pogema_port/negotiation/agent_validator.py

- [ ] 3.2 — Fix all import paths:
  - `from ..llm import OpenRouterClient` → relative imports within pogema_port/negotiation/
  - `from ..llm.agent_validator import` → local imports
  - `from .openrouter_config import OpenRouterConfig` etc.

- [ ] 3.3 — Update CentralNegotiator prompts:
  - Remove all references to "boxes", "picking up", "delivering"
  - Replace with "navigating to target position"
  - Update coordinate descriptions: clarify (row, col) format in prompts
  - Keep the refinement loop (max 5 iterations) as-is
  - Keep spatial hints toggle for future wiggle space implementation

- [ ] 3.4 — Update AgentValidator:
  - Remove box-related validation
  - Validate proposed moves are within POGEMA grid bounds
  - Validate proposed positions are not walls
  - Keep the LLM-based validation flow otherwise unchanged

- [ ] 3.5 — Smoke test with FAKE OpenRouter calls:
  - Create `pogema_port/tests/test_negotiation_mock.py`
  - Mock `OpenRouterClient.send_request()` to return a canned JSON response
    (valid negotiation resolution with paths for 2 agents)
  - Instantiate CentralNegotiator with the mock
  - Pass dummy conflict data (2 agents, head-on conflict at (1, 4))
  - Verify it returns a resolution dict with agent_actions
  - Verify the returned paths are lists of (row, col) tuples
  - NO real API calls — test must pass without .env file

---

## Stage 4: Agent Controller (The Core Integration)

Goal: Build the bridge between POGEMA's per-step API and multi-step
A* + LLM negotiation. This is the most important stage.

POGEMA accepts exactly ONE action per agent per step. Negotiation produces
multi-step paths. The controller must:
1. Compute A* paths for all agents
2. Detect conflicts
3. If conflicts → call CentralNegotiator → get resolution paths
4. Feed resolved paths ONE STEP AT A TIME into POGEMA

- [ ] 4.1 — Create `pogema_port/agent_controller.py` with class:

  ```python
  class LLMNegotiationController:
      def __init__(self, grid_map, agents_xy, targets_xy, num_agents):
          self.pathfinder = SimplePathfinder(...)
          self.conflict_detector = ConflictDetector(...)
          self.negotiator = CentralNegotiator()
          self.agent_validators = {i: AgentValidator() for i in range(num_agents)}
          self.action_queues = {i: deque() for i in range(num_agents)}
          self.agent_positions = list(agents_xy)  # track current (row,col)
          self.agent_targets = list(targets_xy)
          self.agent_done = [False] * num_agents
          self.walls = set()  # extracted from grid_map
          self.grid_map = grid_map
          # ... extract walls from grid_map

      def get_actions(self, observations, terminated, step_number) -> List[int]:
          """
          Called once per POGEMA step.
          Returns list of actions, one per agent.
          """
          # 1. Mark done agents (terminated[i] == True)
          # 2. For done agents: action = 0 (idle)
          # 3. If any active agent has empty queue → trigger replanning:
          #    a. Update agent_positions from last known state
          #    b. Run A* for each active agent (using global grid)
          #    c. Run conflict detection on all paths
          #    d. If conflicts:
          #       - Build conflict_data dict
          #       - Call self.negotiator.negotiate_path_conflict(conflict_data)
          #       - Convert resolved paths → action sequences → fill queues
          #    e. If no conflicts:
          #       - Convert A* paths → action sequences → fill queues
          # 4. Pop first action from each agent's queue
          # 5. Return list of actions

      def update_positions(self, observations_or_env):
          """Update internal position tracking after POGEMA step."""
          # Option A: Use env.get_agents_xy() if available
          # Option B: Predict from last position + action taken
          # Option C: Parse from observations (partial — less reliable)
  ```

- [ ] 4.2 — Implement action queue system:
  - Each agent gets a `collections.deque` of pending POGEMA actions (ints 0-4)
  - After A* or negotiation: convert path → actions via `path_to_pogema_actions()`
  - Store in queue
  - Each POGEMA step: `popleft()` one action per agent
  - If queue empty for an active agent → triggers replanning for ALL active agents

- [ ] 4.3 — Implement replanning triggers:
  - All queues empty (normal replanning)
  - Agent position doesn't match expected position after step (POGEMA collision)
    - If mismatch detected: clear ALL queues, force full replan
  - A negotiated action failed (agent stayed in place unexpectedly)

- [ ] 4.4 — Handle agent completion:
  - When POGEMA terminated[i] == True → set agent_done[i] = True
  - Done agents always get action 0 (idle)
  - Remove done agents from conflict detection and pathfinding

- [ ] 4.5 — Position tracking strategy:
  - Primary: use `env.get_agents_xy()` after each step (if available in pogema API)
  - Fallback: maintain positions internally by applying action deltas
  - After each step, call `controller.update_positions(env)` to sync state

- [ ] 4.6 — Write test with mock negotiator:
  - Create env with `corridor.json` (2 agents, guaranteed head-on conflict)
  - Create LLMNegotiationController with mocked negotiator
  - Run 5 steps manually, verify:
    - Actions are valid ints 0-4
    - Conflict detection triggers on head-on paths
    - Negotiator is called when conflicts detected
    - Action queues are populated from negotiation result

---

## Stage 5: Main Simulation Loop

Goal: Wire everything together into a runnable main.py.

- [ ] 5.1 — Create `pogema_port/main.py`:
  ```python
  def run_pogema_simulation(map_path: str, render: bool = True):
      # 1. Load .env (OpenRouter config)
      # 2. Load map config JSON → create GridConfig
      # 3. Create POGEMA env: env = pogema_v0(grid_config=config)
      # 4. Extract grid_map for global pathfinding
      # 5. Create LLMNegotiationController
      # 6. obs, info = env.reset()
      # 7. Main loop:
      #    actions = controller.get_actions(obs, terminated, step)
      #    obs, rewards, terminated, truncated, info = env.step(actions)
      #    controller.update_positions(env)
      #    if render: env.render()
      #    if all(terminated) or all(truncated): break
      # 8. Print final metrics (from POGEMA info dict)
  ```

- [ ] 5.2 — Add CLI argument parsing:
  - `--map` : path to map config JSON (default: pogema_port/maps/corridor.json)
  - `--obs-radius` : observation radius override
  - `--max-steps` : max episode steps override
  - `--no-render` : disable POGEMA rendering
  - `--seed` : random seed override

- [ ] 5.3 — Add .env loading at startup for OpenRouter config

- [ ] 5.4 — Add POGEMA-native metric printing at end:
  - Steps taken
  - Which agents reached target (from terminated flags)
  - Number of LLM negotiations triggered (from controller stats)
  - Token usage (from OpenRouterClient.get_token_usage())

- [ ] 5.5 — Test end-to-end:
  - Run with `corridor.json` → verify agents navigate
  - Run with `open_warehouse.json` → verify multi-agent works
  - Run with `bottleneck.json` → verify negotiation triggers at chokepoint

---

## Stage 6: Metrics & Logging (POGEMA-Native)

Goal: Rely entirely on POGEMA + pogema-toolbox for metrics, logging, and
visualization. Trash the old UnifiedLogger. Only add a thin wrapper to
capture LLM-specific data that POGEMA doesn't track natively.

- [ ] 6.1 — Install and configure `pogema-toolbox` (already in requirements.txt)

- [ ] 6.2 — Add LLM stats tracker in `agent_controller.py`:
  - Track: number of negotiations, total tokens used, negotiation durations
  - Expose via `controller.get_llm_stats() -> dict`
  - This is the ONLY custom logging needed — everything else comes from POGEMA

- [ ] 6.3 — In main.py, after simulation ends:
  - Print POGEMA episode metrics (from info dict / env API)
  - Print LLM stats from controller
  - Optionally save combined stats to a JSON file

- [ ] 6.4 — Use POGEMA's built-in `env.render()` for visualization during runs
  - No custom rendering code needed
  - For saved animations: use pogema-toolbox's animation export if needed later

---

## Stage 7: Verification & Benchmarking Readiness

- [ ] 7.1 — Manual test: Verify agents navigate and negotiate correctly on all example maps.
- [ ] 7.2 — **Implement `pogema-toolbox` Algorithm Interface**:
  - Create `pogema_port/integration.py` to wrap the controller.
  - Implement `act(observations) -> List[int]` (batch-compatible).
  - Implement `reset(observations)` to clear internal path/queue states.
  - Register the algorithm: `ToolboxRegistry.register_algorithm('LLMNegotiation', LLMNegotiationAlgorithm)`.
- [ ] 7.3 — **Concurrency & Rate Limiting**:
  - Ensure `openrouter_client.py` handles parallel requests (cooldowns/retries) for `pogema-toolbox` multi-seed runs.
  - Verify thread-safety for the controller during parallel evaluation.
- [ ] 7.4 — **Run Toolbox Benchmark**:
  - Test via CLI: `pogema-toolbox evaluate -a LLMNegotiation -m corridor.json open_warehouse.json`.
  - Record and verify standard toolbox results.

---

## Stage 8: Documentation & Cleanup

- [ ] 8.1 — Create `pogema_port/README.md` with:
  - How to install dependencies
  - How to configure .env (OpenRouter API key, model settings)
  - How to run a simulation (CLI commands with examples)
  - How to create/edit custom maps (GridConfig JSON format)
  - How to run benchmarks with pogema-toolbox

- [ ] 8.2 — Update root README.md to mention the POGEMA port and link to
  pogema_port/README.md

- [ ] 8.3 — Update `.env.example` with new POGEMA-specific variables:
  ```
  # POGEMA Configuration
  OBS_RADIUS=5
  MAX_EPISODE_STEPS=256
  POGEMA_SEED=42
  ```

- [ ] 8.4 — Clean up: ensure no dead imports referencing old src/ modules
  in the pogema_port/ directory

---

## Quick Reference: What Gets Reused vs. Replaced vs. Trashed

| Component                          | Status        | Notes                                           |
|------------------------------------|---------------|-------------------------------------------------|
| src/simulation/game_engine.py      | ❌ TRASHED    | Replaced by POGEMA env + agent_controller.py    |
| src/map_generator/ (all files)     | ❌ TRASHED    | Replaced by POGEMA GridConfig + config.py       |
| src/navigation/SimplePathfinder    | ✅ PORTED     | A* ported to (row,col), lives in pathfinding/   |
| src/navigation/ConflictDetector    | ✅ PORTED     | Adapted to (row,col), lives in negotiation/     |
| src/agents/__init__.py (RobotAgent)| ❌ TRASHED    | Replaced by LLMNegotiationController            |
| src/llm/__init__.py (OpenRouter)   | ✅ REUSED     | Copied as-is to negotiation/openrouter_client   |
| src/llm/openrouter_config.py       | ✅ REUSED     | Copied as-is                                    |
| src/llm/central_negotiator.py      | ✅ REUSED     | Prompts updated (no boxes), coords updated      |
| src/llm/agent_validator.py         | ✅ REUSED     | Box validation removed, bounds check updated    |
| src/logging/ (UnifiedLogger)       | ❌ TRASHED    | POGEMA + pogema-toolbox handles metrics/logging  |
| visualization.py                   | ❌ TRASHED    | POGEMA has built-in rendering                   |
| src/tools/layout_editor            | ❌ TRASHED    | Maps are now simple JSON GridConfig files        |
| benchmark_tool.py                  | ❌ TRASHED    | pogema-toolbox provides benchmarking             |

## Action Space Reference (POGEMA)

| Value | Action | Delta (row, col) |
|-------|--------|-------------------|
| 0     | idle   | (0, 0)            |
| 1     | up     | (-1, 0)           |
| 2     | down   | (+1, 0)           |
| 3     | left   | (0, -1)           |
| 4     | right  | (0, +1)           |