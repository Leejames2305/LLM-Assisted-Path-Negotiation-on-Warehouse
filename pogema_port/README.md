# POGEMA Port — LLM-Assisted Path Negotiation

A port of the LLM-Assisted Path Negotiation warehouse simulation to the [POGEMA](https://github.com/AIRI-Institute/pogema) multi-agent pathfinding environment.

## Directory Structure

```
pogema_port/
├── __init__.py
├── main.py                    # POGEMA entrypoint & CLI
├── config.py                  # GridConfig builder, map save/load, coordinate helpers
├── agent_controller.py        # LLMNegotiationController (core integration)
├── integration.py             # pogema-toolbox Algorithm Interface
├── requirements.txt           # Dependencies
├── smoke_test.py              # Basic POGEMA smoke test
├── maps/
│   ├── corridor.json          # Narrow corridor forcing head-on conflict
│   ├── open_warehouse.json    # 8×8 open warehouse, 3 agents
│   └── bottleneck.json        # Chokepoint forcing negotiation
├── pathfinding/
│   └── astar.py               # A* pathfinder (row,col based) + action converters
├── negotiation/
│   ├── conflict_detector.py   # Conflict detection for POGEMA grid
│   ├── central_negotiator.py  # LLM central negotiator (updated prompts)
│   ├── agent_validator.py     # LLM action validator
│   ├── openrouter_client.py   # OpenRouter API client
│   └── openrouter_config.py   # OpenRouter configuration helpers
└── tests/
    └── test_negotiation_mock.py  # Mock tests (no API key needed)
```

## Installation

```bash
pip install pogema>=1.4.0 requests>=2.31.0 python-dotenv>=1.0.0 numpy>=1.24.0 colorama>=0.4.6
```

> ⚠️ **Note:** `pogema-toolbox` is currently incompatible with Python 3.12 due to a dependency issue. Install manually when the upstream fix is available.

## Configuration

Copy `.env.example` to `.env` and fill in your settings:

```bash
cp .env.example .env
```

Key variables:

```
# OpenRouter (required for LLM negotiation)
OPENROUTER_API_KEY=your_key_here
CENTRAL_LLM_MODEL=zai/glm-4.5-air:free
AGENT_LLM_MODEL=nvidia/nemotron-3-nano-30b-a3b:free

# POGEMA settings
OBS_RADIUS=5
MAX_EPISODE_STEPS=256
POGEMA_SEED=42
```

## Running a Simulation

```bash
# Run with default corridor map (requires OPENROUTER_API_KEY)
python -m pogema_port.main

# Run with a specific map
python -m pogema_port.main --map pogema_port/maps/open_warehouse.json

# Run without LLM negotiation (A* only, no API key needed)
python -m pogema_port.main --map pogema_port/maps/open_warehouse.json --no-negotiate

# Override settings
python -m pogema_port.main --map pogema_port/maps/corridor.json --max-steps 256 --seed 7

# Save metrics to JSON
python -m pogema_port.main --map pogema_port/maps/corridor.json --save-metrics results.json
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--map PATH` | Path to map config JSON |
| `--obs-radius N` | Override observation radius |
| `--max-steps N` | Override max episode steps |
| `--seed N` | Override random seed |
| `--no-render` | Disable POGEMA rendering |
| `--no-negotiate` | Disable LLM negotiation (A* only) |
| `--no-spatial-hints` | Disable spatial hints in negotiation |
| `--save-metrics FILE` | Save final metrics to JSON file |

## Map Format

Maps are stored as JSON files. Example:

```json
{
  "name": "corridor",
  "description": "Narrow corridor forcing head-on conflict",
  "map": [
    "........",
    ".######.",
    "........"
  ],
  "agents_xy": [[1, 0], [1, 7]],
  "targets_xy": [[1, 7], [1, 0]],
  "obs_radius": 5,
  "max_episode_steps": 128,
  "seed": 42
}
```

- `map`: List of strings, `'.'` = free, `'#'` = obstacle
- `agents_xy`: Starting positions as `[row, col]`
- `targets_xy`: Goal positions as `[row, col]` (same order as agents)

### Coordinate Convention

POGEMA uses `(row, col)` coordinates where `(0, 0)` is the **top-left**.

```python
from pogema_port.config import xy_to_rc, rc_to_xy

# Convert legacy (x, y) → POGEMA (row, col)
row, col = xy_to_rc(x=3, y=1)   # → (1, 3)

# Convert POGEMA (row, col) → legacy (x, y)
x, y = rc_to_xy(row=1, col=3)   # → (3, 1)
```

## Running Tests (No API Key Needed)

```bash
python -m pogema_port.tests.test_negotiation_mock
```

## Benchmarking (pogema-toolbox)

Once `pogema-toolbox` is compatible with your Python version:

```python
from pogema_port.integration import register_with_toolbox
register_with_toolbox()

# Then run via CLI:
# pogema-toolbox evaluate -a LLMNegotiation -m corridor.json open_warehouse.json
```

## Action Space Reference

| Value | Action | Delta (row, col) |
|-------|--------|-------------------|
| 0 | idle | (0, 0) |
| 1 | up | (−1, 0) |
| 2 | down | (+1, 0) |
| 3 | left | (0, −1) |
| 4 | right | (0, +1) |
