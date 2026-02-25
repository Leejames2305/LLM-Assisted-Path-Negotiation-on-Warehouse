# LLM-Assisted Multi-Robot Path Negotiation (POGEMA)

A* + LLM hybrid multi-agent path negotiation running on the [POGEMA](https://github.com/AIRI-Institute/pogema) environment.

## 🚀 Features
- **A* Pathfinding**: Agents use A* for fast navigation on the POGEMA grid
- **Real-time Conflict Detection**: Identifies vertex and swap conflicts across planned paths
- **LLM Negotiation (Hybrid)**: Iterative LLM-driven conflict resolution
  - Central LLM (SOTA model) for complex multi-agent negotiation
  - Agent LLMs (smaller model) for per-action validation
- **POGEMA Integration**: Uses POGEMA's native env, rendering, and metrics
- **Benchmarking Ready**: `integration.py` exposes a pogema-toolbox compatible interface

## 📁 Project Structure

```
.
├── main.py                    # Simulation entrypoint & CLI
├── config.py                  # GridConfig builder, map save/load, coordinate helpers
├── agent_controller.py        # LLMNegotiationController (core integration)
├── integration.py             # pogema-toolbox Algorithm Interface
├── smoke_test.py              # Basic POGEMA smoke test
├── requirements.txt
├── maps/
│   ├── corridor.json          # Narrow corridor — guaranteed head-on conflict
│   ├── open_warehouse.json    # 8×8 open warehouse, 3 agents
│   └── bottleneck.json        # Chokepoint forcing negotiation
├── pathfinding/
│   └── astar.py               # A* in (row,col) space + POGEMA action converters
├── negotiation/
│   ├── conflict_detector.py   # Conflict detection
│   ├── central_negotiator.py  # LLM central negotiator
│   ├── agent_validator.py     # LLM action validator
│   ├── openrouter_client.py   # OpenRouter API client
│   └── openrouter_config.py   # OpenRouter config helpers
└── tests/
    └── test_negotiation_mock.py  # Mock tests (no API key needed)
```

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

> ⚠️ `pogema-toolbox` is currently incompatible with Python 3.12. Install manually when the upstream fix is available.

## ⚙️ Configuration

Copy `.env.example` to `.env` and add your settings:

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

## 🎮 Running the Simulation

```bash
# Default (corridor map, LLM negotiation — requires OPENROUTER_API_KEY)
python main.py

# Specific map
python main.py --map maps/open_warehouse.json

# A* only — no API key needed
python main.py --map maps/open_warehouse.json --no-negotiate --no-render

# Override settings
python main.py --map maps/corridor.json --max-steps 256 --seed 7

# Save metrics
python main.py --map maps/corridor.json --save-metrics results.json
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--map PATH` | Path to map config JSON (default: `maps/corridor.json`) |
| `--obs-radius N` | Override observation radius |
| `--max-steps N` | Override max episode steps |
| `--seed N` | Override random seed |
| `--no-render` | Disable POGEMA rendering |
| `--no-negotiate` | Disable LLM negotiation (A* only) |
| `--no-spatial-hints` | Disable spatial hints in negotiation |
| `--save-metrics FILE` | Save final metrics to a JSON file |

## 🗺️ Map Format

Maps are stored as JSON files:

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

- `map`: List of strings — `'.'` = free, `'#'` = obstacle
- `agents_xy` / `targets_xy`: Positions as `[row, col]`; `(0,0)` is top-left

## 🧪 Testing (No API Key Needed)

```bash
python -m tests.test_negotiation_mock
```

## 📊 Action Space Reference

| Value | Action | Delta (row, col) |
|-------|--------|-------------------|
| 0 | idle | (0, 0) |
| 1 | up | (−1, 0) |
| 2 | down | (+1, 0) |
| 3 | left | (0, −1) |
| 4 | right | (0, +1) |

## 📄 Docs

- [docs/pogema_port_status.md](docs/pogema_port_status.md) — Implementation status
