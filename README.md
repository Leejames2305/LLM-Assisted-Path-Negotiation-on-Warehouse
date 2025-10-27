# LLM-Assisted Multi-Robot Navigation Negotiation in Warehouse Environments

This project implements a hybrid multi-agent system (HMAS-2) where multiple robots navigate a warehouse environment using Large Language Models for conflict resolution and path planning.

## 🚀 Features

- **User-Defined Layout System**: Create and edit custom warehouse layouts instead of random generation
  - Interactive terminal-based layout editor
  - Pre-built layouts (empty, S-shaped, tunnels, bridges)
  - Comprehensive validation (bounds, reachability, overlaps)
  - JSON-based layout storage for reproducible testing
- **HMAS-2 Framework**: Hybrid system with:
  - Central LLM (State-Of-The-Art, SOTA Model) for complex conflict negotiation
  - Agent LLMs (Smaller model) for action validation
- **Real-time Conflict Detection**: Identifies path conflicts and triggers LLM negotiation
- **Interactive Terminal Interface**: Step-by-step simulation with colored display
- **Comprehensive Logging**: Saves detailed simulation logs for analysis

## 🛠️ Installation

1. **Clone the repository**:

2. **Install dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   Edit `.env` and add your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_actual_api_key_here
   ```

## 🔧 Configuration

Environment variables in `.env.example`:
```env
# OpenRouter API Configuration
OPENROUTER_API_KEY=your_key_here

# Advanced OpenRouter Features
...

# LLM Models
CENTRAL_LLM_MODEL=zai/glm-4.5-air:free
AGENT_LLM_MODEL=google/gemma-3n-e4b-it:free

...
```

## 🎮 Usage

Run the simulation:
```cmd
python main.py
```

Run the simulation, but forced negotiation:
```cmd
python test_negotiation.py
```

### Interactive Commands:
- **Enter**: Execute next simulation step
- **auto**: Switch to automatic mode (1-second intervals)
- **q**: Quit simulation

## 🗺️ Map Symbols

- `A`: Agent (robot)
- `@`: Agent carrying a box
- `B`: Box
- `T`: Target location
- `#`: Wall/Obstacle
- `.`: Empty space

## 🎛️ Layout Management

### Create or Edit Layouts

Launch the interactive layout editor:
```cmd
python -m src.tools.layout_editor
```

**Editor Commands:**
```
w <x> <y>           - Toggle wall at position
a <id> <x> <y>      - Place/move agent
b <id> <x> <y>      - Place/move box
t <id> <x> <y>      - Place/move target
goal <agent> <tgt>  - Set agent goal to target
rand a <count>      - Random agent placement
rand b <count>      - Random box placement
rand t <count>      - Random target placement
clear               - Clear all entities (keep walls)
info                - Show layout info
validate            - Check for errors
save                - Save layout
back                - Return to main menu
```

### Layout Constraints

```
Width:   5-50 cells
Height:  5-50 cells
Agents:  1-10 per layout
Boxes:   1-20 per layout
Targets: 1-20 per layout
```

### Validation Checks

All layouts are automatically validated for:
- ✅ Bounds checking (all entities within grid)
- ✅ Wall placement (no entities on walls)
- ✅ Overlap detection (no duplicate positions)
- ✅ Goal validity (targets exist, agents assigned)
- ✅ Reachability (agents can reach targets via BFS)

## 🏗️ Project Structure

```
CodeBase/
├── src/
│   ├── map_generator/        # Layout management and validation
│   │   ├── constants.py      # Schema, constraints, messages
│   │   ├── layout_validator.py
│   │   ├── layout_manager.py
│   │   └── layout_selector.py
│   ├── tools/
│   │   └── layout_editor.py  # Interactive layout editor
│   ├── agents/               # Robot agent implementation
│   ├── llm/                  # LLM clients and negotiation
│   ├── navigation/           # Pathfinding and conflict detection
│   └── simulation/           # Main game engine
│
├── layouts/
│   ├── prebuilt/             # Official layouts
│   │   ├── empty.json        # Basic 8x6 testing layout
│   │   ├── s_shaped.json     # S-corridor negotiation scenario
│   │   ├── tunnel.json       # Multi-tunnel layout
│   │   └── bridge.json       # Bridge chamber layout
│   └── custom/               # User-created layouts
│
├── logs/                     # Simulation logs (auto-generated)
│
├── tests/                    # All tests file
│   ├── smoke_test.py         # Test all components
│   └── test_....py           # Miscellaneous test files
│
├── main.py                   # Entry point with layout selection
├── test_negotiation.py       # Entry with layout selection & testing
├── requirements.txt          # Dependencies
├── .env.example              # Environment template
└── README.md
```

## 🤖 LLM Configuration

### Central Negotiator
- **Model (SOTA)**: `zai/glm-4.5-air:free`
- **Purpose**: Complex conflict resolution and strategic planning
- **Temperature**: 0.3 (consistent reasoning)

### Agent Validators
- **Model (Smaller Model)**: `google/gemma-3n-e4b-it:free`
- **Purpose**: Quick action validation and safety checks
- **Temperature**: 0.1 (very consistent validation)



## 📊 Simulation Features

### Conflict Resolution
The system detects conflicts when:
- Multiple agents plan to occupy the same cell
- Agents attempt to swap positions (crossing paths)

When conflicts occur:
1. Central LLM analyzes the situation
2. Proposes resolution strategy (priority/reroute/wait)
3. Agent LLMs validate proposed actions
4. Actions are executed with safety checks

### Logging
Detailed logs are saved in `/logs`:
- Turn-by-turn map states
- Agent positions and statuses
- Conflict resolution decisions
- LLM reasoning processes

### Visualise
Run the visualization on logged files:
```cmd
python visualization.py logs/simulation_logs....
```
```cmd
python visualization.py logs/negotiation_simulation_logs....
```


## 🚨 Troubleshooting

### Import Errors
- Install all requirements: `pip install -r requirements.txt`
- Ensure Python 3.8+ is installed

### Map Display Issues
- Windows: Install colorama for colored output
