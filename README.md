# LLM-Assisted Multi-Robot Navigation Negotiation in Warehouse Environments

This project implements a hybrid multi-agent system (HMAS-2) where multiple robots navigate a warehouse environment using Large Language Models for conflict resolution and path planning.

## 🚀 Features

- **Random Warehouse Map Generation**: Creates 8x6 warehouse maps with agents, boxes, targets, and obstacles
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

## 🧪 Testing & Verification


### Full Smoke Test (All Features)
```cmd
python tests/smoke_test.py
```
Tests all components including LLM fallback functionality (works without API keys).

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

## 🏗️ Project Structure

```
CodeBase/
├── src/
│   ├── map_generator/        # Warehouse map generation
│   ├── agents/               # Robot agent implementation
│   ├── llm/                  # LLM clients and negotiation
│   ├── navigation/           # Pathfinding and conflict detection
│   └── simulation/           # Main game engine
│
├── logs/                     # Simulation logs (auto-generated)
│
├── tests/                    # All tests file
│   ├── smoke_test.py         # Test all components
│   └── test_....py           # Miscellaneous test files
│
├── main.py                   # Entry point
├── test_negotiation.py       # Entry with FORCED negotiation  
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

## 🗺️ Map Symbols

- `A`: Agent (robot)
- `@`: Agent carrying a box
- `B`: Box
- `T`: Target location
- `#`: Wall/Obstacle
- `.`: Empty space

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

# Simulation Settings
MAP_WIDTH=8
MAP_HEIGHT=6
...
```

## 🚨 Troubleshooting

### Import Errors
- Install all requirements: `pip install -r requirements.txt`
- Ensure Python 3.8+ is installed

### Map Display Issues
- Windows: Install colorama for colored output
