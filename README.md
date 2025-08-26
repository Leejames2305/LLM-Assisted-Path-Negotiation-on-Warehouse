# LLM-Assisted Multi-Robot Navigation Negotiation in Warehouse Environments

This project implements a hybrid multi-agent system (HMAS-2) where multiple robots navigate a warehouse environment using Large Language Models for conflict resolution and path planning.

## 🚀 Features

- **Random Warehouse Map Generation**: Creates 8x6 warehouse maps with agents, boxes, targets, and obstacles
- **HMAS-2 Framework**: Hybrid system with:
  - Central LLM (GLM-4.5-Air) for complex conflict negotiation
  - Agent LLMs (Gemma-2-9B) for action validation
- **Real-time Conflict Detection**: Identifies path conflicts and triggers LLM negotiation
- **Interactive Terminal Interface**: Step-by-step simulation with colored display
- **Comprehensive Logging**: Saves detailed simulation logs for analysis

## 🛠️ Installation

1. **Clone the repository**:
   ```cmd
   git clone <your-repo-url>
   cd CodeBase
   ```

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

Before running the main simulation, you can verify everything works with these smoke tests:

### Quick Structure Test (No Dependencies)
```cmd
python quick_test.py
```
Verifies project structure and file syntax without requiring any external packages.

### Simple Demo Test (Basic Dependencies)
```cmd
python demo_test.py
```
Runs a comprehensive offline demonstration of core functionality:
- Map generation with visual display
- Agent creation and pathfinding  
- Conflict detection
- Basic simulation logic
- State logging

### Full Smoke Test (All Features)
```cmd
python smoke_test.py
```
Tests all components including LLM fallback functionality (works without API keys).

**💡 Tip**: Start with `demo_test.py` to see the system in action immediately!

## 🎮 Usage

Run the simulation:
```cmd
python main.py
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
│   ├── agents/              # Robot agent implementation
│   ├── llm/                 # LLM clients and negotiation
│   ├── navigation/          # Pathfinding and conflict detection
│   └── simulation/          # Main game engine
├── logs/                    # Simulation logs (auto-generated)
├── main.py                  # Entry point
├── requirements.txt         # Dependencies
├── .env.example            # Environment template
└── README.md               # This file
```

## 🤖 LLM Configuration

### Central Negotiator
- **Model**: `zai/glm-4.5-air:free`
- **Purpose**: Complex conflict resolution and strategic planning
- **Temperature**: 0.3 (consistent reasoning)

### Agent Validators
- **Model**: `google/gemma-3n-e4b-it:free`
- **Purpose**: Quick action validation and safety checks
- **Temperature**: 0.1 (very consistent validation)

## 🗺️ Map Symbols

- `A`: Agent (robot)
- `A*`: Agent carrying a box
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

## 🔧 Configuration

Environment variables in `.env`:
```env
# OpenRouter API Configuration
OPENROUTER_API_KEY=your_key_here

# LLM Models
CENTRAL_LLM_MODEL=zai/glm-4.5-air:free
AGENT_LLM_MODEL=google/gemma-2-9b-it:free

# Simulation Settings
MAP_WIDTH=8
MAP_HEIGHT=6
MIN_AGENTS=2
MAX_AGENTS=4
LOG_SIMULATION=true
```

## 🚨 Troubleshooting

### Import Errors
- Install all requirements: `pip install -r requirements.txt`
- Ensure Python 3.8+ is installed

### Map Display Issues
- Windows: Install colorama for colored output

## 📈 Future Enhancements

- [ ] Box pickup/delivery mechanics
- [ ] More complex warehouse layouts
- [ ] Performance metrics and statistics
- [ ] Web interface for visualization
- [ ] Multi-warehouse scenarios
- [ ] Custom map designer


