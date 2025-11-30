
## 🗺️ Map Symbols

- `A`: Agent (robot)
- `@`: Agent carrying a box
- `B`: Box
- `T`: Target location
- `#`: Wall/Obstacle
- `.`: Empty space

## 🎛️ Layout Management


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
├── docs/                     # Documents
├── src/
│   ├── map_generator/        # Layout management and validation
│   ├── tools/
│   │   └── layout_editor.py  # Interactive layout editor
│   ├── agents/               # Robot agent implementation
│   ├── llm/                  # LLM clients and negotiation
│   ├── navigation/           # Pathfinding and conflict detection
│   └── simulation/           # Main game engine
│   └── logging/              # Unified logging
│
├── layouts/
│   ├── prebuilt/             # Official layouts
│   └── custom/               # User-created layouts
│
├── logs/                     # Simulation logs 
│
├── tests/                    # All tests file
│
├── main.py                   # Entry point with layout selection
│
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

## 🚨 Troubleshooting

### Import Errors
- Install all requirements: `pip install -r requirements.txt`
- Ensure Python 3.8+ is installed