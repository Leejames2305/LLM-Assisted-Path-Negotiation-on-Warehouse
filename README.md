# LLM-Assisted Multi-Robot Navigation Negotiation in Warehouse Environments

This project proposes the use of MAPF + LLM (Hybrid-Feedback setup) to resolve path conflict issues in a simulated warehouse

## 🚀 Features
- **Pluggable MAPF Backend**: Choose either A* or LNS2 as MAPF algorithm
- **Real-time Conflict Detection**: Identifies path conflicts and triggers LLM negotiation
- **LLM Negotiation Framework (Hybrid)**: LLM reasoning negotiation with feedback
  - Central LLM (SOTA Model) for complex conflict negotiation
  - Agent LLMs (Smaller model) for action validation
- **Custom Layout System**: Create and edit custom warehouse layouts
  - Terminal-based layout editor
  - Self-validation (bounds, reachability, overlaps)
- **Logging**: Saves detailed simulation logs for analysis

## 🛠️ Installation

1. **Clone the repository**:

2. **Install dependencies**:

    a) Typical python setup: `pip install -r requirements.txt` OR
    
    b) UV setup: `uv init` then `uv add -r requirement.txt`

3. **Configure environment**:
   Rename `.env.example` to `.env` and add your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_actual_api_key_here
   ```
4. **(Optional) Edit .env file**: For additional configuration such as LLM models


## 🎮 Usage

Run the simulation:

  - `python main.py` OR `uv run main.py`

Run the benchmark tool:
  - `python benchmark_tool.py` OR `uv run benchmark_tool.py`

Run the map editor:
  - `python -m src.tools.layout_editor` OR `uv run python -m src.tools.layout_editor`

Run the visualization on logged files:
  - `python visualization.py logs/sim_log....json` OR `uv run visualization.py logs/sim_log....json`


## 📄 Docs
Additional documentations are located at [here](docs/Basics.md)
