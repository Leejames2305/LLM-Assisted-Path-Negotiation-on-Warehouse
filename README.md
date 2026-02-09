# LLM-Assisted Multi-Robot Navigation Negotiation in Warehouse Environments

This projects propose the use of A* + LLM (Hybrid-Feedback setup) to resolve path conflict issues in a simulated warehouse

## 🚀 Features
- **A-star Path Finding**: Agents uses A* for quick navigation
- **Real-time Conflict Detection**: Identifies path conflicts and triggers LLM negotiation
- **LLM Negotiation Framework (Hybrid)**: LLM reasoning negotiation with feedback
  - Central LLM (SOTA Model) for complex conflict negotiation
  - Agent LLMs (Smaller model) for action validation
- **Custom Layout System**: Create and edit custom warehouse layouts
  - Terminal-based layout editor
  - Self-validation (bounds, reachability, overlaps)
- **Benchmark Tool**: Comprehensive performance testing
  - Random map generation with customizable dimensions and wall density
  - Automated performance metrics collection
  - CSV and JSON output for analysis
- **Logging**: Saves detailed simulation logs for analysis

## 🛠️ Installation

1. **Clone the repository**:

2. **Install dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   Rename `.env.example` to `.env` and add your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_actual_api_key_here
   ```
4. **(Optional) Edit .env file**: For additional configuration such as LLM models


## 🎮 Usage

Run the simulation:
```cmd
python main.py
```
Run the benchmark tool:
```cmd
python benchmark_tool.py
```
Run the map editor:
```cmd
python -m src.tools.layout_editor
```

Run the visualization on logged files:
```cmd
python visualization.py logs/sim_log....
```

## 📄 Docs
Additional documentations are located at [here](docs/Basics.md)