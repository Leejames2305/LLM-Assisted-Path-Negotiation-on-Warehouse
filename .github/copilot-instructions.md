# Github Copilot Guidelines for This Repository

This repository is a Python-based simulation engine that relies on external LLM provider API (e.g. OpenRouter) to works. 
When working on the project interactively with an agent (e.g. the Codex CLI), please follow the guidelines below.

## Initial Setup
To set up the development environment for the first time, follow these steps:
1. (Optional) Create and activate a virtual environment to manage dependencies.
2. Install all necessary dependencies from `requirements.txt`
3. Set up the environment variables in `.env` file, including the OpenRouter API key. Refer to `.env.example` for the required variables.

## Running of the simulation engine
To run the simulation engine locally during agent-assisted development, use the following command in project directory:

| Command                                        | Purpose                                            |
| -----------------------------------------------| -------------------------------------------------- |
| `python main.py`                               | Run the simulation                                 |
| `python -m src.tools.layout_editor`            | Run the map editor tools                           |
| `python visualization.py logs/sim_log....`     | Run the visualization tools on logged files        |

## Testing instructions
Since the project uses OpenRouter API calls when the simulation is running, it will takes time to get response from the API. Wait for the response before proceeding to the next step in the simulation.

This project doesn't have automated tests yet. When making changes, please manually test the following scenarios:
* Always use `main.py` to run the main simulation engine.
* Load up `s_shaped` map to forcefully trigger LLM API calls.
* Checks for basic functionalities, including:
  - Agent navigation and pathfinding.
  - Game engine able to end without errors.
  - Data logging capabilities, saved in the `logs/` directory.
  - Visualization tools, using `visualization.py` on the generated log files.

## PR instructions
- Title format: `[feat/bugfix/etc.] <Title...>`
- Description should include:
  - Summary of changes made.
  - Any dependencies added or removed.
  - Instructions for testing the changes.
  - Linked issues or tickets.