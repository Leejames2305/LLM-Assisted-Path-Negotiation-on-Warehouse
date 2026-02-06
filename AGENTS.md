# AGENTS Guidelines for This Repository

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

## Writing comments
When writing comments in the codebase, please follow these guidelines:
- Use simple sentences to explain the purpose of the functions.
- DO NOT over-comment any function, such as listing down all input and output variables, those are NOT NEEDED.
- For complex logic, provide a brief explanation of the approach taken.
- For single line comments, use `#` followed by a space before the comment text.
- For multi-line comments, use triple quotes `"""` to enclose the comment block.

## Writing documentation
When writing documentation for the project, please adhere to the following guidelines:
- Keep most documentation in the `docs/` directory, unless specified otherwise.
- Keep it concise and to the point.
- Use bullet points or numbered lists for clarity.
- AVOID large paragraphs of text.
- Include code snippets or examples where applicable.

## Writing commit messages
When writing commit messages, please follow these conventions:
- Use the present tense (e.g., "Add feature" instead of "Added feature").
- Follow the Conventional Commits format: `<type>(<scope>): <subject>`
  - `<type>`: feat, fix, docs, style, refactor, test, chore
  - `<scope>`: optional, indicates the area of the code affected
  - `<subject>`: brief description of the change
- Add yourself as co-author for commits generated with its assistance:
  - `Co-authored-by: whoyouare <agent@example.com>`

## PR instructions
- Title format: `[feat/bugfix/etc.] <Title...>`
- Description should include:
  - Summary of changes made.
  - Any dependencies added or removed.
  - Instructions for testing the changes.
  - Linked issues or tickets.