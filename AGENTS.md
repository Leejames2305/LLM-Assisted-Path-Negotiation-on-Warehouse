# AGENTS Guidelines for This Repository

This repository is a Python-based multi-agent path negotiation simulation using the POGEMA environment and OpenRouter LLM API.

## Initial Setup
1. (Optional) Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your OpenRouter API key.

## Running the Simulation

| Command | Purpose |
|---------|---------|
| `python main.py` | Run the simulation (LLM negotiation, requires API key) |
| `python main.py --no-negotiate --no-render` | Run A* only (no API key needed) |
| `python main.py --map maps/open_warehouse.json` | Run with a specific map |

## Testing

```bash
python -m tests.test_negotiation_mock
```

No API key is required for the mock tests.

## Writing Comments
- Use simple sentences to explain the purpose of functions.
- DO NOT over-comment (no need to list all input/output variables).
- Single-line comments: `# comment text`
- Multi-line comments: `"""comment block"""`

## Writing Documentation
- Keep documentation in the `docs/` directory.
- Keep it concise and to the point.
- Use bullet points or numbered lists.
- Include code snippets where applicable.

## Writing Commit Messages
- Use present tense ("Add feature" not "Added feature").
- Format: `<type>(<scope>): <subject>`
  - Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- Add yourself as co-author: `Co-authored-by: whoyouare <agent@example.com>`

## PR Instructions
- Title: `[feat/bugfix/etc.] <Title>`
- Description: summary, dependencies changed, test instructions, linked issues.
