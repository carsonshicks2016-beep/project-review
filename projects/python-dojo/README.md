# Python Dojo

Python Dojo is a local, autonomous Python learning platform. It teaches from absolute beginner material through professional Python patterns with deterministic randomized exercises, safe local grading, adaptive progress, and both browser and terminal clients.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python-dojo web --port 8797
```

Then open `http://127.0.0.1:8797`.

For the terminal dojo:

```bash
python-dojo term
```

## What Makes It Autonomous

- Exercises are generated from audited templates with random seeds.
- Answers and tests are hidden from the browser client.
- Code is checked in a short-lived subprocess after AST safety checks.
- Progress is stored locally in SQLite at `~/.pythondojo/dojo.sqlite3`.
- The next topic is chosen from mastery, unlocks, and review needs.
- Ollama is optional and used only for hints/explanations; deterministic lessons and hints work without it.

## Useful Commands

```bash
python -m pytest
python-dojo web --port 8797
python-dojo term
python-dojo health
python-dojo reset-progress
```

The macOS launcher in this folder creates/updates the local virtual environment and lets you choose browser or terminal mode.
