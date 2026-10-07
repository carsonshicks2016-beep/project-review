"""Read-only inventory of likely project roots under the configured Python workspace."""
from pathlib import Path

ROOT = Path.home() / "Desktop/untitled folder/01 Core Work/Python"
SKIP = {"node_modules", ".git", ".venv", "venv", "__pycache__", "dist", "build", "Library", "archives", "legacy"}
MARKERS = {"pyproject.toml", "requirements.txt", "setup.py", "setup.cfg", "Pipfile", "main.py", "app.py", "train.py"}

for folder in sorted(ROOT.iterdir(), key=lambda item: item.name.casefold()):
    if not folder.is_dir() or folder.name in SKIP or folder.name.startswith("."):
        continue
    files = {item.name for item in folder.iterdir() if item.is_file()}
    if files & MARKERS or (folder / ".git").exists():
        print(f"{folder.name}\t{folder}")
