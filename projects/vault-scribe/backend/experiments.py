"""
Vault Scribe — Experiment Tracker
Finds active biological/fitness experiments and allows appending logs.
"""

import datetime
from pathlib import Path
from backend.config import VAULT_PATH
from backend.indexer import get_indexer

def get_active_experiments() -> list[dict]:
    """Return a list of all active experiment notes."""
    indexer = get_indexer()
    experiments = []
    
    for title, note_data in indexer.index.notes.items():
        if title.startswith("Experiment: "):
            # Check if it's active by reading the file
            file_path = note_data.filepath
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                if "> Status: Active" in content:
                    # Extract intervention
                    intervention = "Unknown"
                    for line in content.splitlines():
                        if line.startswith("> Intervention: "):
                            intervention = line.replace("> Intervention: ", "").strip()
                            break
                    
                    experiments.append({
                        "title": title,
                        "path": note_data.relative_path,
                        "intervention": intervention
                    })
    
    return experiments


def log_experiment_data(title: str, notes: str, metric: str = "—", subjective: str = "—") -> str:
    """Append a row to the Daily Log table of an experiment."""
    indexer = get_indexer()
    if title not in indexer.index.notes:
        raise ValueError(f"Experiment note not found: {title}")
        
    note_path = indexer.index.notes[title].filepath
    if not note_path.exists():
        raise ValueError("File does not exist")
        
    content = note_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    
    # Find the Daily Log table
    table_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == "## Daily Log":
            table_idx = i
            break
            
    if table_idx == -1:
        raise ValueError("Could not find '## Daily Log' section in the note.")
        
    # Find the last row of the table to determine the Day number
    last_row_idx = table_idx + 2 # Skip header and divider
    day_num = 1
    
    while last_row_idx < len(lines) and lines[last_row_idx].strip().startswith("|"):
        row = lines[last_row_idx]
        if row.strip() != "":
            parts = [p.strip() for p in row.split("|")]
            if len(parts) > 1 and parts[1].isdigit():
                day_num = int(parts[1])
        last_row_idx += 1
        
    # We are now inserting at last_row_idx
    new_day = day_num + 1
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Clean inputs for markdown table
    notes = notes.replace("|", "\|").replace("\n", " ")
    metric = metric.replace("|", "\|")
    subjective = subjective.replace("|", "\|")
    
    new_row = f"| {new_day} | {date_str} | {notes} | {metric} | {subjective} |"
    
    lines.insert(last_row_idx, new_row)
    
    note_path.write_text("\n".join(lines), encoding="utf-8")
    
    return new_row
