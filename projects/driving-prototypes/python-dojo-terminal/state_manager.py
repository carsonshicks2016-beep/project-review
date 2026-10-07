import json
import os
from typing import TypedDict

PROGRESS_FILE = "progress.json"

class Progress(TypedDict):
    level: int
    concept: str
    streak: int
    total_solved: int
    seen_concepts: list[str]

DEFAULT_PROGRESS: Progress = {
    "level": 1,
    "concept": "The print() function and basic output",
    "streak": 0,
    "total_solved": 0,
    "seen_concepts": []
}

# A granular curriculum map
CURRICULUM = {
    1: "The print() function and basic output",
    2: "Basic Arithmetic (+, -, *, /)",
    3: "Creating a single variable",
    4: "Variable reassignment (e.g. x = x + 1)",
    5: "Creating Strings",
    6: "String concatenation",
    7: "F-Strings (formatted string literals)",
    8: "Booleans (True/False)",
    9: "Comparison Operators (==, !=, >, <)",
    10: "Basic if statements",
    11: "if/else statements",
    12: "Lists and basic indexing",
    13: "Appending to a list",
    14: "For loops using range()",
    15: "For loops iterating over a list",
    16: "While loops",
    17: "Dictionaries (key-value pairs)",
    18: "Writing a basic function",
    19: "Functions with arguments and return values",
    20: "List comprehensions"
}

def load_progress() -> Progress:
    """Load progress from the JSON file or return defaults."""
    if not os.path.exists(PROGRESS_FILE):
        return DEFAULT_PROGRESS.copy()
    
    try:
        with open(PROGRESS_FILE, 'r') as f:
            data = json.load(f)
            # Handle old progress files safely
            if "level" not in data or data["level"] not in CURRICULUM:
                return DEFAULT_PROGRESS.copy()
            if "seen_concepts" not in data:
                data["seen_concepts"] = []
            return data
    except json.JSONDecodeError:
        return DEFAULT_PROGRESS.copy()

def save_progress(progress: Progress) -> None:
    """Save progress to the JSON file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=4)

def update_progress_on_success(progress: Progress) -> Progress:
    """Update progress when an exercise is solved correctly."""
    progress["streak"] += 1
    progress["total_solved"] += 1
    
    # Scaling logic: Levels 1-5 only require 1 success. Levels 6-10 require 2. Beyond requires 3.
    req_successes = 3
    if progress["level"] <= 5:
        req_successes = 1
    elif progress["level"] <= 10:
        req_successes = 2
        
    if progress["streak"] >= req_successes:
        if progress["level"] < max(CURRICULUM.keys()):
            progress["level"] += 1
            progress["concept"] = CURRICULUM[progress["level"]]
            progress["streak"] = 0 # reset streak on level up
            
    return progress

def update_progress_on_failure(progress: Progress) -> Progress:
    """Update progress when an exercise is failed."""
    progress["streak"] = 0
    return progress
