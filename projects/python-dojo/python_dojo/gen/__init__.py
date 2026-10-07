from __future__ import annotations

from .registry import (
    all_generator_topic_ids,
    archetype,
    archetype_count,
    archetype_names,
    generate_exercise,
    make_exercise,
    parse_exercise_id,
    pick_exercise,
    total_archetypes,
)

# Importing the content modules registers every archetype.
from . import basics  # noqa: E402,F401
from . import flow  # noqa: E402,F401

__all__ = [
    "all_generator_topic_ids",
    "archetype",
    "archetype_count",
    "archetype_names",
    "generate_exercise",
    "make_exercise",
    "parse_exercise_id",
    "pick_exercise",
    "total_archetypes",
]
