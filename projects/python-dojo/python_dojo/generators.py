"""Compatibility shim: the generator system lives in python_dojo.gen."""

from __future__ import annotations

from .gen import (
    all_generator_topic_ids,
    archetype_count,
    archetype_names,
    generate_exercise,
    parse_exercise_id,
    pick_exercise,
    total_archetypes,
)

__all__ = [
    "all_generator_topic_ids",
    "archetype_count",
    "archetype_names",
    "generate_exercise",
    "parse_exercise_id",
    "pick_exercise",
    "total_archetypes",
]
