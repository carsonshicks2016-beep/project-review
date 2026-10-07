from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConstructionTask:
    x: int
    y: int
    task_type: str
    work_remaining: float
