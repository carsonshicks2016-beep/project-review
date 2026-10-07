from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

from biofactory.neural.brain import Brain


@dataclass
class BrainRecord:
    score: float
    role: str
    brain: Brain


@dataclass
class BrainArchive:
    max_records: int = 64
    records: Deque[BrainRecord] = field(default_factory=deque)

    def add(self, record: BrainRecord) -> None:
        self.records.append(record)
        ordered = sorted(self.records, key=lambda item: item.score, reverse=True)[: self.max_records]
        self.records = deque(ordered, maxlen=self.max_records)

    def best_for_role(self, role: str) -> Brain | None:
        for record in self.records:
            if record.role == role:
                return record.brain
        return self.records[0].brain if self.records else None
