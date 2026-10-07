from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReproductionSystem:
    enabled: bool = False

    def update(self) -> None:
        # Eggs, larvae, pupae, and evolved ant spawning are post-MVP systems.
        return
