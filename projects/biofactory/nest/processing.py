from __future__ import annotations

from dataclasses import dataclass

from biofactory.resources.production_chains import ProductionRecipe
from biofactory.resources.resource_types import ResourceType


@dataclass
class ProcessingState:
    recipe: ProductionRecipe
    progress: float = 0.0
    batches_completed: int = 0
    blocked_reason: str | None = "idle"
    last_worker_count: int = 0
    last_efficiency: float = 0.0
    last_completed_tick: int = -999999

    def input_missing(self, chamber) -> dict[ResourceType, float]:
        missing: dict[ResourceType, float] = {}
        for resource_type, required in self.recipe.inputs.items():
            shortage = required - chamber.inventory.get(resource_type)
            if shortage > 1e-6:
                missing[resource_type] = shortage
        return missing

    def output_room_missing(self, chamber) -> dict[ResourceType, float]:
        missing: dict[ResourceType, float] = {}
        for resource_type, produced in self.recipe.outputs.items():
            shortage = produced - chamber.available_capacity(resource_type)
            if shortage > 1e-6:
                missing[resource_type] = shortage
        return missing

    def can_start(self, chamber) -> bool:
        if self.recipe is None:
            self.blocked_reason = "no_recipe"
            return False
        if chamber.health <= 0.0:
            self.blocked_reason = "damaged"
            return False
        if self.input_missing(chamber):
            self.blocked_reason = "missing_inputs"
            return False
        if self.output_room_missing(chamber):
            self.blocked_reason = "output_full"
            return False
        return True

    def advance(self, chamber, work_amount: float) -> bool:
        if work_amount <= 0.0:
            self.last_efficiency = 0.0
            return False
        self.last_efficiency = work_amount
        if not self.can_start(chamber):
            return False

        self.blocked_reason = None
        self.progress += work_amount
        if self.progress < self.recipe.work:
            return False

        for resource_type, required in self.recipe.inputs.items():
            chamber.remove_resource(resource_type, required)
        for resource_type, produced in self.recipe.outputs.items():
            chamber.add_resource(resource_type, produced)
        self.progress = 0.0
        self.batches_completed += 1
        if not self.can_start(chamber):
            return True
        self.blocked_reason = None
        return True
