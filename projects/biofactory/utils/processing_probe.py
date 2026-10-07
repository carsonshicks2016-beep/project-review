from __future__ import annotations

import argparse

from biofactory.config import DEFAULT_CONFIG
from biofactory.resources.resource_types import ResourceType
from biofactory.simulation.engine import SimulationEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic BioFactory processing-chain probe.")
    parser.add_argument("--ticks", type=int, default=1500)
    args = parser.parse_args()

    engine = SimulationEngine(DEFAULT_CONFIG)
    engine.step(args.ticks)
    colony = engine.colony

    harvested = sum(
        chamber.fungus_farm_state.fungus_harvested
        for chamber in colony.fungus_farm_chambers()
        if chamber.fungus_farm_state is not None
    )
    nutrient = next(chamber for chamber in colony.processor_chambers() if chamber.chamber_type == "nutrient_processor")
    nutrient_batches = nutrient.processing_state.batches_completed if nutrient.processing_state is not None else 0
    processed_reserve = colony.inventory.get(ResourceType.PROTEIN_PASTE) + colony.inventory.get(ResourceType.COMPOST)

    assert harvested > 0.0, "fungus farms did not harvest fungus"
    assert nutrient_batches > 0, "nutrient processor did not complete a fungus-food batch"
    assert colony.inventory.get(ResourceType.FOOD) > 0.0, "food reserve was empty after processing probe"
    assert processed_reserve > 0.0, "no secondary processed reserve was produced"

    print("BioFactory processing probe passed")
    print(f"ticks: {engine.tick}")
    print(f"fungus harvested: {harvested:.2f}")
    print(f"nutrient batches: {nutrient_batches}")
    print(f"food stored: {colony.inventory.get(ResourceType.FOOD):.2f}")
    print(f"fungus stored: {colony.inventory.get(ResourceType.FUNGUS):.2f}")
    print(f"protein paste stored: {colony.inventory.get(ResourceType.PROTEIN_PASTE):.2f}")
    print(f"compost stored: {colony.inventory.get(ResourceType.COMPOST):.2f}")


if __name__ == "__main__":
    main()
