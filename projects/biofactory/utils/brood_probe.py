from __future__ import annotations

import argparse
from dataclasses import replace

from biofactory.config import DEFAULT_CONFIG
from biofactory.resources.resource_types import ResourceType
from biofactory.simulation.engine import SimulationEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic BioFactory brood pipeline probe.")
    parser.add_argument("--ticks", type=int, default=120)
    args = parser.parse_args()

    config = replace(
        DEFAULT_CONFIG,
        ants=replace(DEFAULT_CONFIG.ants, count=80),
        colony=replace(
            DEFAULT_CONFIG.colony,
            max_population=120,
            queen_egg_interval_ticks=2,
            egg_to_larva_ticks=8,
            larva_to_pupa_ticks=8,
            pupa_to_worker_ticks=8,
            nursery_base_efficiency=1.0,
            nursery_worker_bonus_per_ant=0.05,
            nursery_max_worker_bonus=0.80,
        ),
        world=replace(DEFAULT_CONFIG.world, width=90, height=60, nest_x=45.0, nest_y=30.0, leaf_patch_count=6),
    )
    engine = SimulationEngine(config)
    for resource_type, amount in (
        (ResourceType.FOOD, 300.0),
        (ResourceType.WATER, 300.0),
        (ResourceType.LEAVES, 600.0),
        (ResourceType.FUNGUS, 120.0),
    ):
        engine.colony.add_to_storage(resource_type, amount, tick=-1)
    for nursery in engine.colony.nursery_chambers():
        nursery.add_resource(ResourceType.FOOD, nursery.capacity_for(ResourceType.FOOD))
        nursery.add_resource(ResourceType.WATER, nursery.capacity_for(ResourceType.WATER))
    engine.colony.refresh_aggregate_inventory()
    engine.colony._refresh_demand_cache()

    initial_population = len(engine.ants)
    engine.step(args.ticks)
    active_nurseries, blocked_nurseries, workers_born, brood_deaths = engine.colony.nursery_summary()

    assert len(engine.ants) > initial_population, "brood pipeline did not spawn workers"
    assert len(engine.ants) <= config.colony.max_population, "population cap was exceeded"
    assert workers_born > 0, "nursery did not report worker births"
    assert engine.colony.stats.ants_born == len(engine.ants), "ant birth stats diverged from population"

    print("BioFactory brood probe passed")
    print(f"ticks: {engine.tick}")
    print(f"population: {len(engine.ants)} / {config.colony.max_population}")
    print(f"workers born: {workers_born}")
    print(f"brood eggs/larvae/pupae: {engine.colony.eggs}/{engine.colony.larvae}/{engine.colony.pupae}")
    print(f"nurseries active/blocked: {active_nurseries}/{blocked_nurseries}")
    print(f"brood deaths: {brood_deaths}")


if __name__ == "__main__":
    main()
