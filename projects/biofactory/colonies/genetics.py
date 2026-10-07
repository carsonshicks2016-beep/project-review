from __future__ import annotations

from dataclasses import dataclass

from numpy.random import Generator


@dataclass(frozen=True)
class Genes:
    speed: float
    strength: float
    health: float
    lifespan: float
    energy_efficiency: float
    vision_range: float
    inventory_capacity: float
    pheromone_sensitivity: float
    aggression: float
    fear: float
    curiosity: float
    traffic_tolerance: float
    mutation_rate: float

    @classmethod
    def random(cls, rng: Generator) -> "Genes":
        return cls(
            speed=float(rng.normal(1.0, 0.08)),
            strength=float(rng.normal(1.0, 0.08)),
            health=float(rng.normal(1.0, 0.06)),
            lifespan=float(rng.normal(1.0, 0.06)),
            energy_efficiency=float(rng.normal(1.0, 0.05)),
            vision_range=float(rng.normal(1.0, 0.08)),
            inventory_capacity=float(rng.normal(1.0, 0.04)),
            pheromone_sensitivity=float(rng.normal(1.0, 0.12)),
            aggression=float(rng.uniform(0.05, 0.25)),
            fear=float(rng.uniform(0.05, 0.25)),
            curiosity=float(rng.uniform(0.55, 1.25)),
            traffic_tolerance=float(rng.uniform(0.45, 1.15)),
            mutation_rate=float(rng.uniform(0.01, 0.04)),
        )
