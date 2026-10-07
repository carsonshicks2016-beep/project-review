from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from .action_space import ActionSpace


FitnessFn = Callable[[list[str]], float]


@dataclass(frozen=True)
class Genome:
    macros: tuple[str, ...]
    fitness: float = 0.0


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 48
    genome_length: int = 180
    generations: int = 50
    elite_count: int = 4
    mutation_rate: float = 0.08
    seed: int = 0


class MacroGA:
    """Simple macro-sequence GA for hard rooms, bosses, and demonstration seeds."""

    def __init__(self, action_space: ActionSpace, config: GAConfig):
        self.action_space = action_space
        self.config = config
        self.rng = random.Random(config.seed)
        self.choices = [name for name in action_space.names() if name not in {"start", "pause_toggle"}]

    def evolve(self, fitness_fn: FitnessFn) -> Genome:
        population = [self._random_genome() for _ in range(self.config.population_size)]
        best = population[0]
        for _generation in range(self.config.generations):
            scored = [Genome(genome.macros, fitness_fn(list(genome.macros))) for genome in population]
            scored.sort(key=lambda genome: genome.fitness, reverse=True)
            best = max(best, scored[0], key=lambda genome: genome.fitness)
            elites = scored[: self.config.elite_count]
            population = elites[:]
            while len(population) < self.config.population_size:
                parent_a = self._tournament(scored)
                parent_b = self._tournament(scored)
                population.append(self._mutate(self._crossover(parent_a, parent_b)))
        return best

    def _random_genome(self) -> Genome:
        return Genome(tuple(self.rng.choice(self.choices) for _ in range(self.config.genome_length)))

    def _tournament(self, population: list[Genome], size: int = 3) -> Genome:
        contenders = self.rng.sample(population, k=min(size, len(population)))
        return max(contenders, key=lambda genome: genome.fitness)

    def _crossover(self, parent_a: Genome, parent_b: Genome) -> Genome:
        cut = self.rng.randrange(1, len(parent_a.macros))
        return Genome(parent_a.macros[:cut] + parent_b.macros[cut:])

    def _mutate(self, genome: Genome) -> Genome:
        macros = list(genome.macros)
        for index in range(len(macros)):
            if self.rng.random() < self.config.mutation_rate:
                macros[index] = self.rng.choice(self.choices)
        return Genome(tuple(macros))

