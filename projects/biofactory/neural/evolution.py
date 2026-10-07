from __future__ import annotations

from numpy.random import Generator

from biofactory.neural.brain import Brain


def mutate_brain(brain: Brain, rng: Generator, mutation_rate: float) -> Brain:
    return brain.copy_mutated(rng, mutation_rate)
