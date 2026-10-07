"""
Genetic algorithm (neuro-evolution).

A population of MLP brains drives a generation of Supras.  When the generation
ends we rank by fitness, keep the elites untouched, breed the survivors with
crossover, and mutate the offspring.  The best genome of all time is retained
("hall of fame") and can be saved/loaded.
"""

from __future__ import annotations
import colorsys
import numpy as np

from .config import Config
from .brain import MLPGenome


def make_palette(n):
    out = []
    for i in range(n):
        h = (i / max(1, n)) * 0.85
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


class Evolution:
    def __init__(self, cfg: Config, seed=None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.evo = cfg.evo
        self.brains = [self._fresh() for _ in range(self.evo.population)]
        self.best_genome = None
        self.best_fitness = -1e9
        self.history = []          # best fitness per generation
        self.avg_history = []

    def _fresh(self):
        return MLPGenome(self.cfg.sensors.n_inputs, self.evo.hidden_layers,
                         self.cfg.sensors.n_outputs, rng=self.rng,
                         init_scale=self.evo.weight_init_scale)

    def colors(self):
        return make_palette(self.evo.population)

    # ------------------------------------------------------------------ #
    def next_generation(self, fitnesses):
        """Given this generation's fitnesses, build + store the next brains."""
        fits = np.asarray(fitnesses, dtype=np.float64)
        order = np.argsort(fits)[::-1]
        ranked = [self.brains[i] for i in order]
        ranked_fit = fits[order]

        # hall of fame
        if ranked_fit[0] > self.best_fitness:
            self.best_fitness = float(ranked_fit[0])
            self.best_genome = ranked[0].clone()
        self.history.append(float(ranked_fit[0]))
        self.avg_history.append(float(np.mean(fits)))

        n = self.evo.population
        n_elite = max(1, int(n * self.evo.elite_fraction))
        n_surv = max(2, int(n * self.evo.survivor_fraction))
        survivors = ranked[:n_surv]

        new = [ranked[i].clone() for i in range(n_elite)]   # elites pass through
        while len(new) < n:
            p1 = self._tournament(survivors)
            p2 = self._tournament(survivors)
            child = MLPGenome.crossover(p1, p2, self.rng)
            child.mutate(self.evo.mutation_rate, self.evo.mutation_scale, self.rng)
            new.append(child)

        self.brains = new
        return self.brains

    def _tournament(self, pool, k=3):
        picks = self.rng.integers(0, len(pool), size=min(k, len(pool)))
        # pool is already fitness-ordered, so the lowest index is the fittest
        return pool[int(min(picks))]

    # ------------------------------------------------------------------ #
    def seed_from(self, genome):
        """Repopulate by mutating a single (e.g. loaded) genome."""
        self.brains = [genome.clone()]
        while len(self.brains) < self.evo.population:
            c = genome.clone()
            c.mutate(self.evo.mutation_rate, self.evo.mutation_scale, self.rng)
            self.brains.append(c)
        self.best_genome = genome.clone()
        self.best_fitness = 0.0     # unknown for a loaded genome; avoid -1e9 display
