"""
The genetic algorithm.

A population of MLP genomes is evaluated by driving each one around the track
(via CarAgent), then bred into the next generation with:

  * elitism            - the best `elite_frac` carry over unchanged,
  * tournament select  - pick the best of k random contenders as a parent,
  * uniform crossover  - each gene from either parent at random,
  * Gaussian mutation  - perturb a fraction of genes by N(0, sigma).

Headless training calls `evaluate_headless` then `advance`. The live view does
its own on-screen evaluation and calls `advance` with the fitnesses it collected
— so both paths share identical breeding + stats.
"""
from __future__ import annotations

import colorsys

import numpy as np

from .agent import CarAgent
from .brain import MLP
from .config import EvoSpec, SimSpec, get_car
from .sensors import SensorSuite
from .track import random_circuit


def _eval_worker(args):
    """Top-level worker function for multiprocessing headless GA evaluations."""
    genome, track, spec, sim, sensors, evo = args
    from .app import AutoBox
    brain = MLP.from_genome(genome, sensors.obs_size, evo.hidden, evo.n_actions, weight_scale=evo.init_weight_scale)
    agent = CarAgent(brain, spec, sim, track, sensors, evo, AutoBox)
    return agent.run()


def _palette(n):
    out = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb((i / max(1, n)) * 0.85, 0.65, 0.95)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


class GA:
    def __init__(self, evo: EvoSpec | None = None, car: str = "supra",
                 seeds=(7,), sim: SimSpec | None = None, rng_seed: int = 0,
                 tracks=None, track_name=None):
        self.evo = evo or EvoSpec()
        self.car = car
        self.spec = get_car(car)
        self.sim = sim or SimSpec()
        self.sensors = SensorSuite()
        self.obs_size = self.sensors.obs_size
        # specialist: evaluate on one (or more) FIXED tracks; generalist: a pool
        # of seeded random circuits (the original behaviour).
        self.tracks = list(tracks) if tracks else [random_circuit(seed=s) for s in seeds]
        self.track_name = track_name
        self.rng = np.random.default_rng(rng_seed)
        self.colors = _palette(self.evo.pop_size)

        self._tmpl = self._new_brain()
        self.genome_size = self._tmpl.size
        self.genomes = [self._new_brain().get_genome() for _ in range(self.evo.pop_size)]

        self.generation = 0
        self.history = []                 # [(gen, best, mean), ...]
        self.best_genome = self.genomes[0].copy()
        self.best_fitness = -1e9

    # ------------------------------------------------------------------ #
    def _new_brain(self) -> MLP:
        e = self.evo
        return MLP(self.obs_size, e.hidden, e.n_actions,
                   weight_scale=e.init_weight_scale,
                   bias_throttle=e.bias_throttle, bias_brake=e.bias_brake)

    def brain_from(self, genome) -> MLP:
        e = self.evo
        return MLP.from_genome(genome, self.obs_size, e.hidden, e.n_actions,
                               weight_scale=e.init_weight_scale)

    def make_agents(self, genomes, track):
        from .app import AutoBox
        return [CarAgent(self.brain_from(g), self.spec, self.sim, track,
                         self.sensors, self.evo, AutoBox,
                         color=self.colors[i % len(self.colors)], index=i)
                for i, g in enumerate(genomes)]

    # ------------------------------------------------------------------ #
    def evaluate_headless(self, genomes, workers=None) -> np.ndarray:
        """Score every genome (mean over the GA's tracks). Headless = fast."""
        import os
        import concurrent.futures

        workers = workers or os.cpu_count() or 4
        fits = np.zeros(len(genomes))
        for track in self.tracks:
            if workers > 1:
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                    args_list = [(g, track, self.spec, self.sim, self.sensors, self.evo) for g in genomes]
                    results = list(executor.map(_eval_worker, args_list))
                fits += np.array(results)
            else:
                agents = self.make_agents(genomes, track)
                for i, a in enumerate(agents):
                    fits[i] += a.run()
        return fits / len(self.tracks)

    # ------------------------------------------------------------------ #
    def advance(self, fitnesses) -> dict:
        """Record stats for the current population, then breed the next one."""
        fitnesses = np.asarray(fitnesses, dtype=float)
        order = np.argsort(fitnesses)[::-1]
        best_i = int(order[0])
        stats = {
            "gen": self.generation,
            "best": float(fitnesses[best_i]),
            "mean": float(fitnesses.mean()),
            "median": float(np.median(fitnesses)),
        }
        self.history.append((self.generation, stats["best"], stats["mean"]))
        if fitnesses[best_i] > self.best_fitness:
            self.best_fitness = float(fitnesses[best_i])
            self.best_genome = self.genomes[best_i].copy()

        self.genomes = self._breed(self.genomes, fitnesses, order)
        self.generation += 1
        return stats

    def _breed(self, genomes, fitnesses, order):
        e = self.evo
        pop = len(genomes)
        n_elite = max(1, int(e.elite_frac * pop))
        nxt = [genomes[int(i)].copy() for i in order[:n_elite]]   # elitism
        while len(nxt) < pop:
            pa = self._tournament(genomes, fitnesses)
            pb = self._tournament(genomes, fitnesses)
            child = self._crossover(pa, pb)
            self._mutate(child)
            nxt.append(child)
        return nxt

    def _tournament(self, genomes, fitnesses):
        idx = self.rng.integers(0, len(genomes), self.evo.tournament_k)
        win = idx[int(np.argmax(fitnesses[idx]))]
        return genomes[int(win)]

    def _crossover(self, pa, pb):
        mask = self.rng.random(pa.size) < 0.5
        return np.where(mask, pa, pb).copy()

    def _mutate(self, g):
        m = self.rng.random(g.size) < self.evo.mutation_rate
        g += m * self.rng.standard_normal(g.size) * self.evo.mutation_sigma

    # ------------------------------------------------------------------ #
    def save_champion(self, path: str):
        import os
        tmp = f"{path}.tmp.npz"      # atomic write (concurrent --watch is safe)
        np.savez(tmp, genome=self.best_genome, fitness=self.best_fitness,
                 obs_size=self.obs_size, hidden=np.array(self.evo.hidden),
                 n_actions=self.evo.n_actions, car=self.car,
                 generation=self.generation, track=(self.track_name or ""))
        os.replace(tmp, path)

    @staticmethod
    def load_champion(path: str):
        d = np.load(path, allow_pickle=True)
        return {
            "genome": d["genome"],
            "fitness": float(d["fitness"]),
            "obs_size": int(d["obs_size"]),
            "hidden": tuple(int(h) for h in d["hidden"]),
            "n_actions": int(d["n_actions"]),
            "car": str(d["car"]),
            "generation": int(d["generation"]),
            "track": (str(d["track"]) if "track" in d.files else ""),
        }

    @staticmethod
    def check_champion(ck: dict, obs_size: int, path: str = ""):
        """Loud pre-hills guard. A stored champion with the old obs layout
        would otherwise reshape into garbage (`from_genome` is called with the
        CURRENT obs_size) or crash mid-drive — refuse with the real reason."""
        got = int(ck["obs_size"])
        if got != obs_size:
            raise ValueError(
                f"GA champion{f' {path!r}' if path else ''} has obs_size {got} "
                f"but the current sensor layout is {obs_size}"
                + ("  [PRE-HILLS checkpoint — the obs vector grew 40 -> 58 "
                   "when terrain/air sensing landed; retrain for the current layout]"
                   if got < obs_size else ""))

    def warm_start(self, genome, fitness: float = -1e9):
        """Resume evolution from a saved champion: seed the population with the
        champion (kept as an exact elite) plus mutated variants of it."""
        g = np.asarray(genome, dtype=float).ravel()
        if g.size != self.genome_size:
            raise ValueError(f"champion genome size {g.size} != current "
                             f"{self.genome_size} (architecture mismatch)")
        self.best_genome = g.copy()
        self.best_fitness = float(fitness)
        self.genomes = [g.copy()]
        for _ in range(self.evo.pop_size - 1):
            c = g.copy()
            self._mutate(c)
            self.genomes.append(c)
