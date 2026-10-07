"""The main evolutionary / quality-diversity loop.

Pattern (MAP-Elites with morphology + training-program mutation):
    1. seed the archive with random bodies near the chosen realism envelope
    2. repeatedly: sample an elite, mutate its genotype + training program,
       develop the resulting agent (adaptation + budgets), evaluate it across
       all biomes, and try to insert it into its behavior niche.

Morphology and controller adaptation are kept separate: every new body is
*re-developed* (its phenotype + skill re-trained for its structure) before it is
judged, so we never declare a new morphology inferior just because it inherited
a mismatched controller (BRAIN AND BODY OPTIMIZATION in PLAN.md).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .genome import Genotype, random_genotype, mutate
from .phenotype import TrainingProgram
from .agent import Agent
from .fitness import Baseline, Weights, evaluate_agent
from .map_elites import MapElites


def agent_zero(weeks: float = 16.0) -> Agent:
    """The user's default twin: prior genotype + balanced training program.

    NOTE: built from population PRIORS (status=unknown), not measurements. Wide
    uncertainty until real metrics are supplied (see data/agent_zero.example.json
    and the reality loop in PLAN.md).
    """
    return Agent(Genotype(), TrainingProgram(), weeks)


def mutate_program(prog: TrainingProgram, rng: np.random.Generator,
                   sigma: float = 0.08) -> TrainingProgram:
    v = prog.as_vector() + rng.normal(0, sigma, size=6)
    v[:4] = np.clip(v[:4], 0.0, None)          # focus weights non-negative
    v[4:] = np.clip(v[4:], 0.05, 1.0)          # volume, intensity in (0,1]
    return TrainingProgram.from_vector(v)


@dataclass
class EvolveConfig:
    iterations: int = 4000
    init_population: int = 200
    resolution: int = 24
    realism: str = "ha"          # "ha" | "hp" | "op"
    weeks: float = 16.0
    seed: int = 0
    gene_sigma: float = 0.06


def run(cfg: EvolveConfig = EvolveConfig(), verbose: bool = True) -> tuple[MapElites, Baseline]:
    rng = np.random.default_rng(cfg.seed)

    z = agent_zero(cfg.weeks)
    baseline = Baseline.from_agent(z)
    weights = Weights()
    archive = MapElites(resolution=cfg.resolution)

    def evaluate_and_insert(geno: Genotype, prog: TrainingProgram) -> None:
        agent = Agent(geno, prog, cfg.weeks)
        ev = evaluate_agent(agent, baseline, weights)
        archive.add((geno, prog), ev.fitness, ev, ev.realism)

    # --- seed: Agent Zero itself + random bodies + Agent-Zero neighborhood ---
    evaluate_and_insert(z.genotype.copy(), z.program)
    for _ in range(cfg.init_population):
        if rng.random() < 0.5:
            geno = random_genotype(cfg.realism, rng)
            prog = mutate_program(TrainingProgram(), rng, sigma=0.25)
        else:  # stay near the user
            geno = mutate(z.genotype, cfg.realism, sigma_frac=0.12, rng=rng)
            prog = mutate_program(z.program, rng, sigma=0.2)
        evaluate_and_insert(geno, prog)

    # --- main QD loop -------------------------------------------------------
    for it in range(cfg.iterations):
        parent = archive.sample_elite(rng)
        pgeno, pprog = parent.genome
        child_geno = mutate(pgeno, cfg.realism, sigma_frac=cfg.gene_sigma, rng=rng)
        child_prog = mutate_program(pprog, rng)
        evaluate_and_insert(child_geno, child_prog)

        if verbose and (it + 1) % max(1, cfg.iterations // 10) == 0:
            b = archive.best()
            print(f"  iter {it+1:5d}  coverage={archive.coverage:5.1%}  "
                  f"cells={len(archive.grid):4d}  qd={archive.qd_score:7.1f}  "
                  f"best_fit={b.fitness:5.2f}")

    return archive, baseline
