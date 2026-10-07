"""Multi-objective fitness, behavior descriptors, and Agent-Zero-relative scoring.

Fitness =  performance
          - w_inj * injury_risk
          - w_eng * energy_cost
          - w_rec * recovery_debt
          - w_imp * biological_implausibility

`performance` is the mean improvement ratio across biomes relative to Agent
Zero, so a score of 1.0 = baseline-you and 1.3 = 30% better on average. The
penalty weights are exposed so you can audit how the ranking was produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .agent import Agent
from .biomes import evaluate_all, BiomeResult
from .realism import implausibility


@dataclass
class Weights:
    injury: float = 0.6
    energy: float = 0.15
    recovery: float = 0.25
    implausibility: float = 0.10
    budget: float = 0.5


@dataclass
class Baseline:
    """Agent Zero's biome scores, used to normalize improvement."""
    scores: dict[str, float]

    @classmethod
    def from_agent(cls, agent: Agent) -> "Baseline":
        return cls({n: r.score for n, r in evaluate_all(agent).items()})


@dataclass
class EvalResult:
    biomes: dict[str, BiomeResult]
    descriptors: tuple[float, float]      # (strength_to_mass_norm, vmax_norm) in [0,1]
    performance: float
    injury_risk: float
    energy_cost: float
    recovery_debt: float
    implausibility: float
    budget_overflow: float
    fitness: float
    realism: int
    detail: dict[str, float] = field(default_factory=dict)


# fixed reference ranges so descriptors are comparable across runs, sized to the
# reachable HUMAN-ACHIEVABLE span (HP/OE bodies clip at the high end, by design)
_S2M_REF = (0.95, 2.0)    # 1RM-equiv kg per kg bodyweight
_VMAX_REF = (6.5, 9.3)    # m/s


def _norm(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def evaluate_agent(agent: Agent, baseline: Baseline, w: Weights = Weights()) -> EvalResult:
    biomes = evaluate_all(agent)

    # performance: mean improvement ratio vs Agent Zero
    ratios = [biomes[n].score / max(baseline.scores[n], 1e-6) for n in biomes]
    performance = float(np.mean(ratios))

    injury_risk = float(np.mean([r.injury for r in biomes.values()]))
    energy_cost = agent.budgets.resting_kcal / 2000.0          # normalized
    recovery_debt = agent.phenotype.recovery_debt + 0.5 * agent.phenotype.tissue_damage
    imp = implausibility(agent.genotype)
    budget_overflow = agent.budgets.total_overflow

    fitness = (performance
               - w.injury * injury_risk
               - w.energy * energy_cost
               - w.recovery * recovery_debt
               - w.implausibility * imp
               - w.budget * budget_overflow)

    s2m = biomes["iron_zone"].detail["strength_to_mass"]
    vmax = biomes["track"].score
    descriptors = (_norm(s2m, *_S2M_REF), _norm(vmax, *_VMAX_REF))

    return EvalResult(
        biomes=biomes, descriptors=descriptors, performance=performance,
        injury_risk=injury_risk, energy_cost=energy_cost, recovery_debt=recovery_debt,
        implausibility=imp, budget_overflow=budget_overflow, fitness=float(fitness),
        realism=int(agent.realism),
        detail={"strength_to_mass": s2m, "v_max": vmax, "body_mass": agent.body_mass},
    )
