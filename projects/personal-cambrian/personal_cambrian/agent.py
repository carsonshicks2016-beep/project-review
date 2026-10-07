"""AGENT = genotype (structure) + training program (process) + phenotype (state)
         + controller/skill, with cached derived quantities.

The three layers are kept distinct so the engine can attribute an improvement to
morphology, trainable adaptation, or control (see `attribute_gain`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .genome import Genotype
from .phenotype import Phenotype, TrainingProgram, adapt
from .budgets import BudgetReport, evaluate_budgets
from .realism import Realism, classify, implausibility


@dataclass
class Agent:
    genotype: Genotype
    program: TrainingProgram = field(default_factory=TrainingProgram)
    weeks: float = 16.0
    phenotype: Phenotype = field(init=False)
    budgets: BudgetReport = field(init=False)
    realism: Realism = field(init=False)

    def __post_init__(self) -> None:
        self.develop()

    def develop(self) -> None:
        """(Re)run adaptation + budget accounting. Call after changing genes/program."""
        self.phenotype = adapt(self.genotype, self.program, self.weeks)
        self.budgets = evaluate_budgets(self.genotype, self.phenotype)
        self.realism = classify(self.genotype)

    # convenience
    @property
    def body_mass(self) -> float:
        return self.budgets.body_mass_kg

    @property
    def implausibility(self) -> float:
        return implausibility(self.genotype)

    def with_naive_controller(self) -> "Agent":
        """A copy whose skill is reset to baseline (no motor learning), used to
        attribute how much performance came from control vs structure/state."""
        clone = Agent(self.genotype.copy(), self.program, self.weeks)
        clone.phenotype.skill = {k: 0.78 for k in clone.phenotype.skill}
        return clone
