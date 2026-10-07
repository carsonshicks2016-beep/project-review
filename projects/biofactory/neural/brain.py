from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.random import Generator


class BrainInput(IntEnum):
    ENERGY = 0
    HEALTH = 1
    AGE = 2
    HAS_CARGO = 3
    INVENTORY_AMOUNT = 4
    INVENTORY_TYPE = 5
    INVENTORY_CAPACITY = 6
    AGGRESSION = 7
    CURIOSITY = 8
    FEAR = 9
    STRENGTH = 10
    SPEED = 11
    VISION_RANGE = 12
    PHEROMONE_SENSITIVITY = 13
    TRAFFIC_TOLERANCE = 14
    TERRAIN_SPEED = 15
    TERRAIN_ENERGY_COST = 16
    TEMPERATURE = 17
    HUMIDITY = 18
    LIGHT = 19
    FOOD_SCENT = 20
    WATER_SCENT = 21
    PROTEIN_SCENT = 22
    CONSTRUCTION_SCENT = 23
    WASTE_SCENT = 24
    TARGET_RESOURCE_AMOUNT = 25
    FOOD_AMOUNT = 26
    WATER_AMOUNT = 27
    PROTEIN_AMOUNT = 28
    CONSTRUCTION_AMOUNT = 29
    WASTE_AMOUNT = 30
    FOOD_PHEROMONE = 31
    WATER_PHEROMONE = 32
    PROTEIN_PHEROMONE = 33
    CONSTRUCTION_PHEROMONE = 34
    DANGER_PHEROMONE = 35
    RECRUITMENT_PHEROMONE = 36
    TRAFFIC_PHEROMONE = 37
    TERRITORY_PHEROMONE = 38
    WASTE_PHEROMONE = 39
    QUEEN_SIGNAL = 40
    EMERGENCY_PHEROMONE = 41
    DEMAND_PHEROMONE = 42
    STORAGE_FULL_PHEROMONE = 43
    NEST_ENTRANCE_PHEROMONE = 44
    DEATH_PHEROMONE = 45
    ALLY_DENSITY = 46
    ENEMY_DENSITY = 47
    PREDATOR_SIGNAL = 48
    TRAFFIC_CONGESTION = 49
    HOME_DISTANCE = 50
    HOME_DIRECTION_X = 51
    HOME_DIRECTION_Y = 52
    WALL_PROXIMITY = 53
    CHAMBER_DEMAND = 54
    STORAGE_FULLNESS = 55
    TARGET_PRIORITY = 56
    TARGET_SATURATION = 57
    MEMORY_0 = 58
    MEMORY_1 = 59
    MEMORY_2 = 60
    MEMORY_3 = 61
    ENERGY_STRESS = 62
    RANDOM_DRIFT = 63


class BrainOutput(IntEnum):
    TURN = 0
    MOVE_SPEED = 1
    PICKUP = 2
    DROP = 3
    LAY_FOOD = 4
    LAY_WATER = 5
    LAY_PROTEIN = 6
    LAY_CONSTRUCTION = 7
    LAY_DANGER = 8
    LAY_RECRUITMENT = 9
    LAY_TRAFFIC = 10
    LAY_TERRITORY = 11
    LAY_WASTE = 12
    DIG = 13
    BUILD_REINFORCE = 14
    ATTACK = 15
    DEFEND = 16
    GROOM_HELP = 17
    TEND_LARVAE = 18
    TEND_FUNGUS = 19
    AVOID_CONGESTION = 20
    EXPLORE = 21
    LAY_DEMAND = 22
    LAY_EMERGENCY = 23
    MEMORY_0 = 24
    MEMORY_1 = 25
    MEMORY_2 = 26
    MEMORY_3 = 27


MEMORY_SIZE = 4
INPUT_SIZE = len(BrainInput)
HIDDEN_SIZE = 24
OUTPUT_SIZE = len(BrainOutput)
TANH_OUTPUTS = (
    BrainOutput.TURN,
    BrainOutput.MOVE_SPEED,
    BrainOutput.MEMORY_0,
    BrainOutput.MEMORY_1,
    BrainOutput.MEMORY_2,
    BrainOutput.MEMORY_3,
)


@dataclass
class Brain:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray

    @classmethod
    def random(cls, rng: Generator, scale: float = 0.45) -> "Brain":
        return cls(
            w1=rng.normal(0.0, scale, (INPUT_SIZE, HIDDEN_SIZE)).astype(np.float32),
            b1=rng.normal(0.0, 0.08, HIDDEN_SIZE).astype(np.float32),
            w2=rng.normal(0.0, scale, (HIDDEN_SIZE, HIDDEN_SIZE)).astype(np.float32),
            b2=rng.normal(0.0, 0.08, HIDDEN_SIZE).astype(np.float32),
            w3=rng.normal(0.0, scale, (HIDDEN_SIZE, OUTPUT_SIZE)).astype(np.float32),
            b3=rng.normal(0.0, 0.08, OUTPUT_SIZE).astype(np.float32),
        )

    @classmethod
    def seeded_worker(cls, rng: Generator) -> "Brain":
        brain = cls.random(rng, scale=0.18)
        # A light initial bias gives the first generation useful reflexes while
        # still leaving movement and trail shape to local signals.
        brain.b3[BrainOutput.MOVE_SPEED] = 0.55
        brain.b3[BrainOutput.PICKUP] = 0.75
        brain.b3[BrainOutput.DROP] = 0.70
        brain.b3[BrainOutput.LAY_FOOD] = 0.82
        brain.b3[BrainOutput.LAY_TRAFFIC] = 0.45
        brain.b3[BrainOutput.AVOID_CONGESTION] = 0.42
        brain.b3[BrainOutput.EXPLORE] = 0.35
        return brain

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        h1 = np.tanh(inputs @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        raw = h2 @ self.w3 + self.b3
        outputs = 1.0 / (1.0 + np.exp(-raw))
        for index in TANH_OUTPUTS:
            outputs[index] = np.tanh(raw[index])
        return outputs.astype(np.float32)

    def copy_mutated(self, rng: Generator, mutation_rate: float) -> "Brain":
        def mutate(array: np.ndarray) -> np.ndarray:
            mask = rng.random(array.shape) < mutation_rate
            noise = rng.normal(0.0, 0.05, array.shape)
            return (array + mask * noise).astype(np.float32)

        return Brain(
            w1=mutate(self.w1),
            b1=mutate(self.b1),
            w2=mutate(self.w2),
            b2=mutate(self.b2),
            w3=mutate(self.w3),
            b3=mutate(self.b3),
        )
