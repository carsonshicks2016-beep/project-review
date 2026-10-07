import copy
import random
import numpy as np
from typing import List

from .config import CarSpec

class CarGenome:
    """
    Wraps a CarSpec and provides genetic mutation operators.
    Enforces realistic physics trade-offs (e.g., downforce adds drag).
    """
    def __init__(self, spec: CarSpec):
        self.spec = copy.deepcopy(spec)
    
    def mutate(self, sigma: float = 0.05):
        """
        Mutates the car parameters by a normal distribution scaled by sigma.
        Enforces realistic limits and tradeoffs.
        """
        # 1. Mass (700 kg to 1500 kg)
        mass_factor = np.clip(np.random.normal(1.0, sigma), 0.8, 1.2)
        self.spec.mass = np.clip(self.spec.mass * mass_factor, 700.0, 1500.0)
        
        # 2. Weight Distribution (0.35 to 0.65 front)
        self.spec.front_weight = np.clip(
            self.spec.front_weight + np.random.normal(0, sigma * 0.1), 
            0.35, 0.65
        )
        
        # 3. Downforce and Drag Tradeoff
        # Higher downforce creates more aerodynamic drag.
        df_factor = np.clip(np.random.normal(1.0, sigma), 0.5, 1.5)
        self.spec.downforce_ClA = np.clip(self.spec.downforce_ClA * df_factor, 0.0, 10.0)
        
        # Base drag + drag induced by downforce
        base_drag = 0.35
        induced_drag = self.spec.downforce_ClA * 0.15
        self.spec.drag_area = base_drag + induced_drag
        
        # 4. Tire Grip (mu) and Durability Tradeoff
        # Stickier tires degrade faster (durability drops).
        mu_factor = np.clip(np.random.normal(1.0, sigma), 0.8, 1.2)
        self.spec.mu = np.clip(self.spec.mu * mu_factor, 1.0, 3.0)
        
        # Baseline durability 1.0. A mu of 2.0 halves durability.
        self.spec.durability = np.clip(2.0 / (self.spec.mu + 0.1), 0.1, 1.0)
        
        # 5. Gearing
        # Mutate the final drive
        fd_factor = np.clip(np.random.normal(1.0, sigma), 0.8, 1.2)
        self.spec.final_drive = np.clip(self.spec.final_drive * fd_factor, 2.0, 5.0)
        
        # Mutate individual gear ratios, keeping them monotonically decreasing
        new_gears = []
        for g in self.spec.gear_ratios:
            g_mut = g * np.clip(np.random.normal(1.0, sigma * 0.5), 0.9, 1.1)
            new_gears.append(g_mut)
        
        # Sort descending to ensure valid transmission
        new_gears.sort(reverse=True)
        self.spec.gear_ratios = new_gears
        
        # 6. Center of Gravity Height
        cg_factor = np.clip(np.random.normal(1.0, sigma), 0.8, 1.2)
        self.spec.cg_height = np.clip(self.spec.cg_height * cg_factor, 0.15, 0.6)

        return self

    def crossover(self, other: 'CarGenome') -> 'CarGenome':
        """
        Uniform crossover between two CarGenomes.
        """
        child_spec = copy.deepcopy(self.spec)
        
        if random.random() < 0.5: child_spec.mass = other.spec.mass
        if random.random() < 0.5: child_spec.front_weight = other.spec.front_weight
        if random.random() < 0.5: child_spec.downforce_ClA = other.spec.downforce_ClA
        if random.random() < 0.5: child_spec.drag_area = other.spec.drag_area
        if random.random() < 0.5: child_spec.mu = other.spec.mu
        if random.random() < 0.5: child_spec.durability = other.spec.durability
        if random.random() < 0.5: child_spec.final_drive = other.spec.final_drive
        if random.random() < 0.5: child_spec.cg_height = other.spec.cg_height
        
        # Crossover gears by taking average to avoid breaking monotonic order
        child_gears = []
        for g1, g2 in zip(self.spec.gear_ratios, other.spec.gear_ratios):
            child_gears.append((g1 + g2) / 2.0)
        child_spec.gear_ratios = child_gears
        
        return CarGenome(child_spec)

