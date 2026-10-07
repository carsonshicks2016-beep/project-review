"""Driver 2.0 configuration dataclasses.

All tunable knobs for the LSTM brain, GA evolution, curriculum schedule,
and evaluation protocol live here.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BrainConfig:
    """LSTM brain architecture."""
    obs_size: int = 68            # fable-v2 observation (58 base + 8 pace + 2 hybrid)
    proj_size: int = 48           # input projection width
    hidden_size: int = 48         # LSTM hidden state width
    n_actions: int = 3            # steer, longitudinal, gear_offset
    bias_long: float = 0.6       # initial longitudinal bias (pre-tanh → mild throttle)


@dataclass
class GAConfig:
    """Genetic algorithm hyperparameters."""
    pop_size: int = 200           # population size
    elite_frac: float = 0.10     # top 10% carry over unchanged
    tournament_k: int = 5        # tournament selection pressure
    mutation_rate: float = 0.06  # per-gene mutation probability
    mutation_sigma: float = 0.12 # Gaussian perturbation std
    workers: int = 0             # 0 = use all CPU cores
    control_hz: float = 30.0     # brain inference rate (physics runs at 120 Hz)


@dataclass
class CurriculumConfig:
    """Progressive condensation schedule."""
    n_initial_segments: int = 64
    condense_mastery_frac: float = 0.90   # fraction of segments that must be mastered
    condense_pace_threshold: float = 0.70  # min pace ratio to count as mastered
    condense_clean_required: bool = True   # mastered = clean (no offtrack)
    min_generations_per_level: int = 50    # minimum gens before condensation allowed
    weak_segment_weight: float = 2.0       # extra weight on weakest segments in fitness
    segment_time_budget_factor: float = 2.0  # time budget = segment_length / 30 * factor


@dataclass
class EvalConfig:
    """Deterministic full-lap evaluation."""
    n_sectors: int = 16           # sector evaluation count
    sector_seconds: float = 30.0  # time budget per sector eval
    lap_budget: float = 900.0     # max seconds for a flying-lap attempt
    n_eval_laps: int = 3          # deterministic repeated evaluations
    rng_seed: int = 42            # reproducibility


@dataclass
class Driver2Config:
    """Top-level configuration."""
    car: str = "porsche_919evo"
    track: str = "nordschleife"
    brain: BrainConfig = field(default_factory=BrainConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    artifact_dir: str = "runtime/driver2"

    # Benchmarks for reference (not used in training logic)
    champion_time: float = 437.73    # Fable PPO champion (frozen)
    classical_time: float = 385.62   # Raceline controller
    bellof_time: float = 371.13      # Stefan Bellof 1983
    record_time: float = 319.55      # 919 Evo 2018 outright

    # Physics timestep (must match supra.config.SimSpec default)
    dt: float = 1.0 / 120.0
