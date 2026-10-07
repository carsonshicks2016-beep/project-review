"""Validated, versioned configuration for the standalone hybrid laboratory."""
from dataclasses import asdict, dataclass
import math

TRACKS = ('club', 'national', 'coast', 'sprint', 'tech', 'oval2', 'akina', 'pass')
CARS = ('supra', 'rx7', 'skyline')

@dataclass
class Config:
    name: str = 'Hybrid experiment'
    car: str = 'supra'
    track: str = 'club'
    mode: str = 'hybrid'
    hills: bool = True
    seed: int = 42
    updates: int = 1000
    environments: int = 4
    rollout: int = 256
    epochs: int = 4
    batch_size: int = 128
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy: float = 0.005
    target_kl: float = 0.025
    style_weight: float = 0.03
    style_warmup: int = 100
    episode_seconds: float = 60.0
    save_every: int = 10
    eval_every: int = 25

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict):
            raise ValueError('Configuration must be an object.')
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError('Unknown settings: ' + ', '.join(sorted(unknown)))
        cfg = cls(**data)
        if not isinstance(cfg.name, str) or not cfg.name.strip() or len(cfg.name) > 80:
            raise ValueError('Give the run a name of 1–80 characters.')
        if cfg.car not in CARS or cfg.track not in TRACKS or cfg.mode not in ('race', 'drift', 'hybrid'):
            raise ValueError('Unknown car, track, or driving mode.')
        if type(cfg.hills) is not bool:
            raise ValueError('hills must be true or false.')
        limits = dict(seed=(0, 2147483647), updates=(1, 1000000), environments=(1, 16),
                      rollout=(16, 4096), epochs=(1, 20), batch_size=(8, 4096),
                      style_warmup=(0, 100000), save_every=(1, 1000), eval_every=(1, 10000))
        for key, (lo, hi) in limits.items():
            value = getattr(cfg, key)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f'{key} must be an integer from {lo} to {hi}.')
        limits = dict(learning_rate=(1e-6, .01), gamma=(.8, .9999), gae_lambda=(.5, 1),
                      clip=(.01, .5), entropy=(0, .1), target_kl=(.001, .2),
                      style_weight=(0, .3), episode_seconds=(5, 600))
        for key, (lo, hi) in limits.items():
            value = getattr(cfg, key)
            if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f'{key} must be from {lo} to {hi}.')
        return cfg

    def dict(self):
        return asdict(self)
