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
    style_progress_gate: float = 0.35   # style also stays capped until the
                                        # policy can complete this much of a
                                        # lap — a pure update-count ramp turns
                                        # style on regardless of readiness, and
                                        # then buys drift with lap progress.
    episode_seconds: float = 60.0
    # --- run stability ---------------------------------------------------
    # A constant learning rate for a long run is what wrecked the 616-update
    # club run: it peaked at eval 154 (update 375) and fell to 16 by update 616
    # while 81% of the last 150 updates were being KL-truncated and the actor
    # weights still drifted 0.49x their own norm. Anneal, and stop throwing good
    # policies away once the run stops improving.
    anneal: bool = True                 # cosine-decay lr (and entropy) over the run
    lr_floor_frac: float = 0.15         # lr decays to this fraction by the last update
    plateau_evals: int = 6              # evals with no new best before intervening; 0 = off
    plateau_actions: int = 3            # revert-to-best + halve-lr cycles before stopping
    save_every: int = 10
    eval_every: int = 25

    @classmethod
    def parse(cls, data, strict=True):
        """strict=True for anything a person typed — an unknown key there is a
        typo and should be refused. strict=False for configs we wrote ourselves
        and are reading back (checkpoints, run folders): a field added or removed
        between versions must not make an existing checkpoint unloadable. That
        failure mode is real and it reads terribly — a lab left running while the
        code changed underneath it reported 'Unknown settings: hills' every time
        someone opened a policy, with no hint that the process was simply stale.
        Returns (cfg, dropped) when strict=False so the caller can say so."""
        if not isinstance(data, dict):
            raise ValueError('Configuration must be an object.')
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown and strict:
            raise ValueError('Unknown settings: ' + ', '.join(sorted(unknown)))
        if unknown:
            data = {k: v for k, v in data.items() if k not in unknown}
        cfg = cls(**data)
        if not isinstance(cfg.name, str) or not cfg.name.strip() or len(cfg.name) > 80:
            raise ValueError('Give the run a name of 1–80 characters.')
        if cfg.car not in CARS or cfg.track not in TRACKS or cfg.mode not in ('race', 'drift', 'hybrid'):
            raise ValueError('Unknown car, track, or driving mode.')
        if type(cfg.hills) is not bool or type(cfg.anneal) is not bool:
            raise ValueError('hills and anneal must be true or false.')
        limits = dict(seed=(0, 2147483647), updates=(1, 1000000), environments=(1, 16),
                      rollout=(16, 4096), epochs=(1, 20), batch_size=(8, 4096),
                      style_warmup=(0, 100000), save_every=(1, 1000), eval_every=(1, 10000),
                      plateau_evals=(0, 1000), plateau_actions=(0, 100))
        for key, (lo, hi) in limits.items():
            value = getattr(cfg, key)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f'{key} must be an integer from {lo} to {hi}.')
        limits = dict(learning_rate=(1e-6, .01), gamma=(.8, .9999), gae_lambda=(.5, 1),
                      clip=(.01, .5), entropy=(0, .1), target_kl=(.001, .2),
                      style_weight=(0, .3), episode_seconds=(5, 600),
                      style_progress_gate=(0, 1), lr_floor_frac=(.01, 1))
        for key, (lo, hi) in limits.items():
            value = getattr(cfg, key)
            if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f'{key} must be from {lo} to {hi}.')
        return cfg if strict else (cfg, sorted(unknown))

    def dict(self):
        return asdict(self)
