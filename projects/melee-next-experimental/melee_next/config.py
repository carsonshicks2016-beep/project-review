from dataclasses import dataclass, asdict, field
from pathlib import Path
import os
import shutil
from .storage import ROOT, read

ROSTER = ['FOX', 'FALCO', 'MARTH', 'SHEIK', 'CPTFALCON', 'PEACH']

@dataclass
class Config:
    dolphin: str = ''
    iso: str = ''
    character: str = 'JIGGLYPUFF'
    stage: str = 'FINAL_DESTINATION'
    opponents: list[str] = field(default_factory=lambda: list(ROSTER))
    cpu_level: int = 1
    workers: int = 2
    base_port: int = 52441
    action_frames: int = 2
    max_frames: int = 28800
    speed: float = 0.0
    seed: int = 17
    fragment: int = 128
    batch_steps: int = 2048
    total_steps: int = 100000
    epochs: int = 4
    minibatch: int = 256
    learning_rate: float = 0.0001
    gamma_frame: float = 0.997
    gae_lambda: float = 0.95
    entropy: float = 0.003
    clip: float = 0.15
    target_kl: float = 0.025
    checkpoint_every: int = 10
    save_replays: bool = False
    curriculum: bool = True
    min_disk_mb: int = 512
    backend: str = 'dolphin'
    # Synthetic backend exists solely for pipeline tests, never Melee evidence.
    mock_delay: float = 0.002
    mock_slow_worker: int = -1
    mock_slow_seconds: float = 0.0

    @classmethod
    def load(cls):
        return cls(**read(ROOT / 'config.local.json'))

    def validate(self):
        import melee
        if self.backend not in ('dolphin', 'synthetic'): raise ValueError('Unknown backend')
        if not 1 <= self.workers <= min(12, os.cpu_count() or 1): raise ValueError('Worker count must be 1–12 and fit CPU count')
        if not 1 <= self.cpu_level <= 9: raise ValueError('CPU level must be 1–9')
        if not 1 <= self.action_frames <= 8: raise ValueError('Action hold must be 1–8 frames')
        if not 60 <= self.max_frames <= 36000: raise ValueError('Match limit must be 60–36000 frames')
        if not 16 <= self.fragment <= 1024: raise ValueError('Fragment must be 16–1024 decisions')
        if self.batch_steps < self.fragment or self.batch_steps > 65536: raise ValueError('Invalid learner batch size')
        if self.total_steps < self.fragment or self.total_steps > 1_000_000_000: raise ValueError('Invalid training budget')
        if not 1 <= self.epochs <= 20 or not 16 <= self.minibatch <= 4096: raise ValueError('Invalid optimizer settings')
        if not 0 < self.learning_rate <= .01: raise ValueError('Invalid learning rate')
        if not .9 <= self.gamma_frame < 1 or not 0 < self.gae_lambda <= 1: raise ValueError('Invalid discount')
        if not 0 < self.clip < 1 or not 0 < self.target_kl < 1 or not 0 <= self.entropy < 1: raise ValueError('Invalid PPO settings')
        if not 1 <= self.checkpoint_every <= 1000: raise ValueError('Invalid checkpoint interval')
        if not 1024 <= self.base_port <= 65520 or not 0 <= self.speed <= 4: raise ValueError('Invalid port/speed')
        if not self.opponents or len(set(self.opponents)) != len(self.opponents): raise ValueError('Choose unique opponents')
        melee.Character[self.character]
        for opponent in self.opponents: melee.Character[opponent]
        if self.stage not in ('FINAL_DESTINATION', 'BATTLEFIELD', 'DREAMLAND', 'POKEMON_STADIUM', 'YOSHIS_STORY', 'FOUNTAIN_OF_DREAMS'):
            raise ValueError('Unsupported stage')
        if self.backend == 'dolphin':
            if not Path(self.dolphin).is_file() or not Path(self.iso).is_file(): raise ValueError('Dolphin and Melee image paths must exist')
        if self.min_disk_mb < 64: raise ValueError('Disk reserve must be at least 64 MB')
        return self

    def asdict(self): return asdict(self)

    def disk_ok(self):
        return shutil.disk_usage(ROOT).free >= self.min_disk_mb * 1024**2
