"""Run configuration. Defaults live here; config.local.json overrides them per machine."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / 'runs'
RUNTIME = ROOT / '.runtime'

MAINLINE = ROOT / 'vendor/mainline/Slippi_Dolphin.app/Contents/MacOS/Slippi_Dolphin'
ISHIIRUKA = ROOT / 'vendor/Slippi Dolphin.app/Contents/MacOS/Slippi Dolphin'

# Opponents the CPU ladder draws from. Weighted toward the characters a Jigglypuff
# actually has to beat; Sheik is absent because the CSS only offers Zelda and the
# transform needs a held input libmelee's helper does not perform.
DEFAULT_OPPONENTS = {
    'FOX': 3.0, 'FALCO': 2.0, 'MARTH': 2.0, 'CPTFALCON': 1.5, 'PEACH': 1.0,
    'JIGGLYPUFF': 1.0, 'SAMUS': 0.75, 'PIKACHU': 0.75, 'LUIGI': 0.5, 'DOC': 0.5,
    'YOSHI': 0.5, 'GANONDORF': 0.5, 'MARIO': 0.5, 'LINK': 0.25,
}


DEFAULT_PHILLIP = {
    'FOX': 3.0, 'FALCO': 2.0, 'MARTH': 2.0, 'CPTFALCON': 1.5, 'PEACH': 1.0, 'JIGGLYPUFF': 1.0,
    'SAMUS': 0.75, 'PIKACHU': 0.75, 'YOSHI': 0.5, 'LUIGI': 0.5,
}


@dataclass
class Config:
    # --- machine ---------------------------------------------------------------
    dolphin: str = str(MAINLINE)
    iso: str = ''

    # --- game --------------------------------------------------------------------
    character: str = 'JIGGLYPUFF'
    stage: str = 'FINAL_DESTINATION'
    # Frames per decision. A macro that needs longer (a full hop) runs to completion.
    act_every: int = 3
    # Partial trigger pulses during falling aerials (measured: halves landing lag, never
    # air-dodges). Applies to every policy-controlled player alike.
    auto_lcancel: bool = True
    # Tech in place automatically at the frame the game accepts (down-throws, tumble
    # landings over the stage); the policy's tech macros choose a roll direction instead.
    auto_tech: bool = True
    # Safety cap in frames. The game's own 8:00 timer normally ends the match first.
    max_game_frames: int = 8 * 60 * 60 + 600

    # --- emulators ---------------------------------------------------------------
    workers: int = 6
    # Workers rendered in a window you can watch; the rest use the Null video backend.
    visible_workers: int = 0
    # 0 = unlimited. Only visible workers honour a non-zero value, so watching one game
    # at real time never slows the others (workers are fully independent).
    visible_speed: float = 0.0
    viewer_audio: bool = False          # sound in visible windows (the watch command turns it on)
    base_port: int = 55441
    # Instant Match keeps replaying a setup with no menus. Re-roll the opponent after
    # this many games so every worker still sees the whole roster.
    games_per_setup: int = 4
    save_replays: bool = False

    # --- opponents ---------------------------------------------------------------
    opponents: dict = field(default_factory=lambda: dict(DEFAULT_OPPONENTS))
    cpu_start_level: int = 3
    cpu_min_level: int = 1
    cpu_max_level: int = 9
    # Promotion window and thresholds for the CPU frontier.
    ladder_window: int = 40
    promote_at: float = 0.65
    demote_at: float = 0.20
    # Share of games against our own policy (current or a frozen snapshot).
    selfplay_fraction: float = 0.2
    # Of self-play games, the share where both sides are the live policy. Both sides
    # then produce training data, which doubles that emulator's useful output.
    mirror_fraction: float = 0.5
    snapshot_every_frames: int = 10_000_000
    max_snapshots: int = 12
    # Phillip (Slippi-AI medium-v2): a learned opponent imitating human replays and then
    # tuned by RL, 21-frame reaction delay. Workers 0..phillip_workers-1 always play it,
    # each keeping one model loaded in a child process (about one CPU core each).
    phillip_workers: int = 0
    phillip_python: str = str(ROOT / 'vendor/slippi-ai/.venv/bin/python')
    phillip_model: str = str(ROOT / 'vendor/slippi-ai-models/medium-v2')
    phillip_characters: dict = field(default_factory=lambda: dict(DEFAULT_PHILLIP))

    # --- stalling watch --------------------------------------------------------------
    # Share of game time spent offstage over the last `stall_window` finished games. The
    # 2026-09-26 run stayed at 25-48% while healthy and went to 65-78% while stalling.
    stall_window: int = 100
    stall_offstage_warn: float = 0.50
    stall_offstage_alarm: float = 0.60

    # --- automatic scorecards ---------------------------------------------------------
    # Every `scorecard_hours` of wall time the newest checkpoint plays the fixed scorecard
    # (Fox, Falco, Marth, Falcon, Peach, Puff x `scorecard_games` vs CPU `scorecard_level`)
    # on `scorecard_workers` extra emulators beside training. 0 hours turns it off.
    scorecard_hours: float = 3.0
    scorecard_workers: int = 2
    scorecard_games: int = 10
    scorecard_level: int = 9

    # --- learning ------------------------------------------------------------------
    # Per frame: 11.5 s half-life (16.7 s time constant). At 0.997 (3.85 s) a stock lost
    # four seconds later cost half as much, and the 2026-09-26 run learned to hide
    # offstage to postpone losses it could not prevent (win rate vs CPU 9: 60% -> 10%).
    gamma: float = 0.999
    # Updates that train only the value side after starting from weights whose planning
    # horizon differs or is unrecorded: the policy stays exactly as it was meanwhile,
    # instead of following advantages from a value estimate still scaled for the old one.
    critic_warmup_updates: int = 150
    lam: float = 0.95
    fragment: int = 128           # decisions per trajectory fragment sent to the learner
    batch_steps: int = 4096       # decisions per learner update
    epochs: int = 2
    minibatch: int = 1024
    learning_rate: float = 3e-4
    entropy: float = 0.01
    entropy_final: float = 0.003
    entropy_anneal_frames: int = 300_000_000
    clip: float = 0.2
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    rho_clip: float = 1.0
    max_policy_lag: int = 4       # updates; older fragments are discarded and counted
    hidden: int = 512
    device: str = 'auto'          # auto | cpu | mps
    checkpoint_minutes: float = 10.0
    total_frames: int = 0         # 0 = run until stopped
    seed: int = 7

    def validate(self) -> 'Config':
        import melee
        if not Path(self.dolphin).is_file():
            raise ValueError(f'Dolphin executable not found: {self.dolphin}')
        if not Path(self.iso).is_file():
            raise ValueError(f'Melee ISO not found: {self.iso}. Set "iso" in config.local.json.')
        melee.Character[self.character]
        melee.Stage[self.stage]
        for name in self.opponents:
            melee.Character[name]
        if self.stage != 'FINAL_DESTINATION':
            raise ValueError('Only Final Destination is supported so far (stage geometry is hard-coded).')
        if not 1 <= self.workers <= 16:
            raise ValueError('workers must be 1-16.')
        if not 0 <= self.visible_workers <= self.workers:
            raise ValueError('visible_workers must be between 0 and workers.')
        if not 1 <= self.act_every <= 8:
            raise ValueError('act_every must be 1-8 frames.')
        if not 1 <= self.cpu_min_level <= self.cpu_start_level <= self.cpu_max_level <= 9:
            raise ValueError('CPU levels must satisfy 1 <= min <= start <= max <= 9.')
        if not 0 <= self.selfplay_fraction <= 1 or not 0 <= self.mirror_fraction <= 1:
            raise ValueError('selfplay_fraction and mirror_fraction must be in [0, 1].')
        if self.batch_steps < self.fragment or self.minibatch > self.batch_steps:
            raise ValueError('Need fragment <= batch_steps and minibatch <= batch_steps.')
        if not 0 <= self.phillip_workers <= self.workers:
            raise ValueError('phillip_workers must be between 0 and workers.')
        if self.phillip_workers:
            from .phillip import SUPPORTED
            for path in (self.phillip_python, self.phillip_model):
                if not Path(path).exists():
                    raise ValueError(f'Phillip needs {path} (see vendor/slippi-ai setup).')
            unknown = set(self.phillip_characters) - set(SUPPORTED)
            if unknown or not self.phillip_characters:
                raise ValueError(f'phillip_characters must be from {SUPPORTED}; got {sorted(unknown)}.')
        if not 0 < self.gamma < 1 or not 0 <= self.lam <= 1:
            raise ValueError('gamma must be in (0,1) and lam in [0,1].')
        if self.scorecard_hours < 0 or self.scorecard_workers < 1 or self.scorecard_games < 1 \
                or not 1 <= self.scorecard_level <= 9:
            raise ValueError('scorecard_hours >= 0, scorecard_workers/games >= 1, scorecard_level 1-9.')
        if self.critic_warmup_updates < 0:
            raise ValueError('critic_warmup_updates must be >= 0.')
        if self.stall_window < 10 or not 0 < self.stall_offstage_warn <= self.stall_offstage_alarm <= 1:
            raise ValueError('stall_window must be >= 10 and 0 < stall_offstage_warn <= stall_offstage_alarm <= 1.')
        return self

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides) -> 'Config':
        path = Path(path or ROOT / 'config.local.json')
        data = json.loads(path.read_text()) if path.exists() else {}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f'Unknown config keys in {path.name}: {sorted(unknown)}')
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
