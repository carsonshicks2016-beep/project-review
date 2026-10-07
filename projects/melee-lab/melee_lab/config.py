from dataclasses import dataclass, asdict
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

@dataclass
class Config:
    dolphin: str = str(ROOT / 'vendor/Slippi Dolphin.app/Contents/MacOS/Slippi Dolphin')
    iso: str = ''
    character: str = 'FOX'
    opponent: str = 'MARIO'
    cpu_level: int = 1
    stage: str = 'FINAL_DESTINATION'
    stocks: int = 3
    action_frames: int = 4
    max_frames: int = 28800
    speed: float = 1.0
    port: int = 51441
    seed: int = 7
    randomize_opponent: bool = False
    action_set: str = 'legacy'
    execution_mode: str = 'assisted'
    strict_evaluation: bool = False
    league_manifest: str | None = None

    def validate(self):
        import melee
        if not Path(self.dolphin).is_file():
            raise ValueError('Slippi Dolphin executable was not found. Set dolphin in config.local.json.')
        if not Path(self.iso).is_file():
            raise ValueError('Melee game image was not found. Set iso in config.local.json.')
        if not 1 <= self.cpu_level <= 9:
            raise ValueError('CPU level must be 1–9.')
        if self.stocks != 3:
            raise ValueError('This experiment requires exactly three starting stocks.')
        if self.action_set not in ('legacy','expanded','controller'): raise ValueError('Unknown action vocabulary.')
        if self.execution_mode not in ('assisted', 'raw'): raise ValueError('Unknown execution mode.')
        if not 1 <= self.action_frames <= 12 or self.max_frames < 60:
            raise ValueError('Invalid action duration or match frame limit.')
        if not 0 <= self.speed <= 4 or not 1024 <= self.port <= 65535:
            raise ValueError('Invalid emulation speed or network port.')
        melee.Character[self.character]
        melee.Character[self.opponent]
        if self.stage != 'FINAL_DESTINATION':
            raise ValueError('Initial training and recovery teacher support Final Destination only.')

    @classmethod
    def load(cls, path=None):
        path = Path(path or ROOT / 'config.local.json')
        return cls(**json.loads(path.read_text())) if path.exists() else cls()

    def save(self, path=None):
        Path(path or ROOT / 'config.local.json').write_text(json.dumps(asdict(self), indent=2))
