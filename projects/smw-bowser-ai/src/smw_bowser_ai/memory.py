from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import default_config_path, load_config
from .protocol import BridgeObservation


class MemoryMapError(RuntimeError):
    """Raised when the memory map config is invalid."""


@dataclass(frozen=True)
class MemorySignal:
    name: str
    addr: int
    kind: str
    required: bool = False
    length: int = 1
    description: str = ""


class MemoryMap:
    def __init__(
        self,
        version: str,
        domain: str,
        signals: dict[str, MemorySignal],
        mode_values: dict[str, set[int]],
        starworld_detection: dict[str, Any],
    ):
        self.version = version
        self.domain = domain
        self.signals = signals
        self.mode_values = mode_values
        self.starworld_detection = starworld_detection

    @classmethod
    def from_file(cls, path: str | Path | None = None) -> "MemoryMap":
        data = load_config(path or default_config_path("memory_map.yaml"))
        signals: dict[str, MemorySignal] = {}
        for name, entry in data.get("signals", {}).items():
            addr = entry.get("addr")
            if isinstance(addr, str):
                addr_int = int(addr, 16) if addr.lower().startswith("0x") else int(addr)
            elif isinstance(addr, int):
                addr_int = addr
            else:
                raise MemoryMapError(f"signal {name} has invalid addr {addr!r}")
            signals[str(name)] = MemorySignal(
                name=str(name),
                addr=addr_int,
                kind=str(entry.get("type", "u8")),
                required=bool(entry.get("required", False)),
                length=int(entry.get("length", 1)),
                description=str(entry.get("description", "")),
            )
        mode_values = {
            str(name): {int(value) for value in values}
            for name, values in data.get("mode_values", {}).items()
        }
        return cls(
            version=str(data.get("version", "unknown")),
            domain=str(data.get("domain", "WRAM")),
            signals=signals,
            mode_values=mode_values,
            starworld_detection=data.get("starworld_detection", {}),
        )

    def required_signal_names(self) -> list[str]:
        return [signal.name for signal in self.signals.values() if signal.required]

    def classify_mode(self, ram: dict[str, Any]) -> str:
        mode = ram.get("game_mode")
        if not isinstance(mode, int):
            return "unknown"
        for name, values in self.mode_values.items():
            if mode in values:
                return name
        return "unknown"

    def missing_required(self, ram: dict[str, Any]) -> list[str]:
        return [name for name in self.required_signal_names() if name not in ram]


def progress_scalar(observation: BridgeObservation) -> int:
    ram = observation.ram
    screen_x = int(ram.get("screen_x", 0) or 0)
    mario_x = int(ram.get("mario_x", 0) or 0)
    return max(screen_x, mario_x)


def lives_decreased(previous: BridgeObservation | None, current: BridgeObservation) -> bool:
    if previous is None:
        return False
    prev_lives = previous.ram.get("lives")
    curr_lives = current.ram.get("lives")
    return isinstance(prev_lives, int) and isinstance(curr_lives, int) and curr_lives < prev_lives


def likely_death(previous: BridgeObservation | None, current: BridgeObservation) -> bool:
    if lives_decreased(previous, current):
        return True
    player_state = current.ram.get("player_state")
    game_mode = current.ram.get("game_mode")
    # Common SMW death animation state candidate; calibration can tighten this.
    return player_state in {0x09, 0x0B} and game_mode in {0x14, 0x0F, 0x10, 0x11, 0x12, 0x13}


def level_cleared(current: BridgeObservation) -> bool:
    goal_state = current.ram.get("goal_state")
    return isinstance(goal_state, int) and goal_state > 0

