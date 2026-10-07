from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import default_config_path, load_config
from .protocol import BridgeAction


class ActionSpaceError(RuntimeError):
    """Raised when a macro action is unknown or malformed."""


@dataclass(frozen=True)
class MacroAction:
    name: str
    hold_frames: int
    buttons: tuple[str, ...]

    def to_bridge_action(self, *, note: str = "", evaluation_lock: bool = False) -> BridgeAction:
        return BridgeAction(
            buttons=self.buttons,
            frames=self.hold_frames,
            macro=self.name,
            evaluation_lock=evaluation_lock,
            note=note,
        )


class ActionSpace:
    def __init__(self, version: str, allowed_buttons: set[str], macros: dict[str, MacroAction]):
        self.version = version
        self.allowed_buttons = allowed_buttons
        self.macros = macros

    @classmethod
    def from_file(cls, path: str | Path | None = None) -> "ActionSpace":
        data = load_config(path or default_config_path("action_space.yaml"))
        allowed = {str(button) for button in data.get("allowed_buttons", [])}
        macros: dict[str, MacroAction] = {}
        for entry in data.get("macros", []):
            name = str(entry["name"])
            buttons = tuple(str(button) for button in entry.get("buttons", []))
            unknown = set(buttons) - allowed
            if unknown:
                raise ActionSpaceError(f"macro {name} uses unknown buttons: {sorted(unknown)}")
            macros[name] = MacroAction(
                name=name,
                hold_frames=max(1, int(entry.get("hold_frames", 1))),
                buttons=buttons,
            )
        if "idle" not in macros:
            raise ActionSpaceError("action space must define an idle macro")
        return cls(version=str(data.get("version", "unknown")), allowed_buttons=allowed, macros=macros)

    def names(self) -> list[str]:
        return list(self.macros)

    def get(self, name: str) -> MacroAction:
        try:
            return self.macros[name]
        except KeyError as error:
            raise ActionSpaceError(f"unknown macro action: {name}") from error

    def bridge_action(self, name: str, *, note: str = "", evaluation_lock: bool = False) -> BridgeAction:
        return self.get(name).to_bridge_action(note=note, evaluation_lock=evaluation_lock)

    def noop(self, *, note: str = "noop") -> BridgeAction:
        return self.bridge_action("idle", note=note)

