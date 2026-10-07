"""Construction of the raw stable-retro emulator environment.

Everything funnels through here so the custom integration path (our own
data.json, scenario.json, ROM and save states) is registered exactly once and
`render_mode` is always explicit -- stable-retro defaults to opening a pyglet
window, which both costs throughput and crashes on close under macOS.
"""

from __future__ import annotations

import os
import hashlib
from pathlib import Path

import stable_retro as retro

GAME = "SuperMarioWorld-Snes-v0"
EXPECTED_ROM_SHA1 = "6b47bb75d16514b6a476aa0c73a683a2a4c18765"

INTEGRATION_DIR = Path(__file__).resolve().parent.parent / "integration"

_registered = False


def register_integration() -> None:
    """Point stable-retro at our in-repo integration directory."""
    global _registered
    if _registered:
        return
    rom = INTEGRATION_DIR / GAME / "rom.sfc"
    if not rom.exists():
        raise FileNotFoundError(
            f"No ROM at {INTEGRATION_DIR / GAME / 'rom.sfc'}.\n"
            "Supply your own legally-obtained Super Mario World (USA) dump; "
            "expected SHA-1 6b47bb75d16514b6a476aa0c73a683a2a4c18765."
        )
    actual = hashlib.sha1(rom.read_bytes()).hexdigest()
    if actual != EXPECTED_ROM_SHA1:
        raise ValueError(
            f"Wrong ROM at {rom}: SHA-1 {actual}, expected {EXPECTED_ROM_SHA1}. "
            "Use the headerless Super Mario World (USA) dump."
        )
    retro.data.Integrations.add_custom_path(os.fspath(INTEGRATION_DIR))
    _registered = True


def list_states() -> list[str]:
    return sorted(p.stem for p in (INTEGRATION_DIR / GAME).glob("*.state"))


def make_raw_env(state: str | None = "YoshiIsland1", render_mode: str | None = None):
    """A bare emulator env: full 12-button MultiBinary actions, RGB frames."""
    register_integration()
    return retro.make(
        GAME,
        state=state,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        render_mode=render_mode,
    )
