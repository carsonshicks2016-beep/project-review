"""Interpretation of the SMW RAM variables exposed by our data.json.

Every constant here was confirmed against a live emulator with
`python -m smwrl.probe_ram` rather than taken from a memory map on faith.
"""

from __future__ import annotations

from dataclasses import dataclass

# game_mode values ($7E0100). 0x14 is normal in-level gameplay; the death and
# level-transition sequences walk through the others.
MODE_LEVEL = 0x14

# player_anim ($7E0071). 9 is the death animation -- observed firing exactly one
# frame before `lives` decremented.
ANIM_DYING = 0x09

# y_pos is signed: Mario reads negative when above the top of the level.
Y_SIGNED_LIMIT = 32768


@dataclass(frozen=True)
class GameState:
    """A decoded snapshot of the emulator's RAM for one frame."""

    x: int
    y: int
    game_mode: int
    player_anim: int
    lives: int
    coins: int
    score: int
    powerup: int
    end_level_timer: int
    secret_exit: bool
    overworld_exit_mode: int
    timer: int
    translevel: int

    @property
    def in_level(self) -> bool:
        return self.game_mode == MODE_LEVEL

    @property
    def dying(self) -> bool:
        return self.player_anim == ANIM_DYING

    @property
    def cleared(self) -> bool:
        """Goal tape or keyhole triggered: the end-of-level timer starts running."""
        return self.end_level_timer != 0


def decode(info: dict) -> GameState:
    """Turn a stable-retro info dict into a GameState.

    stable-retro hands back an empty dict on reset(), so missing keys default
    to 0 rather than raising.
    """
    y = int(info.get("y_pos", 0))
    if y >= Y_SIGNED_LIMIT:  # stored unsigned; negative means above the level
        y -= 65536
    return GameState(
        x=int(info.get("x_pos", 0)),
        y=y,
        game_mode=int(info.get("game_mode", 0)),
        player_anim=int(info.get("player_anim", 0)),
        lives=int(info.get("lives", 0)),
        coins=int(info.get("coins", 0)),
        score=int(info.get("score", 0)),
        powerup=int(info.get("powerup", 0)),
        end_level_timer=int(info.get("end_level_timer", 0)),
        secret_exit=bool(info.get("secret_exit", 0)),
        overworld_exit_mode=int(info.get("overworld_exit_mode", 0)),
        timer=(
            int(info.get("timer_hundreds", 0)) * 100
            + int(info.get("timer_tens", 0)) * 10
            + int(info.get("timer_ones", 0))
        ),
        translevel=int(info.get("translevel", 0)),
    )
