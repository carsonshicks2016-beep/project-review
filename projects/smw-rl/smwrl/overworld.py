"""Boot the game from power-on and drive the overworld map.

None of this is reinforcement learning. Menus and map movement are
deterministic, so they are scripted -- putting a neural net on a file-select
screen would burn compute to learn a fixed button sequence.

Every constant here was found by driving the real game, not read off a memory
map:

* `MODE_*` came from logging `game_mode` through a real boot.
* `OW_X` / `OW_Y` came from a differential RAM scan: hold RIGHT, hold LEFT, and
  keep the addresses that move +48 one way and -48 the other while staying
  still when idle. 0x7E1F17 and 0x7E1F19 were the only non-mirror hits.
* One directional tap moves exactly one map node.

`translevel` (0x7E13BF) deliberately is *not* used to identify map nodes -- it
stays 0 on the overworld and is only set once a level loads. Nodes are
identified by their (x, y) map coordinates instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- verified RAM offsets (index into get_ram(), i.e. 0x7E0000 + offset) ----
OW_X = 0x1F17
OW_Y = 0x1F19
OW_PROCESS = 0x13D9
OW_PLAYER_ANIMATION = 0x1F13
# Backward-compatible names for existing callers/tests.
OW_ANIM = OW_PROCESS
OW_CONTROL = OW_PLAYER_ANIMATION
TRANSLEVEL = 0x13BF
LIVES = 0x0DBE
GAME_MODE = 0x0100

# --- verified game_mode values ---------------------------------------------
MODE_UNINIT = 0x55        # RAM before the game initialises anything
MODE_TITLE_MAX = 0x0C     # everything <= this is logo / title / file select
MODE_OVERWORLD = 0x0E
MODE_LEVEL = 0x14

# OW_PROCESS value while the CONTINUE AND SAVE / CONTINUE WITHOUT SAVE prompt is
# up -- the game shows it after a switch palace or a castle. The map never
# becomes controllable until it is answered, and idling will not answer it:
# verified over 6,000 NOOP frames with the process byte never leaving 5. A takes
# it to 3 in 28 frames (the cursor starts on CONTINUE AND SAVE); START does
# nothing at all.
SAVE_PROMPT = 5

# OW_PLAYER_ANIMATION values that mean "standing still and accepting input".
# An allowlist rather than a loosened test, because requiring a specific value is
# what stopped commands being sent during the map's quiet lead-in. 2 was the
# original live-verified value; 10 is Mario standing on an uncleared level tile,
# and demanding 2 there rejected a map that is demonstrably controllable -- at
# Yoshi's Island 4 (184, 72) with animation 10, DOWN moves to (152, 136) and A
# enters the level.
READY_ANIMATIONS = (2, 10)

BUTTONS = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]


def buttons(*names: str) -> np.ndarray:
    a = np.zeros(12, np.uint8)
    for n in names:
        a[BUTTONS.index(n)] = 1
    return a


NOOP = buttons()


def read(env, offset: int) -> int:
    return int(env.get_ram()[offset])


def read_u16(env, offset: int) -> int:
    ram = env.get_ram()
    return int(ram[offset]) | (int(ram[offset + 1]) << 8)


def mode(env) -> int:
    return read(env, GAME_MODE)


def position(env) -> tuple[int, int]:
    return read_u16(env, OW_X), read_u16(env, OW_Y)


def _step(env, action, n: int = 1):
    for _ in range(n):
        env.step(action)


def wait_for_mode(env, target: int, timeout: int = 900, action=NOOP) -> bool:
    for _ in range(timeout):
        env.step(action)
        if mode(env) == target:
            return True
    return False


@dataclass
class BootResult:
    ok: bool
    frames: int
    detail: str


def boot_to_map(env, timeout: int = 12_000) -> BootResult:
    """Power-on -> menus -> Yoshi's House -> out onto the overworld map.

    Mirrors what a person does: mash through the logo and title, start a
    1-player game (which drops Mario onto the map standing on Yoshi's House and
    walks him straight in), read the welcome message, then walk back out.

    The title screen at mode 0x07 ignores START and only responds to A, so the
    two are alternated. Before the game initialises, `game_mode` reads 0x55 --
    treating that as "past the menus" makes the boot stop at frame 0 and sit in
    the attract-mode demo, which is also mode 0x14.
    """
    env.reset()
    frames = 0

    # A no-state stable-retro environment starts with uninitialised RAM (0x55).
    # reset() does not power-cycle such an environment a second time, and a
    # level save state starts directly in mode 0x14.  Refuse both cases instead
    # of mistaking an existing/dead run for a new game.
    if mode(env) != MODE_UNINIT:
        return BootResult(
            False, 0,
            "environment is not at power-on; create it with make_raw_env(None) "
            "and call boot_to_map only once",
        )

    # -- phase 1: through the menus until a level actually loads -------------
    start, a_btn = buttons("START"), buttons("A")
    while frames < timeout:
        m = mode(env)
        if m == MODE_LEVEL and read(env, LIVES) > 0:
            break
        in_menus = m == MODE_UNINIT or m <= MODE_TITLE_MAX
        act = NOOP
        if in_menus and frames % 24 < 3:
            act = start if (frames // 24) % 2 == 0 else a_btn
        env.step(act)
        frames += 1
    else:
        return BootResult(False, frames, "never reached gameplay from the menus")

    # -- phase 2: dismiss Yoshi's welcome text and walk out to the map -------
    # B has to be tapped throughout, not just at the start: phase 1 exits the
    # moment the level loads, which is *before* the welcome box appears, and an
    # open message box blocks movement entirely.
    b_btn, right = buttons("B"), buttons("RIGHT", "Y")
    for i in range(3000):
        env.step(b_btn if i % 20 < 3 else right)
        frames += 1
        if mode(env) == MODE_OVERWORLD:
            try:
                wait_settled(env)         # Mario is still walking out of the house
            except TimeoutError as e:
                return BootResult(False, frames, str(e))
            return BootResult(True, frames, "on the overworld")
    return BootResult(False, frames, "left the house but never reached the map")


def wait_settled(env, stable_frames: int = 45, timeout: int = 2400,
                 min_frames: int = 0) -> tuple[int, int]:
    """Idle until the map position stops changing.

    Mario keeps walking for a while after exiting a level or a house, and any
    direction pressed during that animation is swallowed -- which looks exactly
    like "there is no path that way".  More subtly, after a normal exit the map
    can sit motionless for about 190 frames before the path-reveal animation
    starts.  A stability-only test returned during that quiet lead-in and sent
    the next command to the old node.  The live-verified ready state is
    OW_ANIM=3 and OW_CONTROL=2; require it as well as coordinate stability.
    """
    last = position(env)
    same = 0
    for frame in range(timeout):
        # Answer a save prompt rather than idling in front of it forever. This
        # is guarded on the process byte on purpose: A on a *controllable* map
        # enters a level, which is the last thing a "wait" should do. Without
        # this, every state captured after a switch palace was unusable -- the
        # two cells holding all the banked game progress failed 57 excursions
        # out of 57, and the search could not advance past them.
        blocked = read(env, OW_PROCESS) == SAVE_PROMPT
        env.step(buttons("A") if blocked and frame % 16 < 4 else NOOP)
        p = position(env)
        if p == last:
            same += 1
        else:
            same, last = 0, p
        ready = (mode(env) == MODE_OVERWORLD and read(env, OW_PROCESS) == 3
                 and read(env, OW_PLAYER_ANIMATION) in READY_ANIMATIONS)
        if frame + 1 >= min_frames and same >= stable_frames and ready:
            return last
    raise TimeoutError(
        f"overworld never became controllable in {timeout} frames: "
        f"mode=0x{mode(env):02X}, position={last}, process={read(env, OW_PROCESS)}, "
        f"player_animation={read(env, OW_PLAYER_ANIMATION)}"
    )


def move(env, direction: str, settle: int = 150) -> tuple[int, int]:
    """Tap a direction and let Mario walk to the next map node.

    Returns the resulting (x, y). If the position is unchanged there was no
    path that way -- the caller should treat that as a routing error rather
    than retrying forever.
    """
    if direction not in ("UP", "DOWN", "LEFT", "RIGHT"):
        raise ValueError(f"bad direction: {direction}")
    wait_settled(env)
    _step(env, buttons(direction), 4)
    _step(env, NOOP, settle)
    return wait_settled(env)


def enter_level(env, timeout: int = 600) -> bool:
    """Press A on the current node and wait for the level to load."""
    for i in range(timeout):
        env.step(buttons("A") if i % 16 < 3 else NOOP)
        if mode(env) == MODE_LEVEL:
            _step(env, NOOP, 20)
            return True
    return False


def wait_for_map(env, timeout: int = 2000) -> bool:
    """After a level ends, wait until control returns to the overworld."""
    return wait_for_mode(env, MODE_OVERWORLD, timeout)
