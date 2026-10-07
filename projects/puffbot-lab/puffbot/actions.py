"""Jigglypuff's action vocabulary: 46 macros instead of a raw controller.

Version one's Jigglypuff played a *factored* controller: every button, trigger and stick
angle re-sampled independently 60 times a second. Under a near-uniform starting policy
that presses the shield trigger in the air about one frame in eight, and a shield press
in the air is an air-dodge, which offstage is helpless fall and a lost stock. Measured,
68-87% of that Puff's deaths were self-destructs, and the policy never learned to stop.

So the unit of choice here is a *technique*, the way the original Melee RL work that
reached pro level did it: a short hop, a bair drifting in, Rest, a roll. Each macro is a
short script of controller frames; directions are absolute (left/right of the screen),
and the policy sees which way it faces.

Two things are left out on purpose, symmetric for every player the policy controls:

* No free air-dodge. Shield actions are legal only on the ground or on the ledge.
  Jigglypuff has no use for an air-dodge on Final Destination that is worth the stocks.
* No neutral-B (Rollout) or up-B (Sing): both are near-useless and Rollout offstage is
  a self-destruct.

Rest is legal anywhere except offstage, where it is only ever a suicide.

Step 2 (2026-09-25) added L-cancelling and teching, from measurements in this exact
Dolphin build (mainline Slippi 4.0 beta 19) rather than assumptions:

* A *partial* (analog-only) trigger press never air-dodged (0 of 10 at 0.3-1.0) and
  L-cancelled exactly like a full click (short-hop nair landing lag 20 -> 10 frames, 3/3).
  Only the digital click air-dodges. So the executor L-cancels automatically: during an
  aerial attack while falling it pulses a partial press on alternate frames.
* Teching needs the digital click; a partial press never teched (0 of 36). A click
  during Fox's 32-frame down-throw teched 10/10 from throw frame 14 on, and a click in
  tumble within 3 frames of the stage floor teched 4/4. A click higher up in tumble is
  an air-dodge, which above the stage is a waveland, not a lost stock.

The tech macros *arm* a tech (in place, or rolling left/right); the executor clicks on the
frame those probes found works: from frame TECH_THROW_FRAME of a down-throw, or when the
landing is at most TECH_LAND_FRAMES away in tumble. Letting the policy time the click
itself taught it almost nothing: 6% of knockdowns teched after ~40 game-hours, falling
to 3% (training run 20260925-221408). The click also requires Puff's position *plus her
momentum* to stay inside the stage, so a tech can never carry her offstage (one
offstage air-dodge in 373 games came from ignoring momentum).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EDGE_X = 85.5657                      # Final Destination ledge
NEUTRAL = (0.5, 0.5)
LEFT, RIGHT, UP, DOWN = (0.0, 0.5), (1.0, 0.5), (0.5, 1.0), (0.5, 0.0)
UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT = (0.15, 0.85), (0.85, 0.85), (0.15, 0.15), (0.85, 0.15)
# Tilt magnitude: enough to pick an aerial or a tilt, short of a smash input.
T_LEFT, T_RIGHT, T_UP, T_DOWN = (0.15, 0.5), (0.85, 0.5), (0.5, 0.85), (0.5, 0.15)
WALK_LEFT, WALK_RIGHT = (0.3, 0.5), (0.7, 0.5)

LEDGE_STATES = {0xFC, 0xFD}           # EDGE_CATCHING, EDGE_HANGING
THROWN_STATES = {0xEF, 0xF0, 0xF1, 0xF2, 0xF3}
KNOCKED_FLYING = {0x26} | set(range(0x57, 0x5C))   # TUMBLING, DAMAGE_FLY_*: can end in a knockdown
AERIAL_ATTACKS = {0x41, 0x42, 0x43, 0x44, 0x45}   # NAIR, FAIR, BAIR, UAIR, DAIR
TECH_HEIGHT = 30.0                   # "low over the stage" for arming a tech
TECH_MARGIN, TECH_ROLL_MARGIN = 5.0, 35.0   # distance inside the ledge; rolls travel
TECH_THROW_FRAME = 14                # Fox down-throw (32 frames): clicks from 14 on teched 10/10
TECH_LAND_FRAMES = 3                 # tumble: a click <=3 frames before landing teched 4/4
TECH_ARM_FRAMES = 45                 # an armed tech that finds no moment lapses
TECH_LOCKOUT = 40                    # Melee: a click that does not tech blocks teching for 40 frames
DOWN_THROWN = {0xF2, 0xF3}           # only a down-throw lands you straight on the floor
# Jigglypuff's air physics, for predicting when a tumble lands.
PUFF_GRAVITY, PUFF_MAX_FALL, KNOCKBACK_DECAY = 0.064, 1.3, 0.051


def landing(player, horizon: int):
    """(frames until landing on the stage floor, x on landing), or None if not within
    `horizon` frames. Knockback is reported separately from Puff's own velocity and
    decays by 0.051 per frame along its direction."""
    import math
    x, y = float(player.position.x), float(player.position.y)
    vx = float(player.speed_air_x_self)
    vy = float(player.speed_y_self)
    ax, ay = float(player.speed_x_attack), float(player.speed_y_attack)
    for t in range(1, horizon + 1):
        vy = max(vy - PUFF_GRAVITY, -PUFF_MAX_FALL)
        mag = math.hypot(ax, ay)
        if mag > 0:
            k = max(mag - KNOCKBACK_DECAY, 0.0) / mag
            ax, ay = ax * k, ay * k
        x += vx + ax
        y += vy + ay
        if y <= 0:
            return t, x
    return None
LCANCEL_ANALOG = 0.5                  # partial press: L-cancels, never air-dodges


@dataclass(frozen=True)
class Pad:
    """One frame of controller input."""
    stick: tuple = NEUTRAL
    cstick: tuple = NEUTRAL
    buttons: tuple = ()
    trigger: float = 0.0


@dataclass(frozen=True)
class Macro:
    name: str
    script: tuple = ()                # explicit frames, played first
    hold: Pad = Pad()                 # held for the rest of the decision window
    # Short-hop aerials: after the jump press, wait until airborne, then play these.
    airborne: tuple = ()
    rule: str = ''                    # '', 'grounded' (ground or ledge only), 'onstage'


def _press(buttons, stick=NEUTRAL, cstick=NEUTRAL, trigger=0.0):
    return Pad(stick, cstick, tuple(buttons), trigger)


def _build():
    m = []
    add = lambda name, **kw: m.append(Macro(name=name, **kw))
    add('noop')
    # Movement, drift, crouch / fast-fall, DI. Held for the whole window.
    for name, stick in (('left', LEFT), ('right', RIGHT), ('walk_left', WALK_LEFT),
                        ('walk_right', WALK_RIGHT), ('down', DOWN), ('up', UP),
                        ('down_left', DOWN_LEFT), ('down_right', DOWN_RIGHT),
                        ('up_left', UP_LEFT), ('up_right', UP_RIGHT)):
        add(name, hold=Pad(stick))
    # Jumps. One frame of Y is a short hop on the ground; in the air any press is a
    # mid-air jump (Jigglypuff has five). Held Y through jump-squat is a full hop.
    for name, stick in (('short_hop', NEUTRAL), ('short_hop_left', LEFT), ('short_hop_right', RIGHT)):
        add(name, script=(_press(['BUTTON_Y'], stick),), hold=Pad(stick))
    for name, stick in (('jump', NEUTRAL), ('jump_left', LEFT), ('jump_right', RIGHT)):
        add(name, script=(_press(['BUTTON_Y'], stick),) * 6, hold=Pad(stick))
    # C-stick: aerials in the air, smash attacks on the ground. Two frames of flick,
    # then neutral so the next choice can flick again.
    for name, c in (('c_left', LEFT), ('c_right', RIGHT), ('c_up', UP), ('c_down', DOWN)):
        add(name, script=(_press([], NEUTRAL, c),) * 2)
    # The same aerials while drifting: the wall of pain is bair while drifting in.
    for name, c, stick in (('c_left_drift_left', LEFT, LEFT), ('c_left_drift_right', LEFT, RIGHT),
                           ('c_right_drift_left', RIGHT, LEFT), ('c_right_drift_right', RIGHT, RIGHT)):
        add(name, script=(_press([], stick, c),) * 2, hold=Pad(stick))
    # Short hop straight into an aerial on the first airborne frame.
    jump = (_press(['BUTTON_Y']),)
    for name, c in (('sh_c_left', LEFT), ('sh_c_right', RIGHT), ('sh_c_up', UP), ('sh_c_down', DOWN)):
        add(name, script=jump, airborne=(_press([], NEUTRAL, c),) * 2)
    add('sh_nair', script=jump, airborne=(_press(['BUTTON_A']),))
    # A: jab / nair, and tilts or aerials with a tilt-strength direction.
    add('a', script=(_press(['BUTTON_A']),))
    for name, stick in (('a_left', T_LEFT), ('a_right', T_RIGHT), ('a_up', T_UP), ('a_down', T_DOWN)):
        add(name, script=(_press(['BUTTON_A'], stick),), hold=Pad(stick))
    # Specials. Rest is the kill move; Pound is recovery and edge-guarding.
    add('rest', script=(_press(['BUTTON_B'], DOWN), _press([], DOWN)), rule='onstage')
    add('pound_left', script=(_press(['BUTTON_B'], LEFT),), hold=Pad(LEFT))
    add('pound_right', script=(_press(['BUTTON_B'], RIGHT),), hold=Pad(RIGHT))
    # Defence, grounded only: a shield press in the air would be an air-dodge.
    shield = Pad(NEUTRAL, NEUTRAL, ('BUTTON_L',), 1.0)
    add('shield', hold=shield, rule='grounded')
    for name, stick in (('roll_left', LEFT), ('roll_right', RIGHT), ('spotdodge', DOWN)):
        add(name, script=(shield,), hold=Pad(stick, NEUTRAL, ('BUTTON_L',), 1.0), rule='grounded')
    add('grab', script=(_press(['BUTTON_Z']),), rule='grounded')
    # Tech: one digital click (a partial press cannot tech), stick held for a tech roll.
    # Appended last so policies trained on the first 43 macros keep their indices.
    # They arm the executor's timed click (see Executor._tech); no click in the script.
    add('tech', rule='tech')
    add('tech_left', hold=Pad(LEFT), rule='tech_roll')
    add('tech_right', hold=Pad(RIGHT), rule='tech_roll')
    return tuple(m)


MACROS = _build()
NAMES = tuple(a.name for a in MACROS)
INDEX = {n: i for i, n in enumerate(NAMES)}
COUNT = len(MACROS)
GROUNDED = np.array([a.rule == 'grounded' for a in MACROS])
ONSTAGE = np.array([a.rule == 'onstage' for a in MACROS])
TECH = np.array([a.rule == 'tech' for a in MACROS])
TECH_ROLL = np.array([a.rule == 'tech_roll' for a in MACROS])
V1_NAMES = NAMES[:43]                   # the vocabulary before step 2
MAX_FRAMES = 16                        # hard cap on one decision window


def legal_mask(player) -> np.ndarray:
    """Which macros this player may choose right now."""
    mask = np.ones(COUNT, dtype=bool)
    action = int(getattr(player.action, 'value', player.action))
    grounded = bool(player.on_ground) or action in LEDGE_STATES
    if not grounded:
        mask &= ~GROUNDED
    x, y = float(player.position.x), float(player.position.y)
    if abs(x) > EDGE_X + 2.0 or y < -8.0:
        mask &= ~ONSTAGE
    # A tech is only offered when a knockdown is coming (being thrown, or knocked flying)
    # so ordinary play never sees it, and only where a mistimed click is harmless: while
    # held (cannot air-dodge) or low over the stage well inside the ledges, where an
    # air-dodge is a waveland. Offering it everywhere near the ground made an untrained
    # converted policy tap shield at random (live A/B, 2026-09-25).
    thrown = action in THROWN_STATES
    low = (action in KNOCKED_FLYING and 0.0 <= y < TECH_HEIGHT
           and not bool(getattr(player, 'off_stage', False)))
    if not (thrown or (low and abs(x) < EDGE_X - TECH_MARGIN)):
        mask &= ~TECH
    if not (thrown or (low and abs(x) < EDGE_X - TECH_ROLL_MARGIN)):
        mask &= ~TECH_ROLL
    return mask


# Inputs the game acts on when they are *pressed*, not while they are held: a second
# jump needs Y released and pressed again, a second aerial needs the C-stick back at
# neutral. The shield trigger is deliberately absent: shield -> roll relies on holding it.
EDGE_BUTTONS = frozenset({'BUTTON_A', 'BUTTON_B', 'BUTTON_X', 'BUTTON_Y', 'BUTTON_Z'})
EXECUTOR_VERSION = 'edge-release-lcancel-autotech-default-v5'
TECH_DIRECTION = {'tech': NEUTRAL, 'tech_left': LEFT, 'tech_right': RIGHT}


def _first_frame(m: Macro) -> Pad:
    return m.script[0] if m.script else m.hold


class Executor:
    """Plays one chosen macro out frame by frame for one player.

    Macros are scripted in isolation, but they run back to back. When a macro would
    press a button that the previous frame was already holding (a full jump ends with Y
    down, and the next jump starts with Y down), the game sees one long hold, not a
    second press, and the second jump silently never happens. So the executor remembers
    the last frame it sent and, when needed, spends one frame releasing those inputs
    before the macro's script begins, exactly as a thumb has to."""

    def __init__(self, act_every: int, auto_lcancel: bool = True, auto_tech: bool = True):
        self.act_every = max(1, int(act_every))
        self.auto_lcancel = auto_lcancel
        self.auto_tech = auto_tech
        self.lc_phase = False
        self.lcancel_pulses = 0
        self.armed = None            # (stick direction, frames left) of an armed tech
        self.tech_clicks = 0
        self.tech_lock = 0           # frames until another click could tech
        self.macro = MACROS[0]
        self.last = Pad()
        self.t = 0            # frames this decision has covered
        self.i = 0            # position in the macro's script
        self.lead = None      # release frame to send before the script, if any
        self.phase = 'script'
        self.air_t = 0
        self.air_wait = 0

    def released(self):
        """The pad was released outside the executor (menus, between games)."""
        self.last = Pad()

    def start(self, index: int):
        self.macro = MACROS[int(index)]
        if self.macro.name in TECH_DIRECTION:
            self.armed = (TECH_DIRECTION[self.macro.name], TECH_ARM_FRAMES)
        self.t = 0
        self.i = 0
        self.phase = 'script'
        self.air_t = 0
        self.air_wait = 0
        first = _first_frame(self.macro)
        held = EDGE_BUTTONS.intersection(self.last.buttons).intersection(first.buttons)
        cstick_held = self.last.cstick != NEUTRAL and first.cstick != NEUTRAL
        self.lead = None
        if held or cstick_held:
            self.lead = Pad(first.stick, NEUTRAL if cstick_held else first.cstick,
                            tuple(b for b in first.buttons if b not in held), first.trigger)

    @property
    def done(self) -> bool:
        """Ready for a new decision (the window has elapsed and the script finished)."""
        if self.t >= MAX_FRAMES:
            return True
        return self.t >= self.act_every and self.phase == 'hold' and self.lead is None

    def next_input(self, player) -> Pad:
        """The input for the coming frame. `player` is this player's current state."""
        out = self._tech(self._lcancel(self._next(player), player), player)
        self.last = out
        self.t += 1
        return out

    def _tech(self, out: Pad, player) -> Pad:
        """Fire an armed tech on the frame the game accepts it, and only where even a
        mistimed click (an air-dodge) would land on the stage."""
        self.tech_lock = max(0, self.tech_lock - 1)
        if self.tech_lock:
            return out
        action = int(getattr(player.action, 'value', -1))
        if self.armed is not None:
            direction, left = self.armed
            self.armed = (direction, left - 1) if left > 1 else None
        elif self.auto_tech and (action in DOWN_THROWN or action in KNOCKED_FLYING):
            # Teching in place is the default, like a human's muscle memory: arming by
            # choice was almost never learned (5% of knockdowns teched after the timing
            # fix, 2026-09-26). The tech macros now only choose a roll direction.
            direction = NEUTRAL
        else:
            return out
        if action in THROWN_STATES:
            # Other throws launch you: the chance comes on landing, and a click now would
            # lock it out. (Native probe: always-armed clicking in every throw teched 8 of
            # 36 knockdowns from 193 clicks.)
            fire = action in DOWN_THROWN and int(player.action_frame) >= TECH_THROW_FRAME
        elif action in KNOCKED_FLYING and not bool(player.on_ground):
            hit = landing(player, TECH_LAND_FRAMES)
            margin = TECH_ROLL_MARGIN if direction != NEUTRAL else TECH_MARGIN
            fire = (hit is not None and abs(float(player.position.x)) < EDGE_X - TECH_MARGIN
                    and abs(hit[1]) < EDGE_X - margin)
        else:
            fire = False
        if not fire:
            return out
        self.armed = None
        self.tech_clicks += 1
        self.tech_lock = TECH_LOCKOUT
        return Pad(direction, NEUTRAL, ('BUTTON_L',), 1.0)

    def _lcancel(self, out: Pad, player) -> Pad:
        """During a falling aerial, add a partial trigger press on alternate frames: each
        rising edge restarts Melee's 7-frame L-cancel window, and a partial press cannot
        become an air-dodge even when the aerial ends mid-air."""
        if not self.auto_lcancel or out.trigger > 0 or 'BUTTON_L' in out.buttons:
            return out
        falling_aerial = (int(getattr(player.action, 'value', -1)) in AERIAL_ATTACKS
                          and not bool(player.on_ground) and float(player.speed_y_self) < 0)
        if not falling_aerial:
            self.lc_phase = False
            return out
        self.lc_phase = not self.lc_phase
        if not self.lc_phase:
            return out
        self.lcancel_pulses += 1
        return Pad(out.stick, out.cstick, out.buttons, LCANCEL_ANALOG)

    def _next(self, player) -> Pad:
        m = self.macro
        if self.lead is not None:
            out, self.lead = self.lead, None
            return out
        out = m.hold
        if self.phase == 'script':
            if self.i < len(m.script):
                out = m.script[self.i]
            self.i += 1
            if self.i >= len(m.script):
                self.phase = 'airborne' if m.airborne else 'hold'
        elif self.phase == 'airborne':
            if self.air_t == 0 and bool(player.on_ground) and self.air_wait < 10:
                self.air_wait += 1
                out = Pad()
            else:
                out = m.airborne[self.air_t]
                self.air_t += 1
                if self.air_t >= len(m.airborne):
                    self.phase = 'hold'
        return out


def write(pad, frame: Pad):
    """Assert every axis of the controller for one frame (flushed by console.step)."""
    import melee
    pad.release_all()
    if frame.stick != NEUTRAL:
        pad.tilt_analog(melee.Button.BUTTON_MAIN, *frame.stick)
    if frame.cstick != NEUTRAL:
        pad.tilt_analog(melee.Button.BUTTON_C, *frame.cstick)
    for b in frame.buttons:
        pad.press_button(melee.Button[b])
    if frame.trigger > 0:
        pad.press_shoulder(melee.Button.BUTTON_L, frame.trigger)
