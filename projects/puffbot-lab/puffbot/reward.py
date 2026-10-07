"""The objective, and the match bookkeeping that explains a result.

Version one's first reward paid unbounded per-frame shaping over a 10,000-frame match:
a LOSS averaged +226.71, a WIN +143.29 and a TIMEOUT +699.60, so PPO learned to hold
shield 63% of the time and stall. After the rewrite, the same agent went from 1W-9L to
7W-3L at CPU 5. The rules below keep that property by construction:

* Everything is bounded per game. Stocks are worth +/-1 each, damage 0.01 per percent,
  the result +/-1. There is nothing a policy can loop to farm.
* The only shaping is a potential on distance offstage, discounted per frame like the
  learner. Within a stock it telescopes, so no loop of states can farm it (tested).
* It is deliberately NOT paid back when a stock is lost, so dying far offstage never
  looks like "returning to the stage". That makes it an extra recovery penalty of up to
  about one stock for dying deep offstage, which means the shaping is not
  policy-invariant: it can discourage deep, risky edgeguards. (An earlier version of
  this docstring claimed invariance; the GPT audit of 2026-09-25 showed it does not
  hold. Whether to keep the penalty is a training experiment, not a bug fix.)
"""
from __future__ import annotations

import math

from . import actions
from .observation import DEAD_STATES, REST_STATES

TECH_STATES = {0xC7, 0xC8, 0xC9}                  # NEUTRAL / FORWARD / BACKWARD_TECH
TECH_MISS_STATES = {0xB7, 0xBF}                    # knocked down: TECH_MISS_UP / DOWN
AIRDODGE = 0xEC
AERIAL_LANDINGS = {0x46, 0x47, 0x48, 0x49, 0x4A}   # NAIR..DAIR_LANDING

STOCK = 1.0
DAMAGE = 0.01
RESULT = 1.0
OFFSTAGE = 0.004              # per unit of distance beyond the stage
SELF_DESTRUCT_WINDOW = 45     # frames; v1's corrected classifier


def offstage_distance(p) -> float:
    x, y = abs(float(p.position.x)), float(p.position.y)
    dx = max(0.0, x - actions.EDGE_X)
    dy = min(0.0, y)
    return math.hypot(dx, dy)


def potential(p) -> float:
    return -OFFSTAGE * offstage_distance(p)


def frame_reward(prev, cur, me: int, gamma: float) -> float:
    """Reward for one frame transition from `me`'s point of view (no result bonus)."""
    you = 3 - me
    a, b = prev.players[me], prev.players[you]
    c, d = cur.players[me], cur.players[you]
    lost = max(0, int(a.stock) - int(c.stock))
    taken = max(0, int(b.stock) - int(d.stock))
    dealt = 0.0 if taken else max(0.0, float(d.percent) - float(b.percent))
    received = 0.0 if lost else max(0.0, float(c.percent) - float(a.percent))
    r = STOCK * (taken - lost) + DAMAGE * (dealt - received)
    if not lost and _alive(a) and _alive(c):
        r += gamma * potential(c) - potential(a)
    return r


def _alive(p) -> bool:
    return int(getattr(p.action, 'value', -1)) not in DEAD_STATES


def result_of(g, me: int) -> str:
    """Final result from the last in-game frame: stocks, then percent (the game's rule)."""
    a, b = g.players[me], g.players[3 - me]
    sa, sb = int(a.stock), int(b.stock)
    if sa != sb:
        return 'win' if sa > sb else 'loss'
    pa, pb = float(a.percent), float(b.percent)
    if pa != pb:
        return 'win' if pa < pb else 'loss'
    return 'draw'


def result_reward(result: str) -> float:
    return {'win': RESULT, 'loss': -RESULT}.get(result, 0.0)


class GameStats:
    """Per-game bookkeeping for one player: damage, stocks, why stocks were lost, Rest."""

    def __init__(self, me: int):
        self.me = me
        self.frames = 0
        self.dealt = self.received = 0.0
        self.taken = self.lost = 0
        self.self_destructs = 0
        self.rests = self.rest_hits = 0
        self.offstage_frames = 0
        self.techs = self.missed_techs = 0
        self.airdodges = self.offstage_airdodges = 0
        self.aerial_landings = self.aerial_landing_frames = 0
        self.landing_since = None
        self.last_hit = 0
        self.controlled_since_hit = True
        self.rest_started = None
        self.rest_opp_percent = 0.0
        self.actions = [0] * actions.COUNT

    def update(self, prev, cur):
        me, you = self.me, 3 - self.me
        a, b = prev.players[me], prev.players[you]
        c, d = cur.players[me], cur.players[you]
        frame = int(cur.frame)
        self.frames += 1
        lost = int(a.stock) - int(c.stock) > 0
        taken = int(b.stock) - int(d.stock) > 0
        if not taken:
            self.dealt += max(0.0, float(d.percent) - float(b.percent))
        if not lost:
            self.received += max(0.0, float(c.percent) - float(a.percent))
        self.taken += int(taken)
        self.offstage_frames += int(bool(c.off_stage))
        struck = float(c.percent) > float(a.percent) or int(c.hitstun_frames_left) > 0
        if struck:
            self.last_hit = frame
            self.controlled_since_hit = False
        elif _in_control(c):
            self.controlled_since_hit = True
        if lost:
            self.lost += 1
            recently_struck = frame - self.last_hit <= SELF_DESTRUCT_WINDOW
            if not recently_struck and self.controlled_since_hit:
                self.self_destructs += 1
            self.last_hit = frame
            self.controlled_since_hit = True
        was, now = int(getattr(a.action, 'value', -1)), int(getattr(c.action, 'value', -1))
        if now != was:
            if now in TECH_STATES:
                self.techs += 1
            elif now in TECH_MISS_STATES:
                self.missed_techs += 1
            elif now == AIRDODGE:
                self.airdodges += 1
                self.offstage_airdodges += int(bool(c.off_stage))
            if now in AERIAL_LANDINGS:
                self.aerial_landings += 1
                self.landing_since = frame
            elif was in AERIAL_LANDINGS and self.landing_since is not None:
                self.aerial_landing_frames += frame - self.landing_since
                self.landing_since = None
        # Rest: a new entry into a Rest state; it hit if the opponent took damage
        # within the first few frames (the hitbox is out on frame 1).
        now_rest = int(getattr(c.action, 'value', -1)) in REST_STATES
        was_rest = int(getattr(a.action, 'value', -1)) in REST_STATES
        if now_rest and not was_rest:
            self.rests += 1
            self.rest_started = frame
            self.rest_opp_percent = float(b.percent)
        if self.rest_started is not None and frame - self.rest_started <= 4:
            if float(d.percent) - self.rest_opp_percent >= 10 or taken:
                self.rest_hits += 1
                self.rest_started = None
        elif self.rest_started is not None:
            self.rest_started = None

    def summary(self) -> dict:
        return dict(frames=self.frames, damage_dealt=round(self.dealt, 1),
                    damage_received=round(self.received, 1), stocks_taken=self.taken,
                    stocks_lost=self.lost, self_destructs=self.self_destructs,
                    rests=self.rests, rest_hits=self.rest_hits,
                    offstage_frames=self.offstage_frames, techs=self.techs, missed_techs=self.missed_techs,
                    airdodges=self.airdodges, offstage_airdodges=self.offstage_airdodges,
                    aerial_landings=self.aerial_landings, aerial_landing_frames=self.aerial_landing_frames,
                    actions=list(self.actions))


def _in_control(p) -> bool:
    """On stage or on the ledge with no hitstun: anything after this is the player's own doing."""
    if int(p.hitstun_frames_left) > 0:
        return False
    if int(getattr(p.action, 'value', -1)) in actions.LEDGE_STATES:
        return True
    return bool(p.on_ground) and not bool(p.off_stage)
