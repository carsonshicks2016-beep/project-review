"""What the policy sees: one frame of game state from one player's point of view.

Continuous features go in a float vector; action state and character go in as integer
ids the network embeds. The game already reports velocities, hitstun, hitlag and the
frame of the current animation, so a single frame is close to a complete description
of the state and no frame stacking is needed. Version one's LSTM measured exactly zero
influence on its output, so memory is not where the capacity went.

Both perspectives use one function, so a policy playing port 2 in self-play sees the
game exactly as it does on port 1 (projectile ownership included).
"""
from __future__ import annotations

import math

import numpy as np

from . import actions

ACTION_VOCAB = 512
CHAR_VOCAB = 40
PER_PLAYER = 22
GLOBALS = 6
PROJECTILES = 2
PER_PROJECTILE = 5    # dx, dy, present, vx, vy (v1 had no velocity: a laser coming at
                      # you and one flying away looked identical; GPT audit 2026-09-25)
FLOATS = 2 * PER_PLAYER + GLOBALS + PER_PROJECTILE * PROJECTILES + actions.COUNT
IDS = 4  # self action, opponent action, self character, opponent character

# Bump VERSION whenever a feature changes meaning, even if the sizes stay the same.
VERSION = 'puff-obs-v2'
LAYOUT_V1 = {'version': 'puff-obs-v1', 'floats': 99, 'ids': 4, 'action_vocab': 512, 'char_vocab': 40}


def layout() -> dict:
    return {'version': VERSION, 'floats': FLOATS, 'ids': IDS,
            'action_vocab': ACTION_VOCAB, 'char_vocab': CHAR_VOCAB}


def feature_names(version: str = VERSION) -> list[str]:
    """Name of every float feature, in order. Migration maps weights by these names,
    so a feature keeps its trained weights wherever it moves."""
    names = [f'{who}{i}' for who in ('me_', 'opp_') for i in range(PER_PLAYER)]
    names += [f'global{i}' for i in range(GLOBALS)]
    fields = ('dx', 'dy', 'present') if version == 'puff-obs-v1' else ('dx', 'dy', 'present', 'vx', 'vy')
    names += [f'proj{k}_{f}' for k in range(PROJECTILES) for f in fields]
    vocab = actions.V1_NAMES if version == 'puff-obs-v1' else actions.NAMES
    names += [f'prev_{n}' for n in vocab]
    return names

DEAD_STATES = set(range(0x00, 0x0B)) | {0x0C, 0x0D}   # DEAD_* and ON_HALO_*
REST_STATES = {369, 370, 371, 372}                   # Jigglypuff Rest (ground/air)


def _id(value, vocab):
    v = int(getattr(value, 'value', value if isinstance(value, int) else vocab - 1))
    return v if 0 <= v < vocab else vocab - 1


def _player(p, out, o):
    x, y = float(p.position.x), float(p.position.y)
    action = _id(p.action, ACTION_VOCAB)
    out[o + 0] = x / 100.0
    out[o + 1] = y / 100.0
    out[o + 2] = float(p.percent) / 100.0
    out[o + 3] = float(p.stock) / 4.0
    out[o + 4] = 1.0 if p.facing else -1.0
    out[o + 5] = float(p.on_ground)
    out[o + 6] = float(p.off_stage)
    out[o + 7] = float(p.jumps_left) / 6.0
    out[o + 8] = float(p.shield_strength) / 60.0
    out[o + 9] = float(p.invulnerable)
    out[o + 10] = min(float(p.hitlag_left), 30.0) / 10.0
    out[o + 11] = min(float(p.hitstun_frames_left), 90.0) / 30.0
    out[o + 12] = min(float(p.action_frame), 90.0) / 30.0
    out[o + 13] = float(p.speed_air_x_self) / 3.0
    out[o + 14] = float(p.speed_y_self) / 3.0
    out[o + 15] = float(p.speed_x_attack) / 3.0
    out[o + 16] = float(p.speed_y_attack) / 3.0
    out[o + 17] = float(p.speed_ground_x_self) / 3.0
    out[o + 18] = (actions.EDGE_X - abs(x)) / 100.0
    out[o + 19] = float(action in actions.LEDGE_STATES)
    out[o + 20] = float(action in DEAD_STATES)
    out[o + 21] = float(action in REST_STATES)
    return action


def encode(g, me: int, prev_action: int, frame_limit: int):
    """(floats, ids) for the player on port `me` against the other port."""
    you = 3 - me
    a, b = g.players[me], g.players[you]
    floats = np.zeros(FLOATS, np.float32)
    ids = np.zeros(IDS, np.int64)
    ids[0] = _player(a, floats, 0)
    ids[1] = _player(b, floats, PER_PLAYER)
    ids[2] = _id(a.character, CHAR_VOCAB)
    ids[3] = _id(b.character, CHAR_VOCAB)
    o = 2 * PER_PLAYER
    dx = float(b.position.x) - float(a.position.x)
    dy = float(b.position.y) - float(a.position.y)
    floats[o + 0] = dx / 100.0
    floats[o + 1] = dy / 100.0
    floats[o + 2] = math.hypot(dx, dy) / 100.0
    floats[o + 3] = (1.0 if a.facing else -1.0) * (1.0 if dx >= 0 else -1.0)
    floats[o + 4] = (1.0 if b.facing else -1.0) * (-1.0 if dx >= 0 else 1.0)
    floats[o + 5] = min(max(float(g.frame), 0.0) / max(frame_limit, 1), 1.0)
    o += GLOBALS
    # Nearest projectiles the opponent owns (lasers, needles, turnips...).
    near = []
    ax, ay = float(a.position.x), float(a.position.y)
    for proj in getattr(g, 'projectiles', ()) or ():
        if int(getattr(proj, 'owner', -1)) == me:
            continue
        px, py = float(proj.position.x) - ax, float(proj.position.y) - ay
        speed = getattr(proj, 'speed', None)
        vx = float(getattr(speed, 'x', 0.0)) if speed is not None else 0.0
        vy = float(getattr(speed, 'y', 0.0)) if speed is not None else 0.0
        near.append((px * px + py * py, px, py, vx, vy))
    near.sort()
    for k, (_, px, py, vx, vy) in enumerate(near[:PROJECTILES]):
        base = o + PER_PROJECTILE * k
        floats[base:base + PER_PROJECTILE] = (px / 100.0, py / 100.0, 1.0, vx / 5.0, vy / 5.0)
    o += PER_PROJECTILE * PROJECTILES
    if 0 <= prev_action < actions.COUNT:
        floats[o + prev_action] = 1.0
    np.clip(floats, -10.0, 10.0, out=floats)
    return floats, ids
