"""Versioned state encoding and rewards shared by training and evaluation."""
import numpy as np
from collections import deque

# Absolute coordinates: both controller directions and observations use world axes.
FEATURES = 24
ACTION_STATES = 400
CHARACTERS = 33
PLAYER_SIZE = FEATURES + ACTION_STATES + CHARACTERS
FRAME_SIZE = 2 * PLAYER_SIZE + 4
OBS_SIZE = FRAME_SIZE * 2
SCHEMA = 'melee-lab-v1'

def value(x):
    return int(getattr(x, 'value', x))

def player_features(p, g=None, port=1):
    out = np.zeros(PLAYER_SIZE, np.float32)
    pos = getattr(p, 'position', None)
    px = float(getattr(pos, 'x', 0.0)) if pos else 0.0
    py = float(getattr(pos, 'y', 0.0)) if pos else 0.0
    is_offstage = float(getattr(p, 'off_stage', False))
    # Distance to nearest stage ledge (Final Destination stage edge is 85.56)
    dx_ledge = (abs(px) - 85.56) / 100.0 if is_offstage else 0.0
    dy_ledge = py / 100.0 if is_offstage else 0.0
    act_val = value(p.action)
    is_ledge_hanging = float(act_val in (252, 253))

    # Nearest hostile projectile tracking (e.g. Falco laser, Mario fireball)
    proj_dx, proj_dy = 0.0, 0.0
    if g and getattr(g, 'projectiles', None):
        best_d = float('inf')
        for proj in g.projectiles:
            if getattr(proj, 'owner', -1) != port:
                ppos = getattr(proj, 'position', None)
                if ppos:
                    pdx = ppos.x - px
                    pdy = ppos.y - py
                    d = (pdx * pdx + pdy * pdy) ** 0.5
                    if d < best_d and d < 250.0:
                        best_d = d
                        proj_dx = pdx / 200.0
                        proj_dy = pdy / 150.0

    out[:FEATURES] = [
        px/200, py/150, p.percent/200, p.stock/3,
        float(p.facing), float(p.on_ground), p.jumps_left/2,
        p.shield_strength/60, p.action_frame/60, p.hitstun_frames_left/60,
        p.hitlag_left/15, float(p.invulnerable),
        p.speed_air_x_self/5, p.speed_ground_x_self/5, p.speed_y_self/5,
        p.speed_x_attack/10, p.speed_y_attack/10,
        is_offstage, dx_ledge,
        dy_ledge, is_ledge_hanging,
        proj_dx, proj_dy, float(p.moonwalkwarning),
    ]
    out[FEATURES + min(max(act_val, 0), ACTION_STATES-1)] = 1
    out[FEATURES + ACTION_STATES + min(max(value(p.character),0),CHARACTERS-1)] = 1
    return np.clip(np.nan_to_num(out), -5, 5)

def encode(g, agent_port=1, opponent_port=2):
    a,b = g.players[agent_port], g.players[opponent_port]
    return np.concatenate((player_features(a, g, agent_port), player_features(b, g, opponent_port), np.array([
        (b.position.x-a.position.x)/200, (b.position.y-a.position.y)/150,
        min(max(g.frame,0)/28800,1), float(a.facing == (b.position.x > a.position.x))
    ], np.float32)))

class History:
    def __init__(self, agent_port=1, opponent_port=2):
        self.frames = deque(maxlen=2)
        self.agent_port = agent_port
        self.opponent_port = opponent_port
    def reset(self,g):
        v=encode(g, self.agent_port, self.opponent_port)
        self.frames.clear(); self.frames.extend([v,v]); return self.get()
    def push(self,g):
        v=encode(g, self.agent_port, self.opponent_port)
        if not self.frames: self.frames.extend([v,v])
        else: self.frames.append(v)
        return self.get()
    def get(self): return np.concatenate(self.frames).astype(np.float32)

def snapshot(g):
    # libmelee yields an UnknownAnimation (no .name) for ids outside its enum, the
    # same case value() already guards in player_features.
    return {str(port): {'x':float(p.position.x), 'y':float(p.position.y),
            'percent':float(p.percent), 'stocks':int(p.stock),
            'action':getattr(p.action,'name',None) or f'UNKNOWN_{value(p.action)}',
            'character':getattr(p.character,'name',None) or f'UNKNOWN_{value(p.character)}'}
            for port,p in g.players.items() if port in (1,2)}

# Ports are arguments so a parsed tournament replay -- where the pro Fox may sit on
# any port, and a mirror is scored twice -- is rewarded by exactly the same function
# PPO optimises. A value head fitted to a different reward is worse than none.
def outcome(g,agent_port=1,opponent_port=2):
    a,b = g.players[agent_port],g.players[opponent_port]
    if a.stock == 0 and b.stock == 0: return 'draw'
    if b.stock == 0: return 'win'
    if a.stock == 0: return 'loss'
    return None

def _stage_distance(p):
    pos = getattr(p, 'position', None)
    if not pos: return 0.0
    x, y = float(getattr(pos, 'x', 0.0)), float(getattr(pos, 'y', 0.0))
    dx = max(0.0, abs(x) - 85.56)
    return (dx * dx + y * y) ** 0.5

def _is_hit_or_stunned(p):
    if not p: return False
    if getattr(p, 'hitstun_frames_left', 0) > 0 or getattr(p, 'hitlag_left', 0) > 0: return True
    act = getattr(p.action, 'name', '') if hasattr(p, 'action') else str(getattr(p, 'action', ''))
    return any(k in act for k in ('DAMAGE', 'TUMBL', 'HIT', 'THROWN', 'FLY'))

LEDGE_ACTIONS = ('EDGE_CATCHING', 'EDGE_HANGING')


def in_control(p):
    """Has the agent got the stock back under its own command?

    Standing on the stage, or hanging from the ledge, with no hitstun left. A launch
    that is survived ends here; anything that happens after this point is the agent's
    own doing again. Used to decide who a lost stock belongs to, so it has to mean the
    same thing in the live environment and in offline replay scoring."""
    if int(getattr(p, 'hitstun_frames_left', 0)) > 0:
        return False
    if getattr(p.action, 'name', '') in LEDGE_ACTIONS:
        return True
    return bool(getattr(p, 'on_ground', False)) and not bool(getattr(p, 'off_stage', False))


# Reward scale. Every term below is BOUNDED per episode, which is the property the
# previous version lacked: its bonuses fired per frame or per transition and accumulated
# without limit across a 10,000-frame match. Measured on a 134-match run, that produced
#
#     mean episode return for a LOSS   +226.71
#     mean episode return for a WIN    +143.29
#     mean episode return for a TIMEOUT +699.60
#
# PPO optimised exactly that: over three hours it froze every controller axis except the
# shield, sat at 63% hard shield on stage, stretched median match length from 9,505 to
# 18,573 frames, and was demoted from CPU 5 to CPU 1, where the policy it started from
# had won 10-0. It was not broken. It was winning the game it had been given.
#
# So the objective is now the objective: three stocks and the match. Damage is a dense
# hint an order of magnitude below a stock. The only remaining shaping is potential-based
# and therefore provably cannot be farmed.
STOCK = 4.0
DAMAGE = 0.015           # ~300% over a match = 4.5, comparable to one stock
OUTCOME = 10.0
RECOVERY_POTENTIAL = 0.02


def _potential(p):
    """Potential over distance from the stage: further out is worse.

    Potential-based shaping (Ng, Harada & Russell 1999): a per-step reward of the form
    PHI(s') - PHI(s) telescopes across an episode to PHI(final) - PHI(start), so no loop
    of states can pay more than once and the optimal policy is provably unchanged. That
    is the guarantee the old departure penalty and return bonus did not have -- they paid
    -1.5 and +1.5 on every crossing, roughly 15 crossings a match, which put +/-22 of
    farmable reward against a 12-point stock signal."""
    return -RECOVERY_POTENTIAL * _stage_distance(p)


def reward(previous, current, agent_port=1, opponent_port=2):
    """Reward for one transition. Bounded per episode, and a win always beats a loss.

    Ports are arguments so a parsed tournament replay -- where the pro may sit on any
    port, and a mirror is scored twice -- is rewarded by exactly the same function PPO
    optimises. A value head fitted to a different reward is worse than none."""
    a, b = previous.players[agent_port], previous.players[opponent_port]
    c, d = current.players[agent_port], current.players[opponent_port]

    lost = max(0, int(a.stock) - int(c.stock))
    taken = max(0, int(b.stock) - int(d.stock))
    # No damage credit across a respawn: percent resets to zero, which would otherwise
    # read as a huge hit landed or taken.
    damage_dealt = max(0., float(d.percent) - float(b.percent)) if not taken else 0.
    damage_taken = max(0., float(c.percent) - float(a.percent)) if not lost else 0.

    r = STOCK * (taken - lost) + DAMAGE * (damage_dealt - damage_taken)

    # Telescoping recovery shaping. Skipped across a death, where the position jump to
    # the respawn platform is not something the agent did.
    if not lost:
        r += _potential(c) - _potential(a)

    result = outcome(current, agent_port, opponent_port)
    if result == 'win':
        r += OUTCOME
    elif result == 'loss':
        r -= OUTCOME
    return float(r)


def episode_bound():
    """The most any single episode can pay, ignoring outcome.

    Used by the tests that keep a win worth more than a loss. Three stocks, a generous
    500% of damage either way, and both ends of the recovery potential."""
    return STOCK * 3 + DAMAGE * 500 + 2 * RECOVERY_POTENTIAL * 300


# --- Compact storage for demonstration datasets -----------------------------
#
# An observation is 1,836 floats of which 1,732 are one-hot zeros, so storing
# demonstrations densely costs 7,344 bytes a frame and puts a ceiling of about a
# million samples on a 17 GB machine. Packing keeps the 24 continuous features per
# player plus the two one-hot INDICES, at 428 bytes a frame, and the dense vector is
# rebuilt a minibatch at a time.
#
# Both directions work on the output of encode()/History.get() rather than on a
# gamestate, so the packed form cannot drift out of step with the encoding: whatever
# encode() produces is what expand() reproduces, and the round trip is exact.

IDS_PER_SAMPLE = 8      # 2 frames x 2 players, action then character
CONTINUOUS_PER_SAMPLE = 2 * 2 * FEATURES
GLOBALS_PER_SAMPLE = 2 * 4


def _slots():
    """Where each player's block starts, for both stacked frames."""
    return [(f * FRAME_SIZE + p * PLAYER_SIZE) for f in (0, 1) for p in (0, 1)]


def compress(dense):
    """Dense observations -> (continuous, globals, ids). Exactly invertible."""
    dense = np.asarray(dense, dtype=np.float32).reshape(-1, OBS_SIZE)
    n = len(dense)
    continuous = np.empty((n, CONTINUOUS_PER_SAMPLE), np.float32)
    ids = np.empty((n, IDS_PER_SAMPLE), np.int16)
    for i, start in enumerate(_slots()):
        continuous[:, i*FEATURES:(i+1)*FEATURES] = dense[:, start:start+FEATURES]
        block = dense[:, start+FEATURES:start+PLAYER_SIZE]
        ids[:, i] = block[:, :ACTION_STATES].argmax(axis=1)
        ids[:, 4+i] = block[:, ACTION_STATES:].argmax(axis=1)
    globals_ = np.concatenate([dense[:, f*FRAME_SIZE+2*PLAYER_SIZE:(f+1)*FRAME_SIZE] for f in (0, 1)], axis=1)
    return continuous, globals_.astype(np.float32), ids


def expand(continuous, globals_, ids, out=None):
    """(continuous, globals, ids) -> dense observations. `out` is reused across
    minibatches so training does not allocate a fresh array per step."""
    n = len(ids)
    if out is None or len(out) != n:
        out = np.zeros((n, OBS_SIZE), np.float32)
    else:
        out.fill(0)
    rows = np.arange(n)
    for i, start in enumerate(_slots()):
        out[:, start:start+FEATURES] = continuous[:, i*FEATURES:(i+1)*FEATURES]
        out[rows, start + FEATURES + ids[:, i]] = 1
        out[rows, start + FEATURES + ACTION_STATES + ids[:, 4+i]] = 1
    for f in (0, 1):
        out[:, f*FRAME_SIZE+2*PLAYER_SIZE:(f+1)*FRAME_SIZE] = globals_[:, f*4:(f+1)*4]
    return out
