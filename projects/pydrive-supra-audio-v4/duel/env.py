"""
Duel — a vectorized 2-player top-down arena for self-play PPO.

Two automata duel in a bounded void.  Each frame an agent thrusts (continuous
2D acceleration), turns its aim toward a target direction at a limited rate,
may fire a traveling projectile (on a cooldown), and may dash (an i-frame
burst on a longer cooldown).  Last one alive wins.

The whole simulation is batched over N independent arenas with numpy so PPO can
collect tens of thousands of transitions per second on CPU.  All game constants
live here and are surfaced through `metadata()` so the renderer stays in sync.
"""
from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# Simulation constants (single source of truth; mirrored into replays)
# ----------------------------------------------------------------------------
DT = 1.0 / 30.0
MAX_STEPS = 450                # 15 s hard cap per duel (snappier, harder to stall)

ARENA_W = 24.0                 # tighter arena -> avoidance is costly, fights stay close
ARENA_H = 13.5
AGENT_R = 0.70

ACCEL = 45.0                   # thrust accel (world units / s^2)
DRAG = 4.0                     # linear drag coefficient
MAX_SPEED = 10.0
TURN_RATE = 7.0                # max aim turn (rad / s)

FIRE_COOLDOWN = 0.32
BULLET_SPEED = 19.0
BULLET_LIFETIME = 1.40
BULLET_R = 0.22
DAMAGE = 34.0                  # ~3 clean hits to kill
HEALTH = 100.0

DASH_COOLDOWN = 1.10
DASH_SPEED = 22.0              # burst speed
DASH_DURATION = 0.16          # window where the speed cap is raised
DASH_IFRAMES = 0.12           # invulnerability granted by a dash (a dodge, not a get-out-of-jail)

MAX_BULLETS = 12              # live projectiles per arena (both players)
KNN_BULLETS = 3               # nearest incoming bullets exposed to the policy

# Rewards (per agent, per step) ----------------------------------------------
# Landing a hit is worth MORE than taking one costs, so trading blows is
# positive-EV and the policy is pushed toward aggression instead of mutual
# avoidance.  A heavy draw penalty makes running out the clock nearly as bad as
# losing, which kills the stalemate equilibrium endemic to duelling self-play.
R_WIN = 1.0
R_HIT = 0.70                   # landing a hit
R_GOT_HIT = 0.30              # taking a hit
R_TIME = -0.0015              # mild impatience
R_DRAW = -0.70               # timeout with no kill

MIN_SPAWN_SEP = 8.0


def _wrap(a):
    """Wrap angles to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class VecDuel:
    """A batch of N independent duels stepped in lockstep.

    Action layout (per env, per player), shape (N, 2, 6):
        [accel_x, accel_y, aim_x, aim_y, fire, dash]
    The first four are continuous (clipped/normalised in-env); fire and dash are
    treated as binary (>= 0.5).
    Observation: (N, 2, obs_dim), egocentric per player.
    """

    def __init__(self, n_envs: int, seed: int = 0):
        self.n = n_envs
        self.rng = np.random.default_rng(seed)
        self.diag = float(np.hypot(ARENA_W, ARENA_H))

        # Per-agent state, shape (N, 2, ...)
        self.pos = np.zeros((self.n, 2, 2), np.float32)
        self.vel = np.zeros((self.n, 2, 2), np.float32)
        self.face = np.zeros((self.n, 2), np.float32)
        self.health = np.full((self.n, 2), HEALTH, np.float32)
        self.fire_cd = np.zeros((self.n, 2), np.float32)
        self.dash_cd = np.zeros((self.n, 2), np.float32)
        self.dash_t = np.zeros((self.n, 2), np.float32)
        self.invuln = np.zeros((self.n, 2), np.float32)
        self.fired_flash = np.zeros((self.n, 2), np.float32)  # render-only telemetry

        # Projectiles, shape (N, B, ...)
        self.b_pos = np.zeros((self.n, MAX_BULLETS, 2), np.float32)
        self.b_vel = np.zeros((self.n, MAX_BULLETS, 2), np.float32)
        self.b_life = np.zeros((self.n, MAX_BULLETS), np.float32)
        self.b_owner = np.zeros((self.n, MAX_BULLETS), np.int64)

        self.t = np.zeros(self.n, np.int64)

        self.obs_dim = self._build_obs(0).shape[1]
        self.act_dim = 6
        self._reset_all()

    # ------------------------------------------------------------------ resets
    def _reset_all(self):
        self._reset_idx(np.arange(self.n))

    def _reset_idx(self, idx):
        m = len(idx)
        if m == 0:
            return
        # Rejection-sample two spawn points with a minimum separation.
        p = np.zeros((m, 2, 2), np.float32)
        pad = 2.5
        for k in range(2):
            p[:, k, 0] = self.rng.uniform(pad, ARENA_W - pad, m)
            p[:, k, 1] = self.rng.uniform(pad, ARENA_H - pad, m)
        for _ in range(8):
            d = np.linalg.norm(p[:, 0] - p[:, 1], axis=1)
            bad = d < MIN_SPAWN_SEP
            if not bad.any():
                break
            nb = int(bad.sum())
            p[bad, 1, 0] = self.rng.uniform(pad, ARENA_W - pad, nb)
            p[bad, 1, 1] = self.rng.uniform(pad, ARENA_H - pad, nb)
        self.pos[idx] = p
        self.vel[idx] = 0.0
        # Face roughly toward the opponent, plus noise.
        to1 = p[:, 1] - p[:, 0]
        to0 = -to1
        self.face[idx, 0] = np.arctan2(to1[:, 1], to1[:, 0]) + self.rng.uniform(-0.6, 0.6, m)
        self.face[idx, 1] = np.arctan2(to0[:, 1], to0[:, 0]) + self.rng.uniform(-0.6, 0.6, m)
        self.health[idx] = HEALTH
        self.fire_cd[idx] = 0.0
        self.dash_cd[idx] = self.rng.uniform(0.0, DASH_COOLDOWN, (m, 2)).astype(np.float32)
        self.dash_t[idx] = 0.0
        self.invuln[idx] = 0.0
        self.fired_flash[idx] = 0.0
        self.b_life[idx] = 0.0
        self.t[idx] = 0

    def reset(self):
        self._reset_all()
        return self._obs()

    # -------------------------------------------------------------------- step
    def step(self, action, auto_reset=True):
        """action: (N, 2, 6) float array.  Returns obs, reward(N,2), done(N,), info.

        ``auto_reset`` resets finished arenas in place (the training convention).
        Pass False when recording so the terminal frame is preserved.
        """
        a = np.asarray(action, np.float32).reshape(self.n, 2, 6)
        rew = np.zeros((self.n, 2), np.float32)
        self.fired_flash[:] = 0.0

        # --- timers ---
        self.fire_cd = np.maximum(self.fire_cd - DT, 0.0)
        self.dash_cd = np.maximum(self.dash_cd - DT, 0.0)
        self.dash_t = np.maximum(self.dash_t - DT, 0.0)
        self.invuln = np.maximum(self.invuln - DT, 0.0)

        # --- thrust ---
        acc = a[:, :, 0:2].copy()
        mag = np.linalg.norm(acc, axis=2, keepdims=True)
        acc = np.where(mag > 1.0, acc / np.maximum(mag, 1e-6), acc)  # clamp to unit disc
        self.vel += acc * ACCEL * DT
        self.vel *= max(0.0, 1.0 - DRAG * DT)

        # --- dash trigger ---
        want_dash = (a[:, :, 5] >= 0.5) & (self.dash_cd <= 0.0)
        if want_dash.any():
            # Dash in the intended-move direction, falling back to facing.
            dirv = acc.copy()
            dm = np.linalg.norm(dirv, axis=2, keepdims=True)
            fb = np.stack([np.cos(self.face), np.sin(self.face)], axis=2)
            dirv = np.where(dm > 0.05, dirv / np.maximum(dm, 1e-6), fb)
            wd = want_dash[:, :, None]
            self.vel = np.where(wd, dirv * DASH_SPEED, self.vel)
            self.dash_t = np.where(want_dash, DASH_DURATION, self.dash_t)
            self.invuln = np.where(want_dash, np.maximum(self.invuln, DASH_IFRAMES), self.invuln)
            self.dash_cd = np.where(want_dash, DASH_COOLDOWN, self.dash_cd)

        # --- speed cap (raised briefly during a dash) ---
        cap = np.where(self.dash_t > 0.0, DASH_SPEED, MAX_SPEED)[:, :, None]
        sp = np.linalg.norm(self.vel, axis=2, keepdims=True)
        self.vel = np.where(sp > cap, self.vel / np.maximum(sp, 1e-6) * cap, self.vel)

        # --- integrate + walls ---
        self.pos += self.vel * DT
        lo, hix, hiy = AGENT_R, ARENA_W - AGENT_R, ARENA_H - AGENT_R
        # x
        m = self.pos[:, :, 0] < lo
        self.pos[:, :, 0] = np.where(m, lo, self.pos[:, :, 0])
        self.vel[:, :, 0] = np.where(m, np.maximum(self.vel[:, :, 0], 0.0), self.vel[:, :, 0])
        m = self.pos[:, :, 0] > hix
        self.pos[:, :, 0] = np.where(m, hix, self.pos[:, :, 0])
        self.vel[:, :, 0] = np.where(m, np.minimum(self.vel[:, :, 0], 0.0), self.vel[:, :, 0])
        # y
        m = self.pos[:, :, 1] < lo
        self.pos[:, :, 1] = np.where(m, lo, self.pos[:, :, 1])
        self.vel[:, :, 1] = np.where(m, np.maximum(self.vel[:, :, 1], 0.0), self.vel[:, :, 1])
        m = self.pos[:, :, 1] > hiy
        self.pos[:, :, 1] = np.where(m, hiy, self.pos[:, :, 1])
        self.vel[:, :, 1] = np.where(m, np.minimum(self.vel[:, :, 1], 0.0), self.vel[:, :, 1])

        # --- aim (limited turn rate toward target direction) ---
        aim = a[:, :, 2:4]
        amag = np.linalg.norm(aim, axis=2)
        tgt = np.where(amag > 1e-3, np.arctan2(aim[:, :, 1], aim[:, :, 0]), self.face)
        diff = _wrap(tgt - self.face)
        step = np.clip(diff, -TURN_RATE * DT, TURN_RATE * DT)
        self.face = _wrap(self.face + step)

        # --- fire ---
        fdir = np.stack([np.cos(self.face), np.sin(self.face)], axis=2)  # (N,2,2)
        for p in range(2):
            want = (a[:, p, 4] >= 0.5) & (self.fire_cd[:, p] <= 0.0)
            dead = self.b_life <= 0.0
            has = dead.any(axis=1)
            spawn = want & has
            if not spawn.any():
                continue
            ei = np.where(spawn)[0]
            slot = np.argmax(dead[ei], axis=1)
            muzzle = self.pos[ei, p] + fdir[ei, p] * (AGENT_R + 0.25)
            self.b_pos[ei, slot] = muzzle
            self.b_vel[ei, slot] = fdir[ei, p] * BULLET_SPEED
            self.b_life[ei, slot] = BULLET_LIFETIME
            self.b_owner[ei, slot] = p
            self.fire_cd[ei, p] = FIRE_COOLDOWN
            self.fired_flash[ei, p] = 1.0

        # --- bullets advance ---
        self.b_pos += self.b_vel * DT
        self.b_life -= DT
        oob = ((self.b_pos[:, :, 0] < 0) | (self.b_pos[:, :, 0] > ARENA_W) |
               (self.b_pos[:, :, 1] < 0) | (self.b_pos[:, :, 1] > ARENA_H))
        self.b_life = np.where(oob, 0.0, self.b_life)

        # --- collisions ---
        alive = self.b_life > 0.0
        opp = 1 - self.b_owner                                    # (N,B)
        rows = np.arange(self.n)[:, None]
        opp_pos = self.pos[rows, opp]                             # (N,B,2)
        opp_inv = self.invuln[rows, opp]                          # (N,B)
        d = np.linalg.norm(self.b_pos - opp_pos, axis=2)
        hit = alive & (d < (BULLET_R + AGENT_R)) & (opp_inv <= 0.0)
        if hit.any():
            hit_on0 = (hit & (self.b_owner == 1)).sum(axis=1)     # bullets from p1 hitting p0
            hit_on1 = (hit & (self.b_owner == 0)).sum(axis=1)
            self.health[:, 0] -= hit_on0 * DAMAGE
            self.health[:, 1] -= hit_on1 * DAMAGE
            self.b_life = np.where(hit, 0.0, self.b_life)
            # reward: + for hits you land, - for hits you take
            rew[:, 0] += R_HIT * hit_on1 - R_GOT_HIT * hit_on0
            rew[:, 1] += R_HIT * hit_on0 - R_GOT_HIT * hit_on1
        self.health = np.maximum(self.health, 0.0)

        # --- time + termination ---
        self.t += 1
        rew += R_TIME
        dead0 = self.health[:, 0] <= 0.0
        dead1 = self.health[:, 1] <= 0.0
        timeout = self.t >= MAX_STEPS
        done = dead0 | dead1 | timeout

        # win/loss/draw terminal bonus
        p0_win = dead1 & ~dead0
        p1_win = dead0 & ~dead1
        rew[:, 0] += R_WIN * p0_win.astype(np.float32) - R_WIN * p1_win.astype(np.float32)
        rew[:, 1] += R_WIN * p1_win.astype(np.float32) - R_WIN * p0_win.astype(np.float32)
        stalemate = timeout & ~dead0 & ~dead1
        rew[:, 0] += R_DRAW * stalemate.astype(np.float32)
        rew[:, 1] += R_DRAW * stalemate.astype(np.float32)

        winner = np.full(self.n, -1, np.int64)
        winner[p0_win] = 0
        winner[p1_win] = 1
        info = {"winner": winner, "done": done.copy()}

        # auto-reset finished arenas (training convention)
        if auto_reset:
            di = np.where(done)[0]
            if len(di):
                self._reset_idx(di)

        return self._obs(), rew, done, info

    # ------------------------------------------------------------- observations
    def _build_obs(self, p):
        o = 1 - p
        rows = np.arange(self.n)
        spos, opos = self.pos[:, p], self.pos[:, o]
        svel, ovel = self.vel[:, p], self.vel[:, o]
        sface, oface = self.face[:, p], self.face[:, o]
        sfd = np.stack([np.cos(sface), np.sin(sface)], 1)
        ofd = np.stack([np.cos(oface), np.sin(oface)], 1)

        rel = opos - spos
        dist = np.linalg.norm(rel, axis=1, keepdims=True)
        to_opp = rel / np.maximum(dist, 1e-6)
        # signed aim error of self vs direction-to-opponent
        aim_err = np.stack([
            sfd[:, 0] * to_opp[:, 0] + sfd[:, 1] * to_opp[:, 1],          # cos
            sfd[:, 0] * to_opp[:, 1] - sfd[:, 1] * to_opp[:, 0],          # sin
        ], 1)
        # is the opponent aiming at me?
        to_me = -to_opp
        opp_aim = np.stack([
            ofd[:, 0] * to_me[:, 0] + ofd[:, 1] * to_me[:, 1],
            ofd[:, 0] * to_me[:, 1] - ofd[:, 1] * to_me[:, 0],
        ], 1)

        walls = np.stack([
            spos[:, 0] / ARENA_W, (ARENA_W - spos[:, 0]) / ARENA_W,
            spos[:, 1] / ARENA_H, (ARENA_H - spos[:, 1]) / ARENA_H,
        ], 1)

        # k nearest incoming (opponent-owned) bullets, relative pos + vel
        incoming = (self.b_life > 0.0) & (self.b_owner == o)             # (N,B)
        brel = self.b_pos - spos[:, None, :]                            # (N,B,2)
        bd = np.linalg.norm(brel, axis=2)
        bd = np.where(incoming, bd, 1e6)
        order = np.argsort(bd, axis=1)[:, :KNN_BULLETS]                 # (N,k)
        bsel = np.take_along_axis(incoming, order, 1)                   # (N,k) validity
        rp = np.take_along_axis(brel, order[:, :, None], 1)            # (N,k,2)
        rv = np.take_along_axis(self.b_vel, order[:, :, None], 1)
        rp = rp * bsel[:, :, None]
        rv = rv * bsel[:, :, None]
        bul = np.concatenate([
            (rp / self.diag).reshape(self.n, -1),
            (rv / BULLET_SPEED).reshape(self.n, -1),
        ], 1)

        tleft = ((MAX_STEPS - self.t) / MAX_STEPS).astype(np.float32)[:, None]

        obs = np.concatenate([
            spos / [ARENA_W, ARENA_H] * 2.0 - 1.0,
            svel / MAX_SPEED,
            sfd,
            self.health[:, p:p + 1] / HEALTH,
            (1.0 - self.dash_cd[:, p:p + 1] / DASH_COOLDOWN),
            (1.0 - self.fire_cd[:, p:p + 1] / FIRE_COOLDOWN),
            (self.invuln[:, p:p + 1] > 0).astype(np.float32),
            rel / self.diag,
            dist / self.diag,
            aim_err,
            ovel / MAX_SPEED,
            ofd,
            self.health[:, o:o + 1] / HEALTH,
            opp_aim,
            walls,
            bul,
            tleft,
        ], 1).astype(np.float32)
        return obs

    def _obs(self):
        return np.stack([self._build_obs(0), self._build_obs(1)], axis=1)  # (N,2,D)

    # --------------------------------------------------------------- rendering
    def snapshot(self):
        """Lightweight per-arena state for replay recording (arena 0..n)."""
        return {
            "pos": self.pos.copy(),
            "face": self.face.copy(),
            "health": self.health.copy(),
            "invuln": (self.invuln > 0).copy(),
            "dash_t": (self.dash_t > 0).copy(),
            "fired": self.fired_flash.copy(),
            "b_pos": self.b_pos.copy(),
            "b_life": self.b_life.copy(),
            "b_owner": self.b_owner.copy(),
        }

    @staticmethod
    def metadata():
        return {
            "dt": DT, "max_steps": MAX_STEPS,
            "arena_w": ARENA_W, "arena_h": ARENA_H, "agent_r": AGENT_R,
            "bullet_r": BULLET_R, "health": HEALTH,
            "max_speed": MAX_SPEED, "dash_speed": DASH_SPEED,
        }
