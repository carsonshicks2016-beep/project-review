"""
PPO environment + parallel rollout machinery.

Deliberately *torch-free* so the multiprocessing rollout workers stay light
(they only need numpy + the physics/track code, never torch).  ``ppo.py``
imports from here.

Key ideas that make a *fast* lap emergent rather than hand-coded:

  * Episodes run on a fixed TIME budget (``PPOSpec.max_episode_seconds``) and
    end early only on a crash or a stall.  The per-step reward is centre-line
    *progress*.  Summed over a fixed time window that is exactly "how far did
    you get in the time you had" -> maximising it means driving FAST.
  * Hitting the time budget is a *truncation*, not a true terminal: the trainer
    bootstraps the value there (see ppo.py).  A crash / stall is a true
    terminal (value 0), so the policy learns the cliff edge.
"""

from __future__ import annotations
import math
import multiprocessing as mp
from collections import deque
import numpy as np

from .config import Config
from .track import make_track, sample_kind
from .physics import Vehicle
from .simulation import observe


# ---------------------------------------------------------------------------
# Reward (one place, shared by the env and the live trainer)
# ---------------------------------------------------------------------------
def step_reward(progress, speed, jerk, dt, ppo, crashed):
    """Per-frame RACE reward.  ``progress`` is metres advanced along the
    centre-line this frame (the dominant term); the rest are small nudges."""
    r = (ppo.progress_weight * progress
         + ppo.speed_weight * speed * dt
         - ppo.jerk_weight * jerk
         - ppo.time_cost)
    if crashed:
        r -= ppo.crash_penalty
    return r


def drift_reward(progress, speed, slip, jerk, dt, ppo, crashed,
                 sustain_t=0.0, slip_rate=0.0, wheelspin=0.0):
    """Per-frame DRIFT reward — speed × angle, made *learnable* and *un-trapped*.

    Key improvements over a naive version:
      * Soft window entry (smoothstep entry_lo -> min_angle): a gradient that
        pulls the car INTO a slide instead of a hard zero just below threshold.
      * Initiation term: rewards breaking the rears loose (wheelspin) at speed,
        strongest *before* the window — the hardest step to discover.
      * Speed-shaping CAPPED at drift_min_speed: rewards getting up to pace then
        stops paying for raw speed, so "grip fast forever" stops being a trap.
    Chain and gutter bonuses are added by the env (they need track/episode state).
    """
    a = abs(slip)
    r = 0.0

    # soft drift window: 0 below entry_lo, ramps to 1 at min_angle (smoothstep)
    lo, hi = ppo.drift_entry_lo_angle, ppo.drift_min_angle
    if a <= lo:
        w = 0.0
    elif a >= hi:
        w = 1.0
    else:
        x = (a - lo) / max(hi - lo, 1e-6)
        w = x * x * (3.0 - 2.0 * x)

    if w > 0.0 and speed >= ppo.drift_min_speed:
        quality = speed * math.sin(min(a, ppo.drift_cap_angle))           # core
        angle_excess = max(0.0, a - ppo.drift_min_angle)
        angle_bonus = ppo.drift_angle_bonus * angle_excess * angle_excess * speed
        speed_bonus = ppo.drift_speed_bonus * (speed / ppo.drift_min_speed)
        sustain_mult = 1.0 + ppo.drift_sustain_bonus * min(sustain_t, ppo.drift_sustain_max)
        r += (ppo.drift_weight * quality + angle_bonus + speed_bonus) * w * dt * sustain_mult

    # initiation: pay for breaking the rears loose at speed (the hard first step),
    # strongest *before* the slide is established (1 - w)
    if speed >= ppo.drift_min_speed:
        r += ppo.drift_init_weight * float(wheelspin) * (1.0 - w) * dt

    # speed shaping, CAPPED at drift speed -> no reward for gripping fast forever
    r += ppo.drift_speed_shaping * min(speed, ppo.drift_min_speed) * dt
    if speed < ppo.drift_min_speed:
        r -= ppo.drift_slow_penalty * dt

    r += ppo.drift_progress_weight * progress        # tiny: just breaks donut ties
    r -= ppo.jerk_weight * jerk
    if crashed:
        r -= getattr(ppo, 'crash_penalty_drift', ppo.crash_penalty)
    return r


class DriftChainTracker:
    """Detects linked drift transitions (direction changes while maintaining speed)."""

    def __init__(self, ppo):
        self.min_speed = ppo.drift_chain_min_speed
        self.transition_window = ppo.drift_chain_transition_window
        self.min_angle = ppo.drift_min_angle
        self.reset()

    def reset(self):
        self.chain = 0
        self.prev_sign = 0
        self.grip_time = 0.0
        self.was_drifting = False
        self.best_chain = 0

    def update(self, slip, speed, dt):
        a = abs(slip)
        in_drift = a >= self.min_angle and speed >= self.min_speed
        sign = 1 if slip > 0 else (-1 if slip < 0 else 0)

        if in_drift:
            if self.was_drifting and sign != 0 and self.prev_sign != 0 and sign != self.prev_sign:
                if self.grip_time <= self.transition_window:
                    self.chain += 1
                    self.best_chain = max(self.best_chain, self.chain)
            if sign != 0:
                self.prev_sign = sign
            self.grip_time = 0.0
            self.was_drifting = True
        else:
            self.grip_time += dt
            if self.grip_time > self.transition_window:
                self.chain = 0
                self.was_drifting = False
            if speed < self.min_speed * 0.7:
                self.chain = 0
                self.was_drifting = False
                self.prev_sign = 0

        return self.chain


class DriftObs:
    """Per-car drift-state tracker.  Produces the extra observation features the
    drift reward depends on (so the policy can actually *see* and optimise them)
    AND is the single source of sustain/chain/gutter used by the reward.  Shared
    by the training env and the live (watch) Car so they can never diverge."""

    N_EXTRA = 6      # sustain, chain, slip-rate, lateral, gutter, grade

    def __init__(self, ppo):
        self.ppo = ppo
        self.reset()

    def reset(self):
        self.drift_time = 0.0
        self.prev_slip = 0.0
        self.slip_rate = 0.0
        self._chain = DriftChainTracker(self.ppo)
        self.chain = 0
        self.gutter = 0.0
        self.grade = 0.0
        self.lateral_n = 0.0

    def update(self, v, track, idx, lateral, dt):
        slip = v.slip_angle
        a = abs(slip)
        if a >= self.ppo.drift_min_angle and v.speed >= self.ppo.drift_min_speed:
            self.drift_time += dt
        else:
            self.drift_time = 0.0
        self.slip_rate = (slip - self.prev_slip) / max(dt, 1e-6)
        self.prev_slip = slip
        self.chain = self._chain.update(slip, v.speed, dt)
        g = 0.0
        if hasattr(track, "gutter_zone") and track.gutter_zone(idx, lateral):
            g = 1.0 if lateral >= 0 else -1.0
        self.gutter = g
        self.grade = track.grade_at(idx) if hasattr(track, "grade_at") else 0.0
        hw = getattr(track, "half_width", 11.0)
        self.lateral_n = float(np.clip(lateral / max(hw, 1e-6), -1.0, 1.0))

    def features(self):
        return np.array([
            math.tanh(self.drift_time / max(self.ppo.drift_sustain_max, 1e-6)),
            math.tanh(self.chain / 4.0),
            float(np.clip(self.slip_rate / 3.0, -1.0, 1.0)),
            self.lateral_n,
            self.gutter,
            float(np.clip(self.grade / 0.25, -1.0, 1.0)),
        ], dtype=np.float64)


def obs_dim(cfg, objective="race"):
    """Observation size for an objective: race = base; drift = base + drift state."""
    return cfg.sensors.n_inputs + (DriftObs.N_EXTRA if objective == "drift" else 0)


# ---------------------------------------------------------------------------
# Running observation normaliser (numpy / Welford, parallel-batch update)
# ---------------------------------------------------------------------------
class RunningNorm:
    def __init__(self, n, clip=5.0, eps=1e-4):
        self.mean = np.zeros(n, dtype=np.float64)
        self.var = np.ones(n, dtype=np.float64)
        self.count = eps
        self.clip = float(clip)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        b_mean = x.mean(axis=0)
        b_var = x.var(axis=0)
        b_count = x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean += delta * b_count / tot
        m_a = self.var * self.count
        m_b = b_var * b_count
        M2 = m_a + m_b + delta ** 2 * self.count * b_count / tot
        self.var = M2 / tot
        self.count = tot

    def normalize(self, x):
        z = (np.asarray(x, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(z, -self.clip, self.clip)

    def state(self):
        return {"mean": self.mean.copy(), "var": self.var.copy(),
                "count": float(self.count)}

    def load(self, s):
        self.mean = np.asarray(s["mean"], dtype=np.float64)
        self.var = np.asarray(s["var"], dtype=np.float64)
        self.count = float(s["count"])
        return self


# ---------------------------------------------------------------------------
# Single-car environment
# ---------------------------------------------------------------------------
class SupraEnv:
    """One car on its own randomly-sampled track, GA-compatible observation."""

    def __init__(self, cfg: Config, seed=None, stage=0, objective="race"):
        self.cfg = cfg
        self.ppo = cfg.ppo
        self.objective = objective       # "race" (lap-time) or "drift"
        self.rng = np.random.default_rng(seed)
        self.stage = stage
        self.drift = (objective == "drift")
        # Race-only hybrid action: 4 continuous (steer/throttle/brake/clutch) +
        # a categorical gear.  Drift keeps the legacy continuous layout.
        self.gear_head = bool(cfg.ppo.gear_head) and not self.drift
        self.n_cont = (cfg.sensors.n_outputs - 2) if self.gear_head else cfg.sensors.n_outputs
        # gear-assist phase: the env auto-shifts and the action is continuous-only
        # (gear is not policy-controlled yet).  Toggled by the trainer.
        self._assist = False
        self.vehicle = Vehicle(cfg.car, cfg.sim)
        self.n_obs = obs_dim(cfg, objective)     # drift gets the extra drift-state inputs
        self.n_act = cfg.sensors.n_outputs
        self.reset()

    def set_stage(self, stage):
        self.stage = stage

    def set_assist(self, assist):
        self._assist = bool(assist)

    def reset(self):
        # domain randomisation: a fresh track each episode.  Drift trains on a
        # cornery, alternating-corner pool so transitions (chaining) can happen;
        # race follows the curriculum tier.
        if self.drift:
            pool = self.ppo.drift_track_pool
            kind = pool[int(self.rng.integers(len(pool)))]
        elif self.cfg.curriculum.enabled:
            kind = sample_kind(self.rng, self.stage)
        else:
            kind = "loop"
        self.track = make_track(self.cfg.track, kind=kind,
                                seed=int(self.rng.integers(1, 1_000_000_000)))
        tr = self.track
        cx, cy = tr.center[0]
        tx, ty = tr.tangent[0]
        self.vehicle.reset(float(cx), float(cy), math.atan2(ty, tx))
        arc, self._idx, _ = tr.progress(float(cx), float(cy))
        self._start = arc
        self._prev_travelled = 0.0
        self._prev_arc = arc
        self.laps = 0
        self.t = 0.0
        self.idle = 0.0
        self.best = 0.0
        # smoothness penalty tracks only the CONTINUOUS controls (a discrete
        # gear change must not read as "jerk")
        self._prev_cont = np.zeros(self.n_cont, dtype=np.float32)
        self._lateral = 0.0
        # drift-specific episode state
        self._driftobs = DriftObs(self.ppo) if self.drift else None
        self._score = 0.0                # clean drift-quality score (best metric)
        self._pos_hist = deque()         # (t, x, y) for the donut/stall check
        return self._obs()

    def _obs(self):
        obs, _, _, self._idx = observe(self.track, self.vehicle,
                                       self.cfg.sensors, self.cfg.car, self._idx)
        if self.drift:
            obs = np.concatenate([obs, self._driftobs.features()])
        return obs.astype(np.float32)

    @staticmethod
    def act_to_controls(raw, gear_head=False):
        """Map the policy's raw action to (steer, throttle, brake, clutch,
        shift_up, shift_down).  Deterministic squash -> no tanh-Jacobian
        correction needed in the PPO log-prob.

        gear_head=True (race hybrid action): raw = [steer, throttle, brake,
        clutch (continuous), gear_class] where gear_class in {0:down,1:hold,2:up}.
        gear_head=False (legacy R^6): raw[4],raw[5] are shift_up/down via
        sigmoid>0.5 (the drift path repurposes raw[4] as a handbrake upstream)."""
        raw = np.asarray(raw, dtype=np.float64)
        if gear_head:
            s = 1.0 / (1.0 + np.exp(-raw[1:4]))         # throttle, brake, clutch
            g = int(round(float(raw[4])))               # 0=down, 1=hold, 2=up
            return (float(np.tanh(raw[0])), float(s[0]), float(s[1]),
                    float(s[2]), g == 2, g == 0)
        s = 1.0 / (1.0 + np.exp(-raw[1:6]))
        return (float(np.tanh(raw[0])), float(s[0]), float(s[1]),
                float(s[2]), bool(s[3] > 0.5), bool(s[4] > 0.5))

    def step(self, raw):
        cfg, ppo = self.cfg, self.ppo
        if self.gear_head and self._assist:
            # gear-assist: continuous-only action; the ENV shifts (rpm schedule)
            r = np.asarray(raw, dtype=np.float64)
            s = 1.0 / (1.0 + np.exp(-r[1:4]))
            steer, thr, brk, clutch = float(np.tanh(r[0])), float(s[0]), float(s[1]), float(s[2])
            su = sd = False
            handbrake = 0.0
            use_manual, use_clutch = False, cfg.control.manual_clutch
        else:
            steer, thr, brk, clutch, su, sd = self.act_to_controls(raw, self.gear_head)
            if self.drift:
                # drift toolkit: auto gears, manual clutch (clutch-kick), and the
                # otherwise-unused shift_up output is repurposed as the HANDBRAKE
                handbrake = float(1.0 / (1.0 + math.exp(-float(raw[4]))))
                use_manual, use_clutch = False, cfg.control.manual_clutch
                su = sd = False
            else:
                handbrake = 0.0
                use_manual, use_clutch = cfg.control.manual_gears, cfg.control.manual_clutch
        ds = cfg.sim.dt / cfg.sim.physics_substeps
        grade = self.track.grade_at(self._idx) if hasattr(self.track, 'grade_at') else 0.0
        for _ in range(cfg.sim.physics_substeps):
            self.vehicle.step(thr, brk, steer, ds, clutch_cmd=clutch,
                              shift_up=su, shift_down=sd,
                              manual_gears=use_manual, manual_clutch=use_clutch,
                              grade=grade, handbrake=handbrake)
        v = self.vehicle
        arc, self._idx, lateral = self.track.progress(v.x, v.y, self._idx)
        self._lateral = lateral
        if arc - self._prev_arc < -self.track.length * 0.5:
            self.laps += 1
        elif arc - self._prev_arc > self.track.length * 0.5:
            self.laps -= 1
        self._prev_arc = arc
        travelled = self.laps * self.track.length + arc - self._start
        progress = travelled - self._prev_travelled
        self._prev_travelled = travelled

        cont = np.asarray(raw, dtype=np.float32)[:self.n_cont]
        jerk = float(np.mean((cont - self._prev_cont) ** 2))
        self._prev_cont = cont
        self.t += cfg.sim.dt

        crashed = bool(self.track.off_track(lateral))
        if self.drift:
            self._driftobs.update(v, self.track, self._idx, lateral, cfg.sim.dt)
            do = self._driftobs
            slip = v.slip_angle
            a = abs(slip)
            reward = drift_reward(progress, v.speed, slip, jerk, cfg.sim.dt, ppo,
                                  crashed, sustain_t=do.drift_time,
                                  slip_rate=do.slip_rate, wheelspin=v.wheelspin)
            if do.chain > 0:
                reward += ppo.drift_chain_bonus * do.chain * cfg.sim.dt
            if do.gutter != 0.0:
                reward += (ppo.gutter_drift_bonus if a >= ppo.drift_min_angle
                           else -ppo.gutter_grip_penalty) * cfg.sim.dt
            # clean, shaping-independent drift-quality score -> the BEST metric
            if a >= ppo.drift_min_angle and v.speed >= ppo.drift_min_speed:
                self._score += v.speed * math.sin(min(a, ppo.drift_cap_angle)) * cfg.sim.dt
            spun = a > ppo.drift_spin_angle and v.speed > 1.0
            if spun and not crashed:
                reward -= ppo.drift_spin_penalty
            terminated = crashed or spun
            # drift stall: donut / stopped (tiny NET displacement over the window)
            self._pos_hist.append((self.t, v.x, v.y))
            while self._pos_hist and self.t - self._pos_hist[0][0] > ppo.drift_stall_seconds:
                self._pos_hist.popleft()
            if self._pos_hist and self.t - self._pos_hist[0][0] >= ppo.drift_stall_seconds * 0.95:
                x0, y0 = self._pos_hist[0][1], self._pos_hist[0][2]
                if math.hypot(v.x - x0, v.y - y0) < ppo.drift_stall_displacement:
                    terminated = True
        else:
            reward = step_reward(progress, v.speed, jerk, cfg.sim.dt, ppo, crashed)
            # bouncing off the rev limiter with a higher gear available = "should
            # have shifted" -> a cost, so 1st-gear-at-redline stops being optimal
            if v.rev_cut and v.gear < len(cfg.car.gear_ratios) - 1:
                reward -= ppo.rev_limit_penalty
            terminated = crashed                  # left the track = true terminal
            # centre-line stall (race): no progress for too long = a failure
            if travelled > self.best + 0.5:
                self.best = travelled
                self.idle = 0.0
            else:
                self.idle += cfg.sim.dt
                if self.idle > ppo.idle_kill_seconds:
                    terminated = True
        # ran out of the time budget = truncation (bootstrap, NOT terminal).
        # Drift uses its own budget so it's independent of the racer's setting.
        budget = ppo.drift_max_episode_seconds if self.drift else ppo.max_episode_seconds
        truncated = (self.t >= budget) and not terminated
        report = self._score if self.drift else travelled   # drift: report quality

        return (self._obs(), float(reward), bool(terminated), bool(truncated),
                {"travelled": float(report)})


# ---------------------------------------------------------------------------
# Vectorised environments
# ---------------------------------------------------------------------------
# Each step returns 7 aligned arrays over the n_envs:
#   next_obs[n,obs], reward[n], terminated[n] bool, truncated[n] bool,
#   final_obs[n,obs] (terminal obs where done; ignore otherwise),
#   lap_done[n] (1.0 if the just-ended episode completed >=1 lap, else 0),
#   travelled[n] (metres; final value where done, current otherwise)
# On done an env auto-resets; ``final_obs`` keeps the terminal observation so
# the trainer can bootstrap a *truncated* episode's value.

def _env_step_one(env, action):
    no, r, term, trunc, info = env.step(action)
    final = no.copy()
    if term or trunc:
        # report how much of a lap this episode covered (capped), so the
        # curriculum can promote on a *reachable* completion fraction rather
        # than on completing whole laps (which never happened on long tracks).
        cap = env.cfg.curriculum.promote_fraction_cap
        lap = min(info["travelled"] / max(env.track.length, 1.0), cap)
        trav = info["travelled"]
        no = env.reset()
    else:
        lap, trav = 0.0, info["travelled"]
    return no, final, r, term, trunc, lap, trav


class DummyVecEnv:
    """In-process vector env (used when n_workers<=1; also the safe fallback)."""

    def __init__(self, cfg, seeds, stage, objective="race"):
        self.envs = [SupraEnv(cfg, seed=s, stage=stage, objective=objective)
                     for s in seeds]
        self.n_envs = len(self.envs)
        self.n_obs = obs_dim(cfg, objective)
        self.n_act = cfg.sensors.n_outputs

    def reset(self):
        return np.asarray([e.reset() for e in self.envs], dtype=np.float32)

    def set_stage(self, stage):
        for e in self.envs:
            e.set_stage(stage)

    def set_assist(self, assist):
        for e in self.envs:
            e.set_assist(assist)

    def step(self, actions):
        cols = [[] for _ in range(7)]
        for e, a in zip(self.envs, actions):
            out = _env_step_one(e, a)
            for k in range(7):
                cols[k].append(out[k])
        return _pack(cols)

    def close(self):
        pass


def _worker(conn, cfg, seeds, stage, objective):
    envs = [SupraEnv(cfg, seed=s, stage=stage, objective=objective) for s in seeds]
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "step":
                conn.send([_env_step_one(e, a) for e, a in zip(envs, data)])
            elif cmd == "reset":
                conn.send([e.reset() for e in envs])
            elif cmd == "set_stage":
                for e in envs:
                    e.set_stage(data)
            elif cmd == "set_assist":
                for e in envs:
                    e.set_assist(data)
            elif cmd == "close":
                break
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        conn.close()


class SubprocVecEnv:
    """Envs split across worker processes for true (GIL-free) parallel rollouts."""

    def __init__(self, cfg, seed_groups, stage, objective="race",
                 recv_timeout=300.0):
        ctx = mp.get_context("spawn")
        self.counts = [len(g) for g in seed_groups]
        self.n_envs = sum(self.counts)
        self.n_obs = obs_dim(cfg, objective)
        self.n_act = cfg.sensors.n_outputs
        # if a worker dies/hangs, don't block the trainer forever: a step or
        # reset that goes silent this long raises so the loop can checkpoint+exit.
        self._recv_timeout = float(recv_timeout)
        self.conns, self.procs = [], []
        for g in seed_groups:
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, cfg, g, stage, objective),
                            daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)

    def _recv(self, c):
        if not c.poll(self._recv_timeout):
            raise RuntimeError(
                f"a rollout worker went silent for {self._recv_timeout:.0f}s "
                "(crashed or hung) — aborting so the trainer can checkpoint "
                "and exit cleanly")
        return c.recv()

    def reset(self):
        for c in self.conns:
            c.send(("reset", None))
        obs = []
        for c in self.conns:
            obs.extend(self._recv(c))
        return np.asarray(obs, dtype=np.float32)

    def set_stage(self, stage):
        for c in self.conns:
            c.send(("set_stage", stage))

    def set_assist(self, assist):
        for c in self.conns:
            c.send(("set_assist", assist))

    def step(self, actions):
        i = 0
        for c, n in zip(self.conns, self.counts):
            c.send(("step", actions[i:i + n]))
            i += n
        cols = [[] for _ in range(7)]
        for c in self.conns:
            for out in self._recv(c):
                for k in range(7):
                    cols[k].append(out[k])
        return _pack(cols)

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()


def _pack(cols):
    nexts, finals, rews, terms, truncs, laps, travs = cols
    return (np.asarray(nexts, np.float32), np.asarray(rews, np.float32),
            np.asarray(terms, bool), np.asarray(truncs, bool),
            np.asarray(finals, np.float32), np.asarray(laps, np.float32),
            np.asarray(travs, np.float32))


def auto_workers(n_envs, requested):
    import os
    if requested is not None and requested >= 1:
        return min(int(requested), n_envs)
    cpu = os.cpu_count() or 2
    return max(1, min(n_envs, cpu - 1))


def make_vec_env(cfg, n_envs, n_workers, base_seed, stage, objective="race"):
    seeds = [int(base_seed) + 1 + i for i in range(n_envs)]
    nw = auto_workers(n_envs, n_workers)
    if nw <= 1:
        return DummyVecEnv(cfg, seeds, stage, objective)
    groups = [g for g in (seeds[w::nw] for w in range(nw)) if g]
    return SubprocVecEnv(cfg, groups, stage, objective)
