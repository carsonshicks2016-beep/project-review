"""
SupraEnv — a Gym-style environment for PPO, hybrid-ready from day one.

Observation = the 40-value sensor vector + a 2-way mode one-hot [race, drift], so
a single policy can be conditioned on what it's being asked to do. (Race is built
now; drift is slice 5, and the goal-conditioned hybrid is slice 6 — all on this
same env, just a different reward + the mode flag flipped.)

Action (race) is 2 continuous values in [-1, 1]:
    a[0] steering,
    a[1] longitudinal (+ = throttle, - = brake)   (auto gearbox handles gears).
Drift adds a 3rd: handbrake.

Reward (race) is dense: metres of centreline progress, a small speed term, and
penalties for going off-track / jerky inputs / crashing. Episodes end by
termination (crash/stall/backwards -> value 0) or truncation (time budget ->
bootstrap), which PPO treats differently.
"""
from __future__ import annotations

import os
import numpy as np

from .config import CarSpec, PPOSpec, RaceReward, SensorSpec, SimSpec, get_car
from .physics import Controls, Vehicle
from .sensors import SensorSuite
from .track import curriculum_track, drift_curriculum_track, named_track

V_REF = 80.0


def straight_speed_bonus(veh, fr, trk, rw, head_err):
    """Reward shaping that makes a race policy SEND IT down the straights.

    The plain progress/align/speed terms are all linear in speed, so the slow
    top-end (this car needs ~30 s of WOT to crawl 70 -> 80 m/s) earns barely more
    than a safe cruise — the policy learns to lift early and never uses the long
    straights. This adds, per substep:

      * a SUPER-LINEAR speed reward ((speed/speed_ref)**speed_exp) so the last few
        m/s toward top speed are worth the most, and
      * a flat reward for holding full throttle,

    both GATED by how straight the road AHEAD is. The lookahead window scales with
    speed (a braking-distance proxy): the faster you go, the further ahead a corner
    must appear before the gate closes — so the policy floors clear straights but
    still lifts in time for the corner at the end of a fast one.

    Returns (speed_bonus, straight_gate, align_gate); the caller adds the throttle
    term itself (it has the action's throttle in scope).
    """
    sp = max(float(veh.speed), 8.0)
    # probe ~0.5–2.8 s down the road (so high speed sees the corner sooner)
    probe = (sp * 0.5, sp * 1.0, sp * 1.8, sp * 2.8)
    look = trk.lookahead_curvature(fr["arc"], probe)
    kmax = max(abs(fr["curvature"]), float(np.max(np.abs(look))))
    straight = float(np.clip(
        1.0 - (kmax - rw.straight_curv) / max(1e-6, rw.straight_span), 0.0, 1.0))
    align = max(0.0, float(np.cos(head_err)))
    v_norm = float(np.clip(veh.speed / rw.speed_ref, 0.0, 1.15))
    bonus = rw.straight_speed * straight * align * (v_norm ** rw.speed_exp)
    return bonus, straight, align


def on_track_factor(fr, trk, edge_band):
    """How much the per-substep DRIVING rewards (progress, align, speed, the straight
    bonus, throttle commit) should count, given where the car sits across the track:

      * 1.0 across the usable inner width (full reward — real racing lines stay free),
      * tapering 1 -> 0 over the outermost `edge_band` metres, and
      * 0.0 once off-track (only the off-track penalty applies out there).

    This kills the exploit where the policy rode the grass/edge flat-out to farm
    speed + centreline-arc progress while flicking back on just often enough to dodge
    the 1.2 s off-track timeout. The smooth taper (vs a hard cliff) gives a gradient
    that pulls a car that drifts wide back toward the road."""
    margin = trk.half - abs(fr["lateral"])      # metres inside the edge (<0 = off-track)
    if edge_band <= 1e-6:
        return 1.0 if margin > 0.0 else 0.0
    return float(np.clip(margin / edge_band, 0.0, 1.0))


class RunningNorm:
    """Welford running mean/var normaliser (per dimension)."""
    def __init__(self, dim: int, clip: float = 5.0):
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.count = 1e-4
        self.clip = clip

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        # Never let one corrupt environment poison every future observation.
        # PPO surfaces the numerical incident separately; this is the final
        # integrity barrier around the long-lived running statistics.
        x = x[np.all(np.isfinite(x), axis=1)]
        if not len(x):
            return False
        b_mean, b_var, b_n = x.mean(0), x.var(0), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        m_a = self.var * self.count
        m_b = b_var * b_n
        self.var = np.maximum(
            (m_a + m_b + delta ** 2 * self.count * b_n / tot) / tot,
            1e-12,
        )
        self.count = tot
        return True

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8),
                       -self.clip, self.clip)


class SupraEnv:
    MODE_DIM = 2

    def __init__(self, mode: str = "race", car: str = "supra",
                 sim: SimSpec | None = None, ppo: PPOSpec | None = None,
                 reward: RaceReward | None = None, fixed_track=None,
                 rng_seed: int | None = None, diagnostics: bool = False):
        self.mode = mode
        self.diagnostics = bool(diagnostics)
        self.spec = get_car(car)
        self.sim = sim or SimSpec()
        self.ppo = ppo or PPOSpec()
        if mode == "drift":
            from .config import DriftReward
            from .drift import DriftScorer
            self.rw = reward or DriftReward()
            self.drift = DriftScorer(self.rw)
        elif mode == "hybrid":
            from .config import HybridReward
            self.rw = reward or HybridReward()   # race backbone + corner-drift style
            self.drift = None
        else:
            self.rw = reward or RaceReward()
            self.drift = None
        sensor_spec = SensorSpec()
        if getattr(self.ppo, "sensor_lookahead_distances", None):
            sensor_spec.lookahead_distances = tuple(self.ppo.sensor_lookahead_distances)
        if getattr(self.ppo, "sensor_pace_block", False):
            sensor_spec.pace_block = True           # Fable Five envelope preview
            if getattr(self.ppo, "sensor_pace_distances", None):
                sensor_spec.pace_distances = tuple(self.ppo.sensor_pace_distances)
        if getattr(self.ppo, "sensor_hybrid_block", False):
            sensor_spec.hybrid_block = True         # fable-v2 SOC + MGU telemetry
        self.sensors = SensorSuite(sensor_spec)
        self.fixed_track = fixed_track
        self.rng = np.random.default_rng(rng_seed)
        self.difficulty = self.ppo.start_difficulty
        self.control_period = max(1, round(1.0 / (self.ppo.control_hz * self.sim.dt)))

        from .app import AutoBox
        self.box = AutoBox(self.spec)
        self.veh = Vehicle(self.spec, self.sim)

        self.action_dim = 3 if mode in ("drift", "hybrid") else 2
        self.obs_dim = self.sensors.obs_size + self.MODE_DIM
        self.sensor_dim = self.sensors.obs_size
        # hybrid = BOTH goals active -> [1, 1] (race fast AND drift the corners)
        self._mode_vec = (np.array([1.0, 1.0]) if mode == "hybrid"
                          else np.array([1.0, 0.0]) if mode == "race"
                          else np.array([0.0, 1.0]))
        self.trk = None
        self._track_diff = None
        self._pool = None
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self):
        # Each env draws from a POOL of tracks at the current difficulty: enough
        # repetition to learn, enough variety (pool x n_envs distinct tracks) to
        # generalise instead of memorising. The pool is regenerated when the
        # curriculum bumps difficulty.
        if self.fixed_track is not None:
            self.trk = self.fixed_track
        else:
            if self._pool is None or self._track_diff != self.difficulty:
                # drift learns on a WIDE->tight curriculum (room to hold a slide),
                # race on the standard one.
                cur = (drift_curriculum_track if self.mode == "drift"
                       else curriculum_track)
                self._pool = [cur(self.difficulty,
                                  seed=int(self.rng.integers(1 << 30)))
                              for _ in range(self.ppo.track_pool)]
                self._track_diff = self.difficulty
            self.trk = self._pool[int(self.rng.integers(len(self._pool)))]
        # exploring starts: drop the car at a random point along the circuit so
        # the policy practises the WHOLE track (middle + end), not just the
        # opening section. Start AT a randomised speed (a bit above/below the
        # natural pace for that corner) + mild entry-angle/offset variety, so it
        # learns to carry/link slides at speed, not just initiate from a stop.
        if getattr(self.ppo, "random_start", False):
            trk = self.trk
            idx = int(self.rng.integers(len(trk.center)))
            sx, sy, syaw = trk.pose_at(idx)
            # lateral offset within the lane + heading jitter -> varied entries
            lat = float(self.rng.uniform(-0.35, 0.35)) * trk.half
            nx, ny = trk.normal[idx]
            sx, sy = sx + nx * lat, sy + ny * lat
            syaw += float(self.rng.uniform(-0.18, 0.18))
            # curvature-aware natural pace, then vary it (and sometimes near-stop)
            arc = float(trk.arc[idx])
            look = np.abs(trk.lookahead_curvature(arc, (15.0, 30.0, 55.0, 85.0)))
            k = max(abs(float(trk.curvature[idx])),
                    float(np.max(look)) if len(look) else 0.0)
            nat = min(34.0, (11.0 / k) ** 0.5) if k > 1e-4 else 34.0
            if self.rng.random() < 0.15:
                v0 = float(self.rng.uniform(0.0, 4.0))         # occasional standing start
            else:
                v0 = float(np.clip(nat * self.rng.uniform(0.55, 1.20), 0.0, 36.0))
        else:
            sx, sy, syaw = self.trk.start_pose()
            v0 = 0.0
        self.veh.reset(sx, sy, syaw, speed=v0)
        self.box.__init__(self.spec)
        fr = self.trk.frame(self.veh.x, self.veh.y)
        self.veh.set_road(fr["grade"], fr["bank"], fr["heading"],
                          fr["z"], fr["vcurv"])
        self.prev_frac = fr["progress"]
        self.cum = 0.0
        self.max_cum = 0.0
        self.best_cum = 0.0
        self.t = 0.0
        self.total_dist = 0.0
        self.off_t = 0.0
        self.stall_t = 0.0
        self.since_prog = 0.0
        self.spin_t = 0.0
        self.step_count = 0
        self.drift_steps = 0
        self.ep_air_time = 0.0     # airborne bookkeeping (info: airtime/jumps)
        self.ep_jumps = 0
        self.ep_max_lg = 0.0
        self._was_air = False
        self.prev_action = np.zeros(self.action_dim)
        self.last_obs = None
        if self.drift:
            self.drift.reset()
        return self._obs()

    def reset_at(self, index, speed=0.0):
        """Deterministic reset at a specific centreline index + speed (no
        randomness). Used by the trainer's deterministic eval to measure drift
        skill around the whole track, reproducibly."""
        self.reset()                       # clears counters + drift scorer
        sx, sy, syaw = self.trk.pose_at(index)
        self.veh.reset(sx, sy, syaw, speed=speed)
        self.box.__init__(self.spec)
        fr = self.trk.frame(self.veh.x, self.veh.y)
        self.veh.set_road(fr["grade"], fr["bank"], fr["heading"],
                          fr["z"], fr["vcurv"])
        self.prev_frac = fr["progress"]    # re-anchor progress to the new spot
        return self._obs()

    def _obs(self):
        o = self.sensors.observe(self.veh, self.trk)
        self.last_obs = o
        return np.concatenate([o.vector, self._mode_vec]).astype(np.float32)

    # ------------------------------------------------------------------ #
    def _hybrid_style(self, fr):
        """Per-substep drift STYLE bonus for hybrid mode: a CONTROLLED slide,
        gated to CORNERS (grip the straights, drift the turns) and to real speed,
        with anti-spin + over-rotation falloff so it stays flashy, not a spin-out.
        On-track only. Returns the style coefficient already applied."""
        c = self.rw
        if fr["off_track"]:
            return 0.0
        ad = abs(np.degrees(self.veh.slip_angle))
        spd = self.veh.speed
        s = np.clip((ad - c.entry_lo_deg) / (c.entry_hi_deg - c.entry_lo_deg), 0.0, 1.0)
        w_entry = s * s * (3 - 2 * s)                      # only real slides count
        w_speed = float(np.clip(spd / c.style_speed_gate, 0.0, 1.0))
        if ad <= c.peak_deg:
            w_rot = 1.0
        else:
            w_rot = max(0.0, 1.0 - (ad - c.peak_deg) / (c.spin_deg - c.peak_deg))
        yaw = abs(np.degrees(self.veh.r))
        w_spin = (1.0 if yaw <= c.spin_rate_deg
                  else max(0.0, 1.0 - (yaw - c.spin_rate_deg) / c.spin_rate_span))
        g = np.clip((abs(fr["curvature"]) - c.corner_curv) / max(c.corner_span, 1e-6),
                    0.0, 1.0)
        corner = g * g * (3 - 2 * g)                       # 0 on straights, 1 in corners
        style = (spd * abs(np.sin(self.veh.slip_angle))
                 * w_entry * w_speed * w_rot * w_spin * corner)
        return c.style * style

    # ------------------------------------------------------------------ #
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        steer = float(a[0])
        long = float(a[1])
        throttle = max(long, 0.0)
        brake = max(-long, 0.0)
        handbrake = max(0.0, float((a[2] + 1) / 2)) if self.action_dim >= 3 else 0.0

        is_drift = self.mode == "drift"
        is_hybrid = self.mode == "hybrid"
        dt = self.sim.dt
        race_reward = 0.0
        style_reward = 0.0         # corner-gated drift STYLE bonus (hybrid mode)
        base_reward = 0.0          # dense drive-at-speed bootstrap for drift mode
        reward_parts = {
            "progress": 0.0, "align": 0.0, "speed": 0.0,
            "straight_speed": 0.0, "throttle_commit": 0.0,
            "offtrack": 0.0, "lap_bonus": 0.0, "style": 0.0,
            "drift": 0.0, "base_drive": 0.0, "base_progress": 0.0,
            "smooth": 0.0, "crash": 0.0,
        }
        diag_values = {"on_track_factor": 1.0, "heading_error": 0.0,
                       "straight_gate": 0.0, "align_gate": 0.0}
        off_substeps = 0
        terminated = False
        termination_reason = None
        last_fr = None
        for _ in range(self.control_period):
            fr = self.trk.frame(self.veh.x, self.veh.y)
            last_fr = fr
            self.veh.surface_grip = (self.spec.offtrack_grip
                                     if fr["off_track"] else 1.0)
            self.veh.set_road(fr["grade"], fr["bank"], fr["heading"],
                              fr["z"], fr["vcurv"])
            # progress (with lap wrap) — tracked for both modes (laps / info)
            frac = fr["progress"]
            d = frac - self.prev_frac
            if d < -0.5:
                d += 1.0
            elif d > 0.5:
                d -= 1.0
            self.prev_frac = frac
            prev_cum = self.cum
            self.cum += d
            self.max_cum = max(self.max_cum, self.cum)
            self.off_t = self.off_t + dt if fr["off_track"] else 0.0

            if not is_drift:
                # race: dense progress + heading-alignment + speed, GATED to on-track
                # (no reward for farming speed/progress out on the grass), off-track pen
                head_err = (self.veh.yaw - fr["heading"] + np.pi) % (2 * np.pi) - np.pi
                ot = on_track_factor(fr, self.trk, self.rw.edge_band)
                p_term = d * self.rw.progress
                a_term = self.rw.align * np.cos(head_err) * (self.veh.speed / V_REF)
                s_term = self.rw.speed * (self.veh.speed / V_REF)
                drive = p_term + a_term + s_term
                # SEND IT: super-linear speed + WOT reward where the road ahead is
                # straight (so it floors long straights instead of safe-cruising).
                sb, straight_gate, align_gate = straight_speed_bonus(
                    self.veh, fr, self.trk, self.rw, head_err)
                drive += sb
                th_term = self.rw.throttle_commit * straight_gate * align_gate * throttle
                drive += th_term
                race_reward += drive * ot
                reward_parts["progress"] += p_term * ot
                reward_parts["align"] += a_term * ot
                reward_parts["speed"] += s_term * ot
                reward_parts["straight_speed"] += sb * ot
                reward_parts["throttle_commit"] += th_term * ot
                diag_values.update(on_track_factor=ot, heading_error=head_err,
                                   straight_gate=straight_gate, align_gate=align_gate)
                if fr["off_track"]:
                    race_reward -= self.rw.offtrack
                    reward_parts["offtrack"] -= self.rw.offtrack
                if int(self.cum) > int(prev_cum) and d > 0:
                    race_reward += self.rw.lap_bonus
                    reward_parts["lap_bonus"] += self.rw.lap_bonus
                crawl = self.t > 1.0 and self.veh.speed < 0.8
                self.stall_t = self.stall_t + dt if crawl else 0.0
                if self.cum > self.best_cum + 1e-3:
                    self.best_cum = self.cum
                    self.since_prog = 0.0
                else:
                    self.since_prog += dt
                back_tol = 0.06 if is_hybrid else 0.04   # drift slides need slack
                if self.off_t > 1.2:
                    terminated = True
                    termination_reason = "offtrack_timeout"
                elif abs(fr["lateral"]) > self.trk.half + 4.0:
                    terminated = True
                    termination_reason = "left_track"
                elif self.stall_t > 2.5:
                    terminated = True
                    termination_reason = "stall"
                elif self.since_prog > 4.0:
                    terminated = True
                    termination_reason = "no_progress"
                elif self.cum < self.max_cum - back_tol:
                    terminated = True
                    termination_reason = "backwards"
                # HYBRID: on the race backbone, add a controlled-drift STYLE bonus
                # that only fires IN CORNERS (grip the straights, drift the turns)
                # and never in the AIR (flying isn't drifting).
                if is_hybrid and not self.veh.airborne:
                    st = self._hybrid_style(fr)
                    style_reward += st
                    reward_parts["style"] += st
            else:
                # drift: speed bootstrap only counts ON TRACK (no reward for
                # sliding across the grass), so it must gauge entry speed to keep
                # the slide on the road. Off-track ends the run fast.
                if fr["off_track"]:
                    off_substeps += 1
                else:
                    if not self.veh.airborne:    # no speed credit mid-air
                        bd = self.rw.drive * (self.veh.speed / V_REF)
                        base_reward += bd
                        reward_parts["base_drive"] += bd
                    # reward getting AROUND the loop (signed, so spinning/backwards
                    # earns nothing) — pushes it to complete, not slide-then-spin.
                    bp = self.rw.progress * d
                    base_reward += bp
                    reward_parts["base_progress"] += bp
                spun = self.t > 1.5 and self.veh.speed < 2.0
                self.spin_t = self.spin_t + dt if spun else 0.0
                # leaving the road ends the run, but with a little grace so a wide
                # slide can be RECOVERED back onto the track (the recovery reward
                # teaches the save); a hard lateral blow-out still ends it at once.
                if self.off_t > self.rw.offtrack_grace:
                    terminated = True
                    termination_reason = "offtrack_timeout"
                elif abs(fr["lateral"]) > self.trk.half + 3.0:
                    terminated = True
                    termination_reason = "left_track"
                elif self.spin_t > self.rw.spin_grace:
                    terminated = True
                    termination_reason = "spin"

            clutch, up, down = self.box.update(self.veh, throttle, dt)
            self.veh.step(Controls(steer=steer, throttle=throttle, brake=brake,
                                   clutch=clutch, handbrake=handbrake,
                                   shift_up=up, shift_down=down))
            self.t += dt
            self.total_dist += self.veh.speed * dt
            if self.veh.airborne:
                self.ep_air_time += dt
                if not self._was_air:
                    self.ep_jumps += 1
            self._was_air = self.veh.airborne
            self.ep_max_lg = max(self.ep_max_lg, self.veh.landing_g)
            if terminated:
                break

        # ---- assemble reward ----
        info = {"laps": self.max_cum, "speed": self.veh.speed,
                "t": self.t, "difficulty": self.difficulty,
                "airtime": self.ep_air_time, "jumps": self.ep_jumps,
                "max_landing_g": self.ep_max_lg}
        if is_drift:
            fr = self.trk.frame(self.veh.x, self.veh.y)
            reward, dinfo = self.drift.step(self.veh, fr, self.trk.half,
                                            self.control_period * dt)
            reward_parts["drift"] = float(reward)
            # gate the drift reward by how much of this step was ON the track
            # (a slide across the grass earns nothing) and by ground contact
            # (a ballistic slip angle isn't a drift).
            on_track = 1.0 - off_substeps / self.control_period
            air_gate = 0.0 if self.veh.airborne else 1.0
            reward = (reward * on_track * air_gate + base_reward
                      - self.rw.offtrack * off_substeps)
            reward_parts["offtrack"] -= self.rw.offtrack * off_substeps
            self.step_count += 1
            # only count ON-TRACK, ON-GROUND drifting toward the metric
            self.drift_steps += int(dinfo["drifting"] and off_substeps == 0
                                    and not self.veh.airborne)
            info.update(dinfo)
            info["drift_frac"] = self.drift_steps / max(1, self.step_count)
        elif is_hybrid:
            # race backbone + corner-drift style, minus action-jerk penalty
            reward = (race_reward + style_reward
                      - self.rw.smooth * float(np.sum((a - self.prev_action) ** 2)))
            reward_parts["smooth"] = -self.rw.smooth * float(np.sum((a - self.prev_action) ** 2))
            self.step_count += 1
            ad = abs(np.degrees(self.veh.slip_angle))
            drifting = (ad > self.rw.drift_min_deg and self.veh.speed > 5.0
                        and not last_fr["off_track"] and not self.veh.airborne)
            self.drift_steps += int(drifting)
            info["drift_frac"] = self.drift_steps / max(1, self.step_count)
            info["slip_deg"] = ad
        else:
            reward = race_reward - self.rw.smooth * float(np.sum((a - self.prev_action) ** 2))
            reward_parts["smooth"] = -self.rw.smooth * float(np.sum((a - self.prev_action) ** 2))
        self.prev_action = a
        if terminated:
            reward -= self.rw.crash
            reward_parts["crash"] = -self.rw.crash
        truncated = (not terminated) and (self.t >= self.ppo.episode_seconds)
        if truncated:
            termination_reason = "time_limit"
        if self.diagnostics:
            info["reward_parts"] = {k: float(v) for k, v in reward_parts.items() if abs(float(v)) > 1e-12}
            info["diagnostics"] = {k: float(v) for k, v in diag_values.items()}
            info["termination_reason"] = termination_reason
        return self._obs(), float(reward), terminated, truncated, info


# --------------------------------------------------------------------------- #
# Parallel (subprocess) vectorised env — runs shards of envs in worker processes
# so rollouts collect across CPU cores. Lives here (numpy only, no torch) so each
# spawned worker stays light. PPO picks this when cfg.n_workers > 1; otherwise the
# in-process SyncVecEnv (in ppo.py) is used. Identical step/reset interface.
# --------------------------------------------------------------------------- #
def _vec_worker(remote, parent_remote, kw, seeds, env_cls):
    parent_remote.close()
    try:
        envs = [env_cls(rng_seed=s, **kw) for s in seeds]
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                out = []
                for e, a in zip(envs, data):
                    # one env blowing up (physics/track edge case) must not
                    # kill the worker — and with it the whole overnight
                    # trainer via EOFError in the parent. Treat it as a
                    # terminal transition and reset that env; only if even
                    # reset fails does the outer guard report the error.
                    try:
                        o, r, te, tr, info = e.step(a)
                        final = o
                        if te or tr:
                            o = e.reset()
                    except Exception as ex:
                        import traceback
                        traceback.print_exc()
                        o = e.reset()          # may raise -> outer guard
                        r, te, tr, final = 0.0, True, False, o
                        info = {"laps": 0.0, "termination_reason": "env_error",
                                "env_error": repr(ex)}
                    out.append((o, r, te, tr, final, info))
                remote.send(out)
            elif cmd == "reset":
                remote.send([e.reset() for e in envs])
            elif cmd == "set_difficulty":
                for e in envs:
                    e.difficulty = data
                remote.send(True)
            elif cmd == "set_fixed_track":
                for e in envs:
                    e.fixed_track = data
                remote.send(True)
            elif cmd == "close":
                break
    except (KeyboardInterrupt, EOFError):
        pass
    except Exception:
        # anything else: tell the parent WHY before dying, so it can respawn
        # this worker (and the log shows the real traceback, not a bare
        # ConnectionResetError three frames away).
        import traceback
        try:
            remote.send(("__error__", traceback.format_exc()))
        except Exception:
            pass
    finally:
        try:
            remote.close()
        except Exception:
            pass


class SubprocVecEnv:
    """Same interface as SyncVecEnv, but the n envs are split into contiguous
    shards run in `n_workers` subprocesses. `n` must be divisible by `n_workers`.

    Crash-tolerant: a worker that dies (or reports an "__error__" sentinel) is
    RESPAWNED in place and its shard contributes one fabricated terminal
    transition for that step — a lost worker costs one episode boundary, not
    the whole overnight run. Respawns are capped so a deterministic crash-loop
    still surfaces instead of spinning forever."""
    MAX_RESPAWNS_PER_WORKER = 5
    # A dead worker raises EOF and was already recoverable. A LIVE but wedged
    # worker used to block remote.recv() forever. Two minutes is deliberately
    # conservative relative to normal sub-second vector steps and lets the
    # outer supervisor recover a truly pathological machine stall.
    REPLY_TIMEOUT_SECONDS = float(os.environ.get("SUPRA_WORKER_TIMEOUT", "120"))

    def __init__(self, n, n_workers, env_cls=SupraEnv, **kw):
        import multiprocessing as mp
        if n % n_workers != 0:
            raise ValueError(f"n_envs ({n}) must be divisible by n_workers ({n_workers})")
        self._ctx = mp.get_context("spawn")
        self._kw = kw
        self._env_cls = env_cls
        self.n = n
        self.s = n // n_workers
        self.remotes, self.procs = [], []
        self._respawns = [0] * n_workers
        self._difficulty = None          # replayed into revived workers
        self._fixed_track = None
        for w in range(n_workers):
            r, p = self._spawn(w)
            self.remotes.append(r)
            self.procs.append(p)
        probe = env_cls(rng_seed=0, **kw)        # one-time, just to read dims
        self.obs_dim, self.act_dim = probe.obs_dim, probe.action_dim
        self.sensor_dim = probe.sensor_dim
        del probe

    def _spawn(self, w):
        seeds = list(range(w * self.s, (w + 1) * self.s))
        r, wr = self._ctx.Pipe()
        p = self._ctx.Process(target=_vec_worker,
                              args=(wr, r, self._kw, seeds, self._env_cls),
                              daemon=True)
        p.start()
        wr.close()
        return r, p

    def _revive(self, w, why):
        """Replace a dead worker and return its shard's fresh reset obs."""
        self._respawns[w] += 1
        if self._respawns[w] > self.MAX_RESPAWNS_PER_WORKER:
            raise RuntimeError(
                f"vec worker {w} died {self._respawns[w]} times — giving up. "
                f"last error:\n{why}")
        print(f"[workers] worker {w} died — respawning "
              f"({self._respawns[w]}/{self.MAX_RESPAWNS_PER_WORKER}). "
              f"cause: {str(why).strip().splitlines()[-1] if why else 'unknown'}",
              flush=True)
        try:
            self.remotes[w].close()
        except Exception:
            pass
        try:
            if self.procs[w].is_alive():
                self.procs[w].terminate()
            self.procs[w].join(timeout=2.0)
            if self.procs[w].is_alive() and hasattr(self.procs[w], "kill"):
                self.procs[w].kill()
                self.procs[w].join(timeout=1.0)
        except Exception:
            pass
        r, p = self._spawn(w)
        self.remotes[w], self.procs[w] = r, p
        # replay post-construction state the old worker carried (a fresh spawn
        # only knows the constructor kwargs)
        def roundtrip(msg):
            r.send(msg)
            timeout = min(30.0, self.REPLY_TIMEOUT_SECONDS)
            if not r.poll(timeout):
                raise TimeoutError(f"revived worker bootstrap timeout after "
                                   f"{timeout:.1f}s on {msg[0]}")
            reply = r.recv()
            if (isinstance(reply, tuple) and len(reply) == 2
                    and reply[0] == "__error__"):
                raise RuntimeError(reply[1] or "revived worker error")
            return reply

        try:
            if self._difficulty is not None:
                roundtrip(("set_difficulty", self._difficulty))
            if self._fixed_track is not None:
                roundtrip(("set_fixed_track", self._fixed_track))
            return roundtrip(("reset", None))
        except (EOFError, BrokenPipeError, OSError, RuntimeError,
                TimeoutError) as exc:
            # Bounded recursion through MAX_RESPAWNS_PER_WORKER: recovery itself
            # can fail, but it can no longer block the entire run forever.
            return self._revive(w, f"bootstrap failed: {exc}")

    def _send_msg(self, w, msg):
        try:
            self.remotes[w].send(msg)
        except Exception:
            pass  # _recv_reply will catch the pipe closure

    def _recv_reply(self, w):
        """recv with one worker; on a dead/errored worker, respawn and
        return (fresh_reset_obs, why) instead. Returns (reply, None) on the
        happy path."""
        try:
            if not self.remotes[w].poll(self.REPLY_TIMEOUT_SECONDS):
                why = (f"worker reply timeout after "
                       f"{self.REPLY_TIMEOUT_SECONDS:.1f}s")
                return self._revive(w, why), why
            reply = self.remotes[w].recv()
            if isinstance(reply, tuple) and len(reply) == 2 \
                    and reply[0] == "__error__":
                return self._revive(w, reply[1]), reply[1] or "worker error"
            return reply, None
        except (EOFError, ConnectionResetError, BrokenPipeError, OSError) as e:
            return self._revive(w, repr(e)), repr(e)

    def reset(self):
        for w in range(len(self.remotes)):
            self._send_msg(w, ("reset", None))
            
        obs = []
        for w in range(len(self.remotes)):
            reply, died = self._recv_reply(w)
            obs.extend(reply)       # a revived worker's reply IS its reset obs
        return np.stack(obs)

    def step(self, actions):
        s = self.s
        for w in range(len(self.remotes)):
            self._send_msg(w, ("step", actions[w * s:(w + 1) * s]))

        nobs, rew, term, trunc, final, infos = [], [], [], [], [], []
        for w in range(len(self.remotes)):
            reply, died = self._recv_reply(w)
            if died is not None:
                # shard restarted mid-step: surface it as terminal transitions
                # (reward 0) so GAE/collect see a clean episode boundary
                for o in reply:
                    nobs.append(o); rew.append(0.0); term.append(True)
                    trunc.append(False); final.append(o)
                    infos.append({"laps": 0.0,
                                  "termination_reason": "worker_died"})
                continue
            for (o, rr, te, tr, fo, info) in reply:
                nobs.append(o); rew.append(rr); term.append(te)
                trunc.append(tr); final.append(fo); infos.append(info)
        return (np.stack(nobs), np.array(rew, np.float32),
                np.array(term, bool), np.array(trunc, bool),
                np.stack(final), infos)

    def set_difficulty(self, d):
        self._difficulty = d
        for w in range(len(self.remotes)):
            self._send_msg(w, ("set_difficulty", d))
        for w in range(len(self.remotes)):
            self._recv_reply(w)

    def set_fixed_track(self, track):
        self._fixed_track = track
        for w in range(len(self.remotes)):
            self._send_msg(w, ("set_fixed_track", track))
        for w in range(len(self.remotes)):
            self._recv_reply(w)

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
            try:
                r.close()
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=1.0)
            if p.is_alive():
                try:
                    p.terminate()
                    p.join(timeout=1.0)
                except Exception:
                    pass
