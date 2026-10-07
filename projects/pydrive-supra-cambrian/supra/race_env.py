"""
RaceEnv — Multi-Agent Competitive Racing Environment for PPO Self-Play.

A single "learning" agent races against 1-4 frozen opponent policies on the same
track. All vehicles share the physics simulation: collisions are resolved via
SAT/OBB impulses, and damage propagates through the existing 4-wheel model.

The learning agent's observation includes the upgraded multi-agent sensor block
(beam type, opponent relative state, own damage), and its reward includes:
  - dense progress (same as SupraEnv race mode)
  - overtake bonuses (+2.0 for a clean positional gain)
  - collision penalties (-5.0 scaled by KE absorbed / durability)
  - drafting bonus (+0.05/s when slipstreaming behind an opponent)

Frozen opponents are loaded from .pt checkpoint files and run deterministically
(mean action, no exploration noise). They use the standard SupraEnv sensor suite
so they don't need the multi-agent observation block — they just drive.
"""
from __future__ import annotations

import os
import numpy as np
import torch

from .config import CarSpec, PPOSpec, RaceReward, SensorSpec, SimSpec, get_car
from .physics import Controls, Vehicle, resolve_collisions, sat_collision
from .sensors import SensorSuite
from .track import curriculum_track, named_track
from .ppo_env import RunningNorm, straight_speed_bonus, on_track_factor

V_REF = 80.0


class FrozenOpponent:
    """A frozen policy checkpoint that drives an opponent car deterministically."""

    def __init__(self, checkpoint_path: str, car: str = None):
        """Loads a frozen policy and spins up a dedicated vehicle instance for it."""
        self.name = os.path.basename(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if car is None:
            car = ckpt.get("car", "supra")
            
        self.spec = get_car(car)
        self.veh = Vehicle(self.spec)
        sensor_spec = SensorSpec()
        if ckpt.get("sensor_lookahead_distances"):
            sensor_spec.lookahead_distances = tuple(ckpt["sensor_lookahead_distances"])
        self.sensors = SensorSuite(sensor_spec)

        # load the ActorCritic from the checkpoint
        from .ppo import ActorCritic
        obs_dim = ckpt.get("obs_dim", self.sensors.obs_size + 2)
        act_dim = ckpt.get("act_dim", 2)
        hidden = ckpt.get("hidden", (128, 128))
        self.policy = ActorCritic(obs_dim, act_dim, hidden)
        self.policy.load_state_dict(ckpt["state_dict"])
        self.policy.eval()
        self.act_dim = act_dim
        self.sdim = ckpt.get("sdim", self.sensors.obs_size)
        
        # load normalization stats
        self.norm = RunningNorm(self.sdim)
        if "norm_mean" in ckpt:
            self.norm.mean = ckpt["norm_mean"]
            self.norm.var = ckpt["norm_var"]
            self.norm.count = ckpt["norm_count"]

        from .app import AutoBox
        self.box = AutoBox(self.spec)
        self._mode_vec = np.array([1.0, 0.0], dtype=np.float32)  # race mode

    def reset(self, x, y, yaw, speed=0.0):
        self.veh.reset(x, y, yaw, speed=speed)
        self.box.__init__(self.spec)

    def act(self, trk, opponents=None):
        """Get action from frozen policy (deterministic: mean only, no noise)."""
        obs = self.sensors.observe(self.veh, trk, opponents=opponents)
        
        vec = obs.vector
        if len(vec) < self.sdim:
            # Multi-agent checkpoint running without opponents provided -> pad with zeros
            pad = np.zeros(self.sdim - len(vec), dtype=np.float32)
            vec = np.concatenate([vec, pad])
            
        snorm = self.norm.normalize(vec[:self.sdim])
        vec = np.concatenate([snorm, self._mode_vec]).astype(np.float32)
        with torch.no_grad():
            t = torch.from_numpy(vec).unsqueeze(0)
            mean, _ = self.policy(t)
            a = mean.squeeze(0).numpy()
        a = np.clip(a, -1.0, 1.0)
        steer = float(a[0])
        long = float(a[1])
        throttle = max(long, 0.0)
        brake = max(-long, 0.0)
        handbrake = max(0.0, float((a[2] + 1) / 2)) if self.act_dim >= 3 else 0.0
        return steer, throttle, brake, handbrake


class RaceEnv:
    """Multi-agent racing environment. One learning agent + N frozen opponents."""
    MODE_DIM = 2

    def __init__(self, car: str = "supra",
                 opponent_configs: list[dict] | None = None,
                 sim: SimSpec | None = None, ppo: PPOSpec | None = None,
                 reward: RaceReward | None = None, fixed_track=None,
                 rng_seed: int | None = None, diagnostics: bool = False):
        self.spec = get_car(car)
        self.diagnostics = bool(diagnostics)
        self.sim = sim or SimSpec()
        self.ppo = ppo or PPOSpec()
        self.rw = reward or RaceReward()
        sensor_spec = SensorSpec()
        if getattr(self.ppo, "sensor_lookahead_distances", None):
            sensor_spec.lookahead_distances = tuple(self.ppo.sensor_lookahead_distances)
        self.sensors = SensorSuite(sensor_spec)
        self.fixed_track = fixed_track
        self.rng = np.random.default_rng(rng_seed)
        self.difficulty = self.ppo.start_difficulty
        self.control_period = max(1, round(1.0 / (self.ppo.control_hz * self.sim.dt)))

        from .app import AutoBox
        self.box = AutoBox(self.spec)
        self.veh = Vehicle(self.spec, self.sim)

        # Frozen opponents
        self.opponents: list[FrozenOpponent] = []
        if opponent_configs:
            for cfg in opponent_configs:
                opp = FrozenOpponent(
                    checkpoint_path=cfg["checkpoint"],
                    car=cfg.get("car", "supra")
                )
                self.opponents.append(opp)

        # Multi-agent obs: base (58) + mode (2) + beam_type (9) + opp_state (3) + damage (6) = 78
        self._all_vehicles: list[Vehicle] = []
        self.action_dim = 2
        # obs_dim depends on whether we have opponents
        base = self.sensors.obs_size
        if self.opponents:
            base += 9 + 3 + 6  # beam_type + opp_state + damage_state
        self.obs_dim = base + self.MODE_DIM
        self.sensor_dim = base
        self._mode_vec = np.array([1.0, 0.0], dtype=np.float32)  # race mode

        self.trk = None
        self._pool = None
        self._track_diff = None

        # Race state
        self._positions = []  # track position ordering for overtake detection
        self.reset()

    def _all_vehs(self) -> list[Vehicle]:
        return [self.veh] + [o.veh for o in self.opponents]

    def reset(self):
        # Track selection (same as SupraEnv)
        if self.fixed_track is not None:
            self.trk = self.fixed_track
        else:
            if self._pool is None or self._track_diff != self.difficulty:
                self._pool = [curriculum_track(self.difficulty,
                              seed=int(self.rng.integers(1 << 30)))
                              for _ in range(self.ppo.track_pool)]
                self._track_diff = self.difficulty
            self.trk = self._pool[int(self.rng.integers(len(self._pool)))]

        # Staggered grid start
        sx, sy, syaw = self.trk.start_pose()
        nx, ny = np.cos(syaw + np.pi/2), np.sin(syaw + np.pi/2)
        # Ego car starts at pole position
        self.veh.reset(sx, sy, syaw, speed=0.0)
        self.box.__init__(self.spec)

        # Opponents start behind, staggered
        for i, opp in enumerate(self.opponents):
            offset = (i + 1) * 6.0  # 6m spacing behind
            ox = sx - np.cos(syaw) * offset
            oy = sy - np.sin(syaw) * offset
            # slight lateral offset for visual variety
            lat = ((i % 2) * 2 - 1) * 0.8
            ox += nx * lat
            oy += ny * lat
            opp.reset(ox, oy, syaw, speed=0.0)

        self._all_vehicles = self._all_vehs()

        # Race tracking
        fr = self.trk.frame(self.veh.x, self.veh.y)
        self.veh.set_road(fr["grade"], fr["bank"], fr["heading"],
                          fr["z"], fr["vcurv"])
        self.prev_frac = fr["progress"]
        self.cum = 0.0
        self.max_cum = 0.0
        self.best_cum = 0.0
        self.t = 0.0
        self.off_t = 0.0
        self.stall_t = 0.0
        self.since_prog = 0.0
        self.step_count = 0
        self.prev_action = np.zeros(self.action_dim)

        # Position tracking for overtakes
        self._ego_rank = 0
        self._pos_cums = None            # re-anchor cumulative-progress ranking
        self._update_positions()
        self._prev_rank = self._ego_rank

        return self._obs()

    def reset_at(self, index, speed=0.0):
        """Deterministic reset at a specific centreline index + speed."""
        self.reset()
        sx, sy, syaw = self.trk.pose_at(index)
        nx, ny = np.cos(syaw + np.pi/2), np.sin(syaw + np.pi/2)
        
        self.veh.reset(sx, sy, syaw, speed=speed)
        self.box.__init__(self.spec)
        
        for i, opp in enumerate(self.opponents):
            offset = (i + 1) * 6.0
            ox = sx - np.cos(syaw) * offset
            oy = sy - np.sin(syaw) * offset
            lat = ((i % 2) * 2 - 1) * 0.8
            ox += nx * lat
            oy += ny * lat
            opp.reset(ox, oy, syaw, speed=speed)
            
        fr = self.trk.frame(self.veh.x, self.veh.y)
        self.veh.set_road(fr["grade"], fr["bank"], fr["heading"],
                          fr["z"], fr["vcurv"])
        self.prev_frac = fr["progress"]
        self._pos_cums = None            # grid moved: re-anchor the ranking
        self._update_positions()
        self._prev_rank = self._ego_rank

        return self._obs()

    def _update_positions(self):
        """Rank all vehicles by cumulative track progress. The instantaneous
        fraction wraps 0.99 -> 0.01 at the lap line, which ranked whoever
        crossed it last for a step (a phantom overtake for everyone else once
        per lap) — so accumulate wrap-corrected deltas per vehicle instead.
        Anchors lazily (re)initialise whenever the grid is (re)placed:
        reset/reset_at set _pos_cums to None."""
        fracs = np.array([self.trk.frame(v.x, v.y)["progress"]
                          for v in self._all_vehicles])
        if getattr(self, "_pos_cums", None) is None \
                or len(self._pos_cums) != len(fracs):
            self._pos_prev_fracs = fracs.copy()
            self._pos_cums = np.zeros(len(fracs))
        d = fracs - self._pos_prev_fracs
        d[d < -0.5] += 1.0
        d[d > 0.5] -= 1.0
        self._pos_prev_fracs = fracs
        self._pos_cums += d
        # Sort descending (leader = most progress)
        ranked = sorted(range(len(fracs)), key=lambda i: self._pos_cums[i],
                        reverse=True)
        self._ego_rank = ranked.index(0)

    def _obs(self):
        o = self.sensors.observe(self.veh, self.trk, opponents=self._all_vehicles)
        self.last_obs = o
        return np.concatenate([o.vector, self._mode_vec]).astype(np.float32)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        steer = float(a[0])
        long = float(a[1])
        throttle = max(long, 0.0)
        brake = max(-long, 0.0)

        dt = self.sim.dt
        race_reward = 0.0
        collision_penalty = 0.0
        reward_parts = {
            "progress": 0.0, "align": 0.0, "speed": 0.0,
            "straight_speed": 0.0, "throttle_commit": 0.0,
            "offtrack": 0.0, "overtake": 0.0, "draft": 0.0,
            "collision": 0.0, "smooth": 0.0, "crash": 0.0,
        }
        diag_values = {"on_track_factor": 1.0, "heading_error": 0.0,
                       "straight_gate": 0.0, "align_gate": 0.0}
        terminated = False
        termination_reason = None

        for _ in range(self.control_period):
            # --- Ego vehicle physics ---
            fr = self.trk.frame(self.veh.x, self.veh.y)
            self.veh.surface_grip = (self.spec.offtrack_grip
                                     if fr["off_track"] else 1.0)
            self.veh.set_road(fr["grade"], fr["bank"], fr["heading"],
                              fr["z"], fr["vcurv"])

            # Progress tracking
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

            # Dense race reward, GATED to on-track (no grass/edge farming)
            head_err = (self.veh.yaw - fr["heading"] + np.pi) % (2 * np.pi) - np.pi
            ot = on_track_factor(fr, self.trk, self.rw.edge_band)
            p_term = d * self.rw.progress
            a_term = self.rw.align * np.cos(head_err) * (self.veh.speed / V_REF)
            s_term = self.rw.speed * (self.veh.speed / V_REF)
            drive = p_term + a_term + s_term
            # SEND IT: super-linear speed + WOT reward on straights (see ppo_env)
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

            # Termination conditions
            crawl = self.t > 1.0 and self.veh.speed < 0.8
            self.stall_t = self.stall_t + dt if crawl else 0.0
            if self.cum > self.best_cum + 1e-3:
                self.best_cum = self.cum
                self.since_prog = 0.0
            else:
                self.since_prog += dt
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
            elif self.cum < self.max_cum - 0.04:
                terminated = True
                termination_reason = "backwards"

            # --- Step ego vehicle ---
            clutch, up, down = self.box.update(self.veh, throttle, dt)
            self.veh.step(Controls(steer=steer, throttle=throttle, brake=brake,
                                   clutch=clutch, handbrake=0.0,
                                   shift_up=up, shift_down=down))

            # --- Step opponent vehicles ---
            for opp in self.opponents:
                ofr = self.trk.frame(opp.veh.x, opp.veh.y)
                opp.veh.surface_grip = (opp.spec.offtrack_grip
                                        if ofr["off_track"] else 1.0)
                opp.veh.set_road(ofr["grade"], ofr["bank"], ofr["heading"],
                                 ofr["z"], ofr["vcurv"])
                other_vehs = [v for v in self._all_vehicles if v != opp.veh]
                os, ot, ob, oh = opp.act(self.trk, opponents=other_vehs)
                oc, ou, od = opp.box.update(opp.veh, ot, dt)
                opp.veh.step(Controls(steer=os, throttle=ot, brake=ob,
                                      clutch=oc, handbrake=oh,
                                      shift_up=ou, shift_down=od))

            # --- Resolve collisions between ALL vehicles ---
            # Check for any physical overlap before resolution to apply strict penalty
            for opp in self.opponents:
                colliding, _, _ = sat_collision(self.veh.get_obb(), opp.veh.get_obb())
                if colliding:
                    collision_penalty += 0.5  # Heavy penalty for ANY contact, not just mechanical damage

            pre_dmg = self.veh.engine_damage + self.veh.aero_damage
            resolve_collisions(self._all_vehicles, dt)
            post_dmg = self.veh.engine_damage + self.veh.aero_damage
            if post_dmg > pre_dmg:
                collision_penalty += 5.0 * (post_dmg - pre_dmg)
            reward_parts["collision"] = -collision_penalty

            self.t += dt
            if terminated:
                break

        # --- Overtake detection ---
        self._prev_rank = self._ego_rank
        self._update_positions()
        overtake_reward = 0.0
        if self._ego_rank < self._prev_rank:
            # Gained a position!
            overtake_reward = 2.0 * (self._prev_rank - self._ego_rank)
        reward_parts["overtake"] = overtake_reward

        # --- Drafting bonus ---
        draft_reward = 0.0
        for opp in self.opponents:
            dx = opp.veh.x - self.veh.x
            dy = opp.veh.y - self.veh.y
            dist = np.hypot(dx, dy)
            if 2.0 < dist < 12.0 and self.veh.speed > 10.0:
                # Check if opponent is roughly ahead (in ego forward direction)
                fwd_dot = dx * np.cos(self.veh.yaw) + dy * np.sin(self.veh.yaw)
                if fwd_dot > 0:
                    draft_reward += 0.05 * (1.0 - dist / 12.0)
        reward_parts["draft"] = draft_reward

        # --- Assemble reward ---
        smooth_penalty = self.rw.smooth * float(np.sum((a - self.prev_action) ** 2))
        reward_parts["smooth"] = -smooth_penalty
        reward = (race_reward + overtake_reward + draft_reward
                  - collision_penalty - smooth_penalty)
        self.prev_action = a
        if terminated:
            reward -= self.rw.crash
            reward_parts["crash"] = -self.rw.crash

        truncated = (not terminated) and (self.t >= self.ppo.episode_seconds)
        if truncated:
            termination_reason = "time_limit"
        self.step_count += 1

        info = {
            "laps": self.max_cum,
            "speed": self.veh.speed,
            "t": self.t,
            "rank": self._ego_rank + 1,
            "n_cars": len(self._all_vehicles),
            "overtakes": max(0, self._prev_rank - self._ego_rank),
            "engine_dmg": self.veh.engine_damage,
            "aero_dmg": self.veh.aero_damage,
            "turbo_broken": self.veh.turbo_broken,
            "collision_penalty": collision_penalty,
        }
        if self.diagnostics:
            info["reward_parts"] = {k: float(v) for k, v in reward_parts.items() if abs(float(v)) > 1e-12}
            info["diagnostics"] = {k: float(v) for k, v in diag_values.items()}
            info["termination_reason"] = termination_reason

        return self._obs(), float(reward), terminated, truncated, info
