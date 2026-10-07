import numpy as np

from .config import CarSpec, PPOSpec, RaceReward, SensorSpec, SimSpec, get_car
from .physics import Controls, Vehicle, resolve_collisions, sat_collision
from .ppo_env import straight_speed_bonus, on_track_factor
from .sensors import SensorSuite
from .track import curriculum_track, named_track

V_REF = 80.0

class MultiAgentRaceEnv:
    """True Multi-Agent Self-Play racing environment. N learning agents race and learn simultaneously."""
    MODE_DIM = 2

    def __init__(self, n_agents: int = 4, car: str = "supra",
                 sim: SimSpec | None = None, ppo: PPOSpec | None = None,
                 reward: RaceReward | None = None, fixed_track=None,
                 rng_seed: int | None = None, diagnostics: bool = False):
        self.n_agents = n_agents
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
        self.vehs = [Vehicle(self.spec, self.sim) for _ in range(n_agents)]
        self.boxes = [AutoBox(self.spec) for _ in range(n_agents)]

        base = self.sensors.obs_size + 9 + 3 + 6
        self.obs_dim = base + self.MODE_DIM
        self.sensor_dim = base
        self.action_dim = 2
        self._mode_vec = np.array([1.0, 0.0], dtype=np.float32)

        self.trk = None
        self._pool = None
        self._track_diff = None

        self.reset()

    def reset(self):
        if self.fixed_track is not None:
            self.trk = self.fixed_track
        else:
            if self._pool is None or self._track_diff != self.difficulty:
                self._pool = [curriculum_track(self.difficulty,
                              seed=int(self.rng.integers(1 << 30)))
                              for _ in range(self.ppo.track_pool)]
                self._track_diff = self.difficulty
            self.trk = self._pool[int(self.rng.integers(len(self._pool)))]

        sx, sy, syaw = self.trk.start_pose()
        nx, ny = np.cos(syaw + np.pi/2), np.sin(syaw + np.pi/2)
        
        for i, veh in enumerate(self.vehs):
            offset = i * 6.0
            ox = sx - np.cos(syaw) * offset
            oy = sy - np.sin(syaw) * offset
            lat = ((i % 2) * 2 - 1) * 0.8 if i > 0 else 0
            ox += nx * lat
            oy += ny * lat
            veh.reset(ox, oy, syaw, speed=0.0)
            self.boxes[i].__init__(self.spec)

        self.prev_fracs = np.array([self.trk.frame(v.x, v.y)["progress"] for v in self.vehs])
        self.cums = np.zeros(self.n_agents)
        self.max_cums = np.zeros(self.n_agents)
        self.best_cums = np.zeros(self.n_agents)
        self.t = 0.0
        self.off_ts = np.zeros(self.n_agents)
        self.stall_ts = np.zeros(self.n_agents)
        self.since_progs = np.zeros(self.n_agents)
        self.step_count = 0
        self.prev_actions = np.zeros((self.n_agents, self.action_dim))
        
        self.ranks = np.arange(self.n_agents)
        self.prev_ranks = np.arange(self.n_agents)

        return self._obs()

    def reset_at(self, index, speed=0.0):
        self.reset()
        sx, sy, syaw = self.trk.pose_at(index)
        nx, ny = np.cos(syaw + np.pi/2), np.sin(syaw + np.pi/2)
        
        for i, veh in enumerate(self.vehs):
            offset = i * 6.0
            ox = sx - np.cos(syaw) * offset
            oy = sy - np.sin(syaw) * offset
            lat = ((i % 2) * 2 - 1) * 0.8 if i > 0 else 0
            ox += nx * lat
            oy += ny * lat
            veh.reset(ox, oy, syaw, speed=speed)
            self.boxes[i].__init__(self.spec)
            fr = self.trk.frame(veh.x, veh.y)
            veh.set_road(fr["grade"], fr["bank"], fr["heading"], fr["z"], fr["vcurv"])
            
        self.prev_fracs = np.array([self.trk.frame(v.x, v.y)["progress"] for v in self.vehs])
        self._update_positions()
        
        return self._obs()

    def _update_positions(self):
        # Rank by CUMULATIVE progress (self.cums), not the instantaneous track
        # fraction: the fraction wraps 0.99 -> 0.01 at the lap line, which
        # dropped the leader to last for a step and paid every other car a
        # phantom overtake bonus once per lap.
        ranked = sorted(range(self.n_agents), key=lambda i: self.cums[i],
                        reverse=True)
        for rank, i in enumerate(ranked):
            self.ranks[i] = rank

    def _respawn(self, i):
        """Restart ONE crashed agent's episode in place: drop it back on the track
        at racing pace in the spot with the most clearance to the other cars, and
        reset only ITS per-episode trackers. The rest of the grid races on, so a
        single crash never ends anyone else's run (replaces the old forced
        joint-termination). PPO already sees terminated[i]=True for this step, so
        this fresh obs simply begins a new trajectory for that agent."""
        veh = self.vehs[i]
        M = len(self.trk.center)
        others = [v for j, v in enumerate(self.vehs) if j != i]
        # sample a few candidate drop-in points, keep the one furthest from traffic
        best = None
        for _ in range(8):
            idx = int(self.rng.integers(M))
            x, y, yaw = self.trk.pose_at(idx)
            gap = min((np.hypot(o.x - x, o.y - y) for o in others), default=1e9)
            if best is None or gap > best[0]:
                best = (gap, idx, x, y, yaw)
        _, idx, x, y, yaw = best
        # racing drop-in speed scaled to the local corner radius (conservative so
        # it never spawns straight into a spin); flat-out cap on straights
        k = abs(float(self.trk.curvature[idx]))
        nat = min(40.0, (11.0 / k) ** 0.5) if k > 1e-4 else 40.0
        veh.reset(x, y, yaw, speed=nat * 0.8)
        self.boxes[i].__init__(self.spec)
        fr = self.trk.frame(veh.x, veh.y)
        veh.set_road(fr["grade"], fr["bank"], fr["heading"], fr["z"], fr["vcurv"])
        # re-anchor only this agent's episode state (everyone else is untouched)
        self.prev_fracs[i] = fr["progress"]
        self.cums[i] = 0.0
        self.max_cums[i] = 0.0
        self.best_cums[i] = 0.0
        self.off_ts[i] = 0.0
        self.stall_ts[i] = 0.0
        self.since_progs[i] = 0.0
        self.prev_actions[i] = 0.0

    def _obs(self):
        obs_list = []
        for i, veh in enumerate(self.vehs):
            o = self.sensors.observe(veh, self.trk, opponents=self.vehs)
            v = np.concatenate([o.vector, self._mode_vec]).astype(np.float32)
            obs_list.append(v)
        return np.stack(obs_list)

    def step(self, actions):
        a = np.clip(np.asarray(actions, dtype=float), -1.0, 1.0)
        steers = a[:, 0]
        longs = a[:, 1]
        throttles = np.maximum(longs, 0.0)
        brakes = np.maximum(-longs, 0.0)

        dt = self.sim.dt
        race_rewards = np.zeros(self.n_agents)
        collision_penalties = np.zeros(self.n_agents)
        terminated = np.zeros(self.n_agents, dtype=bool)
        termination_reasons = [None] * self.n_agents
        reward_parts = [
            {"progress": 0.0, "align": 0.0, "speed": 0.0,
             "straight_speed": 0.0, "throttle_commit": 0.0,
             "offtrack": 0.0, "overtake": 0.0, "draft": 0.0,
             "collision": 0.0, "smooth": 0.0, "crash": 0.0}
            for _ in range(self.n_agents)
        ]
        diag_values = [
            {"on_track_factor": 1.0, "heading_error": 0.0,
             "straight_gate": 0.0, "align_gate": 0.0}
            for _ in range(self.n_agents)
        ]

        for _ in range(self.control_period):
            for i, veh in enumerate(self.vehs):
                if terminated[i]:
                    continue
                fr = self.trk.frame(veh.x, veh.y)
                veh.surface_grip = self.spec.offtrack_grip if fr["off_track"] else 1.0
                veh.set_road(fr["grade"], fr["bank"], fr["heading"], fr["z"], fr["vcurv"])
                
                frac = fr["progress"]
                d = frac - self.prev_fracs[i]
                if d < -0.5: d += 1.0
                elif d > 0.5: d -= 1.0
                self.prev_fracs[i] = frac
                self.cums[i] += d
                self.max_cums[i] = max(self.max_cums[i], self.cums[i])
                self.off_ts[i] = self.off_ts[i] + dt if fr["off_track"] else 0.0

                # all "doing well" rewards are GATED to on-track, so an agent earns
                # nothing for farming speed/centreline-progress out on the grass/edge
                # (the old exploit) — only the off-track penalty applies out there.
                head_err = (veh.yaw - fr["heading"] + np.pi) % (2 * np.pi) - np.pi
                ot = on_track_factor(fr, self.trk, self.rw.edge_band)
                p_term = d * self.rw.progress
                a_term = self.rw.align * np.cos(head_err) * (veh.speed / V_REF)
                s_term = self.rw.speed * (veh.speed / V_REF)
                drive = p_term + a_term + s_term
                # SEND IT: super-linear speed + WOT reward where the road ahead is
                # straight, so each agent floors the long straights at full tilt
                # instead of settling into a safe cruise (see ppo_env for the shape).
                sb, straight_gate, align_gate = straight_speed_bonus(
                    veh, fr, self.trk, self.rw, head_err)
                drive += sb
                th_term = self.rw.throttle_commit * straight_gate * align_gate * throttles[i]
                drive += th_term
                race_rewards[i] += drive * ot
                reward_parts[i]["progress"] += p_term * ot
                reward_parts[i]["align"] += a_term * ot
                reward_parts[i]["speed"] += s_term * ot
                reward_parts[i]["straight_speed"] += sb * ot
                reward_parts[i]["throttle_commit"] += th_term * ot
                diag_values[i].update(on_track_factor=ot, heading_error=head_err,
                                      straight_gate=straight_gate, align_gate=align_gate)
                if fr["off_track"]:
                    race_rewards[i] -= self.rw.offtrack
                    reward_parts[i]["offtrack"] -= self.rw.offtrack

                crawl = self.t > 1.0 and veh.speed < 0.8
                self.stall_ts[i] = self.stall_ts[i] + dt if crawl else 0.0
                if self.cums[i] > self.best_cums[i] + 1e-3:
                    self.best_cums[i] = self.cums[i]
                    self.since_progs[i] = 0.0
                else:
                    self.since_progs[i] += dt

                if self.off_ts[i] > 1.2:
                    terminated[i] = True
                    termination_reasons[i] = "offtrack_timeout"
                elif abs(fr["lateral"]) > self.trk.half + 4.0:
                    terminated[i] = True
                    termination_reasons[i] = "left_track"
                elif self.stall_ts[i] > 2.5:
                    terminated[i] = True
                    termination_reasons[i] = "stall"
                elif self.since_progs[i] > 4.0:
                    terminated[i] = True
                    termination_reasons[i] = "no_progress"
                elif self.cums[i] < self.max_cums[i] - 0.04:
                    terminated[i] = True
                    termination_reasons[i] = "backwards"

            for i, veh in enumerate(self.vehs):
                if terminated[i]:
                    continue
                clutch, up, down = self.boxes[i].update(veh, throttles[i], dt)
                veh.step(Controls(steer=steers[i], throttle=throttles[i], brake=brakes[i],
                                  clutch=clutch, handbrake=0.0,
                                  shift_up=up, shift_down=down))

            # Collisions (Any touch is penalized to enforce clean racing)
            for i in range(self.n_agents):
                for j in range(i + 1, self.n_agents):
                    colliding, _, _ = sat_collision(self.vehs[i].get_obb(), self.vehs[j].get_obb())
                    if colliding:
                        collision_penalties[i] += 0.5
                        collision_penalties[j] += 0.5

            pre_dmg = np.array([v.engine_damage + v.aero_damage for v in self.vehs])
            resolve_collisions(self.vehs, dt)
            post_dmg = np.array([v.engine_damage + v.aero_damage for v in self.vehs])
            dmg_diff = post_dmg - pre_dmg
            collision_penalties += np.where(dmg_diff > 0, 5.0 * dmg_diff, 0.0)
            for i in range(self.n_agents):
                reward_parts[i]["collision"] = -collision_penalties[i]

            self.t += dt
        # No forced joint termination: a crashed agent is skipped for the rest of
        # this control step (the `continue`s above) and respawns into its OWN fresh
        # episode below — so a reckless car can no longer cut a cautious car's run
        # short. Only the shared time limit (truncation) resets the whole grid.

        self.prev_ranks = self.ranks.copy()
        self._update_positions()
        # Overtake/draft bonuses go to LIVE cars only — a crashed agent's position
        # this step is meaningless (it's about to respawn), so don't credit it.
        overtake_rewards = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            if not terminated[i] and self.ranks[i] < self.prev_ranks[i]:
                overtake_rewards[i] = 2.0 * (self.prev_ranks[i] - self.ranks[i])
            reward_parts[i]["overtake"] = overtake_rewards[i]

        draft_rewards = np.zeros(self.n_agents)
        for i, veh in enumerate(self.vehs):
            if terminated[i]:
                continue
            for j, opp in enumerate(self.vehs):
                if i == j: continue
                dx = opp.x - veh.x
                dy = opp.y - veh.y
                dist = np.hypot(dx, dy)
                if 2.0 < dist < 12.0 and veh.speed > 10.0:
                    fwd_dot = dx * np.cos(veh.yaw) + dy * np.sin(veh.yaw)
                    if fwd_dot > 0:
                        draft_rewards[i] += 0.05 * (1.0 - dist / 12.0)
            reward_parts[i]["draft"] = draft_rewards[i]

        smooth_penalties = self.rw.smooth * np.sum((a - self.prev_actions) ** 2, axis=1)
        for i in range(self.n_agents):
            reward_parts[i]["smooth"] = -smooth_penalties[i]
        rewards = (race_rewards + overtake_rewards + draft_rewards
                   - collision_penalties - smooth_penalties)
        self.prev_actions = a
        rewards -= np.where(terminated, self.rw.crash, 0.0)
        for i in range(self.n_agents):
            if terminated[i]:
                reward_parts[i]["crash"] = -self.rw.crash

        # The time limit is the only SHARED episode boundary -> truncate the whole
        # grid together (one clean simultaneous reset upstream). Crashes are now
        # per-agent and self-heal via respawn, so they no longer truncate anyone.
        time_up = self.t >= self.ppo.episode_seconds
        truncated = np.full(self.n_agents, time_up)
        if time_up:
            termination_reasons = ["time_limit" if r is None else r for r in termination_reasons]
        self.step_count += 1

        # Build infos BEFORE respawning so a crashed agent reports the laps it
        # actually reached this episode (the respawn zeroes its progress trackers).
        infos = []
        for i in range(self.n_agents):
            info = {
                "laps": self.max_cums[i],
                "speed": self.vehs[i].speed,
                "t": self.t,
                "rank": self.ranks[i] + 1,
                "n_cars": self.n_agents,
                "overtakes": 0 if terminated[i] else max(0, self.prev_ranks[i] - self.ranks[i]),
                "engine_dmg": self.vehs[i].engine_damage,
                "aero_dmg": self.vehs[i].aero_damage,
            }
            if self.diagnostics:
                info["reward_parts"] = {k: float(v) for k, v in reward_parts[i].items() if abs(float(v)) > 1e-12}
                info["diagnostics"] = {k: float(v) for k, v in diag_values[i].items()}
                info["termination_reason"] = termination_reasons[i]
            infos.append(info)

        # Decoupled respawn: each crashed agent restarts its OWN episode at racing
        # pace in the clearest spot on track; everyone else keeps going untouched.
        # Skipped when the grid is about to reset on the time limit anyway.
        if not time_up and np.any(terminated):
            for i in range(self.n_agents):
                if terminated[i]:
                    self._respawn(i)
            # refresh ranks so next step's overtake delta is vs the post-respawn grid
            self._update_positions()

        return self._obs(), rewards, terminated, truncated, infos


# --------------------------------------------------------------------------- #
class SyncMultiVecEnv:
    """Wraps multiple MultiAgentRaceEnvs and flattens their array outputs, tricking
    PPO into seeing n_envs * n_agents standard environments."""
    def __init__(self, n, n_agents=4, env_cls=MultiAgentRaceEnv, **kw):
        self.n_envs = n
        self.n_agents = n_agents
        self.n = n * n_agents
        self.envs = [env_cls(n_agents=n_agents, rng_seed=i, **kw) for i in range(n)]
        self.obs_dim = self.envs[0].obs_dim
        self.act_dim = self.envs[0].action_dim
        self.sensor_dim = self.envs[0].sensor_dim

    def reset(self):
        obs = np.stack([e.reset() for e in self.envs])
        return obs.reshape(self.n, self.obs_dim)

    def step(self, actions):
        actions = actions.reshape(self.n_envs, self.n_agents, self.act_dim)
        nobs, rew, term, trunc, final, infos = [], [], [], [], [], []
        for e, a in zip(self.envs, actions):
            o, r, te, tr, info = e.step(a)
            final.append(o)
            # crashes self-heal via per-agent respawn inside step(); only the shared
            # time-limit truncation resets the whole grid.
            if np.any(tr):
                o = e.reset()
            nobs.append(o)
            rew.append(r)
            term.append(te)
            trunc.append(tr)
            infos.extend(info)
        return (np.concatenate(nobs), np.concatenate(rew).astype(np.float32),
                np.concatenate(term).astype(bool), np.concatenate(trunc).astype(bool),
                np.concatenate(final).reshape(self.n, self.obs_dim), infos)

    def set_difficulty(self, d):
        for e in self.envs:
            e.difficulty = d

    def set_fixed_track(self, track):
        for e in self.envs:
            e.fixed_track = track

    def close(self):
        pass


def _multi_vec_worker(remote, parent_remote, kw, seeds, env_cls, n_agents):
    parent_remote.close()
    try:
        envs = [env_cls(n_agents=n_agents, rng_seed=s, **kw) for s in seeds]
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                out = []
                for e, a in zip(envs, data):
                    o, r, te, tr, info = e.step(a)
                    final = o
                    # crashes self-heal via respawn inside step(); only the shared
                    # time-limit truncation resets the whole grid.
                    if np.any(tr):
                        o = e.reset()
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
    finally:
        try:
            remote.close()
        except Exception:
            pass


class SubprocMultiVecEnv:
    """Subprocessed version of SyncMultiVecEnv."""
    def __init__(self, n, n_workers, n_agents=4, env_cls=MultiAgentRaceEnv, **kw):
        import multiprocessing as mp
        if n % n_workers != 0:
            raise ValueError(f"n_envs ({n}) must be divisible by n_workers ({n_workers})")
        ctx = mp.get_context("spawn")
        self.n_envs = n
        self.n_agents = n_agents
        self.n = n * n_agents
        self.s = n // n_workers
        self.remotes, self.procs = [], []
        for w in range(n_workers):
            seeds = list(range(w * self.s, (w + 1) * self.s))
            r, wr = ctx.Pipe()
            p = ctx.Process(target=_multi_vec_worker, args=(wr, r, kw, seeds, env_cls, n_agents), daemon=True)
            p.start()
            wr.close()
            self.remotes.append(r)
            self.procs.append(p)
        probe = env_cls(n_agents=n_agents, rng_seed=0, **kw)
        self.obs_dim, self.act_dim = probe.obs_dim, probe.action_dim
        self.sensor_dim = probe.sensor_dim
        del probe

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        obs = []
        for r in self.remotes:
            obs.extend(r.recv())
        return np.stack(obs).reshape(self.n, self.obs_dim)

    def step(self, actions):
        actions = actions.reshape(self.n_envs, self.n_agents, self.act_dim)
        s = self.s
        for i, r in enumerate(self.remotes):
            r.send(("step", actions[i * s:(i + 1) * s]))
        nobs, rew, term, trunc, final, infos = [], [], [], [], [], []
        for r in self.remotes:
            for (o, rr, te, tr, fo, info) in r.recv():
                nobs.append(o); rew.append(rr); term.append(te)
                trunc.append(tr); final.append(fo); infos.extend(info)
        return (np.concatenate(nobs), np.concatenate(rew).astype(np.float32),
                np.concatenate(term).astype(bool), np.concatenate(trunc).astype(bool),
                np.concatenate(final).reshape(self.n, self.obs_dim), infos)

    def set_difficulty(self, d):
        for r in self.remotes:
            r.send(("set_difficulty", d))
        for r in self.remotes:
            r.recv()

    def set_fixed_track(self, track):
        for r in self.remotes:
            r.send(("set_fixed_track", track))
        for r in self.remotes:
            r.recv()

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
