"""Core simulation loop for The Cryptographic Heist Engine."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .autobox import AutoBox
from .city import UrbanGrid
from .comms import CommsChannel
from .config import SimSpec, get_car
from .physics import Controls, Vehicle
from .policies import EvaderPlanner, PursuerCoordinator, drive_toward, world_velocity
from .rewards import RewardSnapshot, RewardSystem


@dataclass
class AgentState:
    name: str
    faction: str
    vehicle: Vehicle
    gearbox: AutoBox
    role: str = ""
    target: tuple[float, float] = (0.0, 0.0)
    spoof_timer: float = 0.0
    spoof_offset: np.ndarray = field(default_factory=lambda: np.zeros(2))
    last_control: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


class HeistSim:
    """MARL-ready chase state with scripted policies for the first build."""

    def __init__(self, seed: int | None = 11, reset_on_capture: bool = True):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.reset_on_capture = reset_on_capture
        self.sim = SimSpec(dt=1.0 / 120.0, fps=120)
        self.city = UrbanGrid(seed=seed)
        self.channel = CommsChannel(self.rng)
        self.coordinator = PursuerCoordinator()
        self.evader_planner = EvaderPlanner(self.city)
        self.evader_spec = get_car("evader")
        self.pursuer_spec = get_car("pursuer")
        self.evader = AgentState(
            "EVADER",
            "evader",
            Vehicle(self.evader_spec, self.sim),
            AutoBox(self.evader_spec),
        )
        self.pursuers = [
            AgentState(f"P{i + 1}", "pursuer", Vehicle(self.pursuer_spec, self.sim), AutoBox(self.pursuer_spec))
            for i in range(5)
        ]
        self.time = 0.0
        self.broadcast_timer = 0.0
        self.broadcast_index = 0
        self.jam_cooldown = 4.5
        self.capture_timer = 0.0
        self.captures = 0
        self.waypoints_hit = 0
        self.deception_score = 0.0
        self.impact_energy = 0.0
        self.decoder_predictions: dict[str, tuple[float, float]] = {}
        self.episode_done = False
        self.capture_event = False
        self.waypoint_event = False
        self.last_collision_impact = 0.0
        self.last_action_request: dict[str, dict] = {}
        self.reward_system = RewardSystem()
        self.last_rewards: dict[str, float] = {}
        self.last_reward_components: dict[str, dict] = {}
        self.reset()

    @property
    def agents(self) -> list[AgentState]:
        return [self.evader] + self.pursuers

    def reset(self, *, preserve_time: bool = False):
        current_time = self.time if preserve_time else 0.0
        self.time = current_time
        self.broadcast_timer = 0.0
        self.broadcast_index = 0
        self.jam_cooldown = 4.5
        self.capture_timer = 0.0
        self.impact_energy = 0.0
        self.episode_done = False
        self.capture_event = False
        self.waypoint_event = False
        self.last_collision_impact = 0.0
        self.last_action_request = {}
        self.reward_system.reset()
        self.last_rewards = {}
        self.last_reward_components = {}
        self.decoder_predictions = {}
        self.channel = CommsChannel(self.rng)

        ex, ey = self.city.random_road_point()
        self.evader.vehicle.reset(ex, ey, yaw=0.0, speed=18.0)
        self.evader.gearbox = AutoBox(self.evader_spec)
        self.evader_planner.reset()
        self.evader_planner.choose_waypoint(self.evader.vehicle, [])
        wx, wy = self.evader_planner.waypoint
        self.evader.vehicle.yaw = float(np.arctan2(wy - ey, wx - ex))

        starts = [
            (ex - 115, ey - 52),
            (ex - 105, ey + 46),
            (ex + 96, ey - 68),
            (ex + 110, ey + 62),
            (ex - 170, ey),
        ]
        for i, agent in enumerate(self.pursuers):
            px, py = self.city.snap_to_road(*starts[i])
            yaw = float(np.arctan2(ey - py, ex - px))
            agent.vehicle.reset(px, py, yaw=yaw, speed=12.0)
            agent.gearbox = AutoBox(self.pursuer_spec)
            agent.role = self.coordinator.roles[i]
            agent.spoof_timer = 0.0
            agent.spoof_offset[:] = 0.0
            agent.last_control = (0.0, 0.0, 0.0, 0.0)
        self.evader.last_control = (0.0, 0.0, 0.0, 0.0)

    def step(self, dt: float | None = None, actions: dict | None = None):
        if self.episode_done:
            return self.snapshot()
        dt = self.sim.dt if dt is None else float(dt)
        self.last_action_request = self._sanitize_action_request(actions)
        reward_before = RewardSnapshot.from_sim(self)
        self.time += dt
        self.broadcast_timer -= dt
        self.jam_cooldown -= dt
        self.capture_event = False
        self.waypoint_event = False
        self.last_collision_impact = 0.0
        self.channel.tick(dt, self.time)

        self._maybe_broadcast(actions)
        self._maybe_jam(actions)
        self._update_decoder(dt)
        self._drive_agents(dt, actions)
        self._resolve_vehicle_contacts()
        self._update_capture(dt)
        self._update_reward_trace(reward_before)
        return self.snapshot()

    def _maybe_broadcast(self, actions: dict | None):
        if self.broadcast_timer > 0.0:
            return
        self.broadcast_timer = 0.45
        assignments = self._assignments()
        for agent, assignment in zip(self.pursuers, assignments):
            agent.role = assignment.role
            agent.target = assignment.target
        i = self.broadcast_index % len(self.pursuers)
        assignment = assignments[i]
        action = self._action_for(actions, f"pursuer_{i}")
        token_ids = self._tokens_from_action(action, "tokens")
        if token_ids is None:
            self.channel.broadcast_pursuer(self.time, i, assignment.role, assignment.target)
        else:
            self.channel.broadcast_tokens(self.time, f"P{i + 1}", token_ids, False, assignment.role)
        self.broadcast_index += 1

    def _maybe_jam(self, actions: dict | None):
        ev = self.evader.vehicle
        close = min(np.hypot(p.vehicle.x - ev.x, p.vehicle.y - ev.y) for p in self.pursuers)
        pressure = sum(np.hypot(p.vehicle.x - ev.x, p.vehicle.y - ev.y) < 95.0 for p in self.pursuers)
        ev_action = self._action_for(actions, "evader_0")
        forced_jam = self._jam_from_action(ev_action)
        should_jam = self.jam_cooldown <= 0.0 and ((pressure >= 2 and close < 85.0) or forced_jam)
        if not should_jam:
            return
        payload = self._tokens_from_action(ev_action, "spoof_tokens")
        if self.channel.inject_jam(self.time, payload=payload):
            self.jam_cooldown = float(self.rng.uniform(5.0, 8.0))
            victims = self._victims_from_action(ev_action)
            if not victims:
                victims = list(self.rng.choice(len(self.pursuers), size=2, replace=False))
            for idx in victims:
                p = self.pursuers[int(idx)]
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                mag = self.rng.uniform(45.0, 85.0)
                p.spoof_offset = np.array([np.cos(angle), np.sin(angle)]) * mag
                p.spoof_timer = 2.5
            self.deception_score += 1.0

    def _assignments(self):
        for p in self.pursuers:
            if p.spoof_timer > 0.0:
                p.spoof_timer = max(0.0, p.spoof_timer - self.sim.dt)
            else:
                p.spoof_offset *= 0.86
        return self.coordinator.assignments(
            self.evader.vehicle,
            [p.vehicle for p in self.pursuers],
            self.city,
            [p.spoof_offset for p in self.pursuers],
        )

    def _update_decoder(self, dt: float):
        conf = self.channel.confidence
        noise = (1.0 - conf) * 34.0
        self.decoder_predictions = {}
        for p in self.pursuers:
            v = world_velocity(p.vehicle)
            pred = np.array([p.vehicle.x, p.vehicle.y]) + v * 0.85
            pred += self.rng.normal(0.0, noise, size=2)
            self.decoder_predictions[p.name] = (float(pred[0]), float(pred[1]))
        self.channel.confidence = float(np.clip(self.channel.confidence + dt * 0.012, 0.04, 0.98))

    def _drive_agents(self, dt: float, actions: dict | None):
        ev = self.evader.vehicle
        ev_ctrl = self._control_from_action(self._action_for(actions, "evader_0"))
        if ev_ctrl is None:
            ev_ctrl = self.evader_planner.control(ev, [p.vehicle for p in self.pursuers])
        self._apply_control(self.evader, ev_ctrl, dt)

        assignments = self._assignments()
        for i, (agent, assignment) in enumerate(zip(self.pursuers, assignments)):
            agent.role = assignment.role
            agent.target = assignment.target
            ctrl = self._control_from_action(self._action_for(actions, f"pursuer_{i}"))
            if ctrl is None:
                ctrl = drive_toward(agent.vehicle, assignment.target, desired_speed=29.0, handbrake_turns=False)
            self._apply_control(agent, ctrl, dt)

    def _apply_control(self, agent: AgentState, ctrl: tuple[float, float, float, float], dt: float):
        veh = agent.vehicle
        steer, throttle, brake, handbrake = ctrl
        agent.last_control = ctrl
        clutch, up, down = agent.gearbox.update(veh, throttle, dt)
        veh.surface_grip = 1.0 if self.city.is_road(veh.x, veh.y, pad=1.0) else veh.spec.offtrack_grip
        veh.set_road(0.0, 0.0, 0.0, 0.0, 0.0)
        veh.step(Controls(
            steer=steer,
            throttle=throttle,
            brake=brake,
            clutch=clutch,
            handbrake=handbrake,
            shift_up=up,
            shift_down=down,
        ), dt)
        impact = self.city.resolve_vehicle(veh)
        self.last_collision_impact = max(self.last_collision_impact, impact)
        self.impact_energy = max(self.impact_energy * 0.96, impact)

    def _resolve_vehicle_contacts(self):
        agents = self.agents
        radius = {"evader": 2.25, "pursuer": 2.65}
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                va, vb = a.vehicle, b.vehicle
                dx, dy = vb.x - va.x, vb.y - va.y
                dist = float(np.hypot(dx, dy))
                min_dist = radius[a.faction] + radius[b.faction]
                if dist <= 1e-6 or dist >= min_dist:
                    continue
                n = np.array([dx / dist, dy / dist])
                penetration = min_dist - dist
                va.x -= float(n[0] * penetration * 0.5)
                va.y -= float(n[1] * penetration * 0.5)
                vb.x += float(n[0] * penetration * 0.5)
                vb.y += float(n[1] * penetration * 0.5)

                wa = world_velocity(va)
                wb = world_velocity(vb)
                rel = float(np.dot(wb - wa, n))
                if rel < 0.0:
                    ma, mb = va.spec.mass, vb.spec.mass
                    impulse = -(1.0 + 0.18) * rel / (1.0 / ma + 1.0 / mb)
                    wa -= (impulse / ma) * n
                    wb += (impulse / mb) * n
                    self._set_world_velocity(va, wa * 0.96)
                    self._set_world_velocity(vb, wb * 0.96)
                    impact = abs(rel)
                    self.last_collision_impact = max(self.last_collision_impact, impact)
                    self.impact_energy = max(self.impact_energy, impact)

    def _update_capture(self, dt: float):
        ev = self.evader.vehicle
        dists = [np.hypot(p.vehicle.x - ev.x, p.vehicle.y - ev.y) for p in self.pursuers]
        boxed = sum(d < 22.0 for d in dists) >= 3
        pinned = min(dists) < 11.0 and ev.speed < 6.5
        self.capture_timer = self.capture_timer + dt if boxed or pinned else max(0.0, self.capture_timer - dt * 1.8)
        if self.capture_timer > 1.2:
            self.captures += 1
            self.capture_event = True
            self.episode_done = True
            if self.reset_on_capture:
                self.reset(preserve_time=True)
            return
        if np.hypot(ev.x - self.evader_planner.waypoint[0], ev.y - self.evader_planner.waypoint[1]) < 18.0:
            self.waypoints_hit += 1
            self.waypoint_event = True
            self.evader_planner.choose_waypoint(ev, [p.vehicle for p in self.pursuers])

    @staticmethod
    def _set_world_velocity(veh: Vehicle, vw: np.ndarray):
        cy, sy = np.cos(veh.yaw), np.sin(veh.yaw)
        veh.vx = float(vw[0] * cy + vw[1] * sy)
        veh.vy = float(-vw[0] * sy + vw[1] * cy)

    @staticmethod
    def _action_for(actions: dict | None, name: str):
        if not actions:
            return None
        return actions.get(name)

    @staticmethod
    def _control_from_action(action) -> tuple[float, float, float, float] | None:
        if action is None:
            return None
        control = action.get("control") if isinstance(action, dict) else action
        if control is None:
            return None
        arr = np.asarray(control, dtype=float).reshape(-1)
        if len(arr) < 3:
            return None
        steer = float(np.clip(arr[0], -1.0, 1.0))
        long = float(np.clip(arr[1], -1.0, 1.0))
        handbrake = float(np.clip((arr[2] + 1.0) * 0.5, 0.0, 1.0))
        throttle = max(0.0, long)
        brake = max(0.0, -long)
        return steer, throttle, brake, handbrake

    @staticmethod
    def _jam_from_action(action) -> bool:
        if action is None or not isinstance(action, dict):
            return False
        return bool(int(action.get("jam", 0)))

    @staticmethod
    def _tokens_from_action(action, key: str):
        if action is None or not isinstance(action, dict) or key not in action:
            return None
        return [int(x) for x in np.asarray(action[key]).reshape(-1)]

    @staticmethod
    def _victims_from_action(action) -> list[int]:
        if action is None or not isinstance(action, dict) or "target_mask" not in action:
            return []
        mask = np.asarray(action["target_mask"]).reshape(-1)
        return [i for i, value in enumerate(mask[:5]) if value > 0]

    def snapshot(self) -> dict:
        return {
            "time": self.time,
            "agents": self.agents,
            "actions": self._action_snapshot(),
            "waypoint": self.evader_planner.waypoint,
            "confidence": self.channel.confidence,
            "events": self.channel.recent(120),
            "predictions": self.decoder_predictions,
            "collisions": self._collision_snapshot(),
            "rewards": self.last_rewards,
            "reward_components": self.last_reward_components,
            "jamming_budget": self.channel.jamming_budget,
            "last_jam_time": self.channel.last_jam_time,
            "last_spoof_words": self.channel.last_spoof_words,
            "captures": self.captures,
            "waypoints_hit": self.waypoints_hit,
            "deception_score": self.deception_score,
            "impact": self.impact_energy,
            "capture_event": self.capture_event,
            "waypoint_event": self.waypoint_event,
            "episode_done": self.episode_done,
            "last_collision_impact": self.last_collision_impact,
        }

    def _sanitize_action_request(self, actions: dict | None) -> dict[str, dict]:
        if not actions:
            return {}
        rows: dict[str, dict] = {}
        mapping = [("evader_0", "EVADER")] + [(f"pursuer_{i}", f"P{i + 1}") for i in range(5)]
        for env_name, replay_name in mapping:
            action = self._action_for(actions, env_name)
            if action is None:
                continue
            row: dict[str, object] = {}
            control = action.get("control") if isinstance(action, dict) else action
            if control is not None:
                arr = np.asarray(control, dtype=float).reshape(-1)
                row["requested_control"] = [float(np.clip(x, -1.0, 1.0)) for x in arr[:3]]
            if isinstance(action, dict):
                tokens = self._tokens_from_action(action, "tokens")
                if tokens is not None:
                    row["tokens"] = tokens[:6]
                spoof_tokens = self._tokens_from_action(action, "spoof_tokens")
                if spoof_tokens is not None:
                    row["spoof_tokens"] = spoof_tokens[:5]
                if "target_mask" in action:
                    mask = np.asarray(action["target_mask"]).reshape(-1)
                    row["target_mask"] = [int(x > 0) for x in mask[:5]]
                if "jam" in action:
                    row["jam"] = bool(int(action.get("jam", 0)))
            rows[replay_name] = row
        return rows

    def _action_snapshot(self) -> dict[str, dict]:
        rows = {}
        for agent in self.agents:
            request = dict(self.last_action_request.get(agent.name, {}))
            steer, throttle, brake, handbrake = agent.last_control
            request["control"] = [float(steer), float(throttle), float(brake), float(handbrake)]
            request["source"] = "external" if agent.name in self.last_action_request else "scripted"
            rows[agent.name] = request
        return rows

    def _collision_snapshot(self) -> list[dict]:
        impact = max(float(self.last_collision_impact), float(self.impact_energy))
        if impact <= 0.0:
            return []
        return [{
            "time": float(self.time),
            "kind": "impact",
            "impact": impact,
            "last_collision_impact": float(self.last_collision_impact),
            "camera_shake": float(min(8.0, impact * 0.32)),
        }]

    def _update_reward_trace(self, before: RewardSnapshot) -> None:
        after = RewardSnapshot.from_sim(self)
        rewards, components = self.reward_system.compute(before, after, self.episode_done)
        self.last_rewards = {
            self._replay_agent_name(name): float(value)
            for name, value in rewards.items()
        }
        self.last_reward_components = {
            self._replay_agent_name(name): self._agent_reward_components(name, rewards, components)
            for name in rewards
        }

    @staticmethod
    def _replay_agent_name(name: str) -> str:
        if name == "evader_0":
            return "EVADER"
        if name.startswith("pursuer_"):
            return f"P{int(name.split('_', 1)[1]) + 1}"
        return name

    @staticmethod
    def _agent_reward_components(name: str, rewards: dict[str, float], components: dict) -> dict[str, float | str]:
        is_evader = name == "evader_0"
        reward = float(rewards.get(name, 0.0))
        row: dict[str, float | str] = {
            "role": "evader" if is_evader else "pursuer",
            "reward": reward,
            "waypoint_progress": float(components.get("waypoint_progress", 0.0)),
            "pursuit_progress": float(components.get("pursuit_progress", 0.0)),
            "waypoint_delta": float(components.get("waypoint_delta", 0.0)),
            "capture_delta": float(components.get("capture_delta", 0.0)),
            "deception_delta": float(components.get("deception_delta", 0.0)),
            "impact_penalty": float(components.get("impact_penalty", 0.0)),
            "confidence": float(components.get("confidence", 0.0)),
            "decoder_error": float(components.get("decoder_error", 0.0)),
            "decoder_accuracy": float(components.get("decoder_accuracy", 0.0)),
            "decoder_accuracy_samples": float(components.get("decoder_accuracy_samples", 0.0)),
            "live_decoder_error": float(components.get("live_decoder_error", 0.0)),
            "spoof_susceptibility": float(components.get("spoof_susceptibility", 0.0)),
            "active_spoofed_pursuers": float(components.get("active_spoofed_pursuers", 0.0)),
            "counterfactual_deception": float(components.get("counterfactual_deception", 0.0)),
        }
        if is_evader:
            row.update({
                "information_reward": float(components.get("evader_information_reward", 0.0)),
                "decoder_accuracy_penalty": 0.0,
                "spoof_susceptibility_penalty": 0.0,
                "auth_penalty": 0.0,
            })
        else:
            row.update({
                "information_reward": -float(components.get("pursuer_auth_penalty", 0.0)),
                "decoder_accuracy_penalty": float(components.get("decoder_accuracy_penalty", 0.0)),
                "spoof_susceptibility_penalty": float(components.get("spoof_susceptibility_penalty", 0.0)),
                "auth_penalty": float(components.get("pursuer_auth_penalty", 0.0)),
            })
        return row
