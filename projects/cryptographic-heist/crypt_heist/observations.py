"""Agent observation construction for training and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .comms import VOCABULARY
from .policies import world_velocity

AGENT_NAMES = ["evader_0"] + [f"pursuer_{i}" for i in range(5)]
NAME_TO_SIM = {"evader_0": "EVADER"} | {f"pursuer_{i}": f"P{i + 1}" for i in range(5)}
SIM_TO_NAME = {v: k for k, v in NAME_TO_SIM.items()}

RAY_ANGLES = np.radians(np.linspace(105.0, -105.0, 9))
MAX_CITY_EXTENT = 900.0
MAX_REL_DIST = 420.0
MAX_SPEED = 80.0
MAX_RPM = 8500.0
COMM_WINDOW = 16


@dataclass(frozen=True)
class ObservationSpec:
    ego_dim: int = 12
    ray_dim: int = 9
    waypoint_dim: int = 3
    other_dim: int = 30
    comm_dim: int = COMM_WINDOW
    status_dim: int = 9

    @property
    def size(self) -> int:
        return self.ego_dim + self.ray_dim + self.waypoint_dim + self.other_dim + self.comm_dim + self.status_dim


OBS_SPEC = ObservationSpec()
OBS_SIZE = OBS_SPEC.size


def build_observation(sim, agent_name: str) -> np.ndarray:
    """Return a fixed-size normalized observation for one training agent."""
    agent = _agent_by_training_name(sim, agent_name)
    veh = agent.vehicle
    city = sim.city
    evader = sim.evader.vehicle

    ego_v = world_velocity(veh)
    ego = np.array([
        veh.x / MAX_CITY_EXTENT,
        veh.y / MAX_CITY_EXTENT,
        veh.vx / MAX_SPEED,
        veh.vy / MAX_SPEED,
        veh.speed / MAX_SPEED,
        np.sin(veh.yaw),
        np.cos(veh.yaw),
        veh.r / 4.0,
        veh.slip_angle / (np.pi / 2.0),
        veh.steer_angle / max(veh.max_steer_angle, 1e-6),
        veh.rpm / MAX_RPM,
        (veh.gear - 1) / max(1, len(veh.spec.gear_ratios) - 1),
    ], dtype=np.float32)

    rays = city.raycast_buildings(veh.x, veh.y, veh.yaw, RAY_ANGLES, max_range=90.0) / 90.0
    wx, wy = sim.evader_planner.waypoint
    wdx = wx - veh.x
    wdy = wy - veh.y
    forward_wp = wdx * np.cos(veh.yaw) + wdy * np.sin(veh.yaw)
    lateral_wp = -wdx * np.sin(veh.yaw) + wdy * np.cos(veh.yaw)
    waypoint = np.array([
        np.clip(forward_wp / MAX_REL_DIST, -1.0, 1.0),
        np.clip(lateral_wp / MAX_REL_DIST, -1.0, 1.0),
        np.clip(np.hypot(wdx, wdy) / MAX_REL_DIST, 0.0, 1.5),
    ], dtype=np.float32)

    others = []
    for other in sim.agents:
        if other.name == agent.name:
            continue
        ov = world_velocity(other.vehicle)
        dx = other.vehicle.x - veh.x
        dy = other.vehicle.y - veh.y
        others.extend([
            np.clip(dx / MAX_REL_DIST, -1.5, 1.5),
            np.clip(dy / MAX_REL_DIST, -1.5, 1.5),
            np.clip(np.hypot(dx, dy) / MAX_REL_DIST, 0.0, 1.5),
            np.clip((ov[0] - ego_v[0]) / MAX_SPEED, -1.5, 1.5),
            np.clip((ov[1] - ego_v[1]) / MAX_SPEED, -1.5, 1.5),
            1.0 if other.faction == "evader" else -1.0,
        ])
    others = np.asarray(others[: OBS_SPEC.other_dim], dtype=np.float32)
    if len(others) < OBS_SPEC.other_dim:
        others = np.pad(others, (0, OBS_SPEC.other_dim - len(others)))

    vocab_size = max(1, len(VOCABULARY) - 1)
    recent = sim.channel.recent(COMM_WINDOW)
    token_values = [VOCABULARY.index(e.word) / vocab_size if e.word in VOCABULARY else 0.0 for e in recent]
    if len(token_values) < COMM_WINDOW:
        token_values = [0.0] * (COMM_WINDOW - len(token_values)) + token_values
    comms = np.asarray(token_values[-COMM_WINDOW:], dtype=np.float32)

    role_idx = getattr(sim.coordinator, "roles", ()).index(agent.role) if agent.role in sim.coordinator.roles else 0
    nearest = min(np.hypot(p.vehicle.x - evader.x, p.vehicle.y - evader.y) for p in sim.pursuers)
    status = np.array([
        sim.channel.confidence,
        sim.channel.jamming_budget / 7.0,
        np.clip(sim.jam_cooldown / 8.0, 0.0, 1.5),
        np.clip(sim.capture_timer / 1.2, 0.0, 2.0),
        np.clip(nearest / MAX_REL_DIST, 0.0, 1.5),
        np.clip(role_idx / 4.0, 0.0, 1.0),
        np.clip(agent.spoof_timer / 2.5, 0.0, 1.5),
        1.0 if agent.faction == "evader" else -1.0,
        np.sin(sim.time * 0.1),
    ], dtype=np.float32)

    obs = np.concatenate([ego, rays.astype(np.float32), waypoint, others, comms, status]).astype(np.float32)
    if obs.shape != (OBS_SIZE,):
        raise RuntimeError(f"bad observation shape {obs.shape}, expected {(OBS_SIZE,)}")
    return np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)


def _agent_by_training_name(sim, agent_name: str):
    sim_name = NAME_TO_SIM.get(agent_name, agent_name)
    for agent in sim.agents:
        if agent.name == sim_name:
            return agent
    raise KeyError(agent_name)
