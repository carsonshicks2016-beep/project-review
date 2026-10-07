"""
PettingZoo-Compatible Multi-Agent Traffic Signal Control Environment.

Implements a ``pettingzoo.ParallelEnv``-style interface for multi-agent
reinforcement learning on the Norman, Oklahoma game-day traffic network.

Seven signal control agents at critical intersections simultaneously
adjust green-split allocations and phase hold/transition decisions.
The environment wraps SteppableTrafficEngine and exposes standard
Gymnasium observation/action spaces.

Compatible with:
    - PettingZoo >= 1.24 (ParallelEnv API)
    - Gymnasium >= 0.29 (Box, MultiBinary spaces)
    - StableBaselines3, RLlib, CleanRL, TorchRL via standard wrappers

Usage::

    from soonergrid.gym import NormanTrafficEnv

    env = NormanTrafficEnv()
    obs = env.reset()
    for _ in range(1000):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        if all(terms.values()) or all(truncs.values()):
            obs = env.reset()

Paper Reference:
    Observation space design follows Chu et al. (2019) "Multi-Agent Deep
    Reinforcement Learning for Large-Scale Traffic Signal Control" with
    extensions for neighbor communication vectors and game-day temporal
    encoding.
"""

from __future__ import annotations

import functools
import math
from typing import Any, Dict, List, Optional, Tuple

from soonergrid.gym.steppable_engine import (
    SteppableTrafficEngine,
    SIGNAL_AGENT_DEFS,
    OBS_DIM,
    ACTION_DIM,
)
from soonergrid.gym.spaces import make_observation_space, make_action_space


class NormanTrafficEnv:
    """
    PettingZoo ParallelEnv-compatible multi-agent traffic signal environment.

    This environment does NOT inherit from ``pettingzoo.ParallelEnv`` directly
    to avoid a hard dependency on the pettingzoo package. However, it implements
    the complete ParallelEnv API contract so that PettingZoo's ``parallel_to_aec``
    wrapper and compatibility checkers work out of the box.

    Agents:
        7 signal controllers at critical Norman intersections:
        - sig_i35_lindsey: I-35 & W Lindsey St SPUI
        - sig_lindsey_mcgee: W Lindsey St & McGee Dr
        - sig_lindsey_berry: W Lindsey St & Berry Rd (Major Chokepoint)
        - sig_lindsey_chautauqua: W Lindsey St & Chautauqua Ave
        - sig_lindsey_jenkins: W Lindsey St & S Jenkins Ave (Stadium SE Corner)
        - sig_classen_lindsey: Classen Blvd & E Lindsey St
        - sig_sh9_jenkins: SH-9 & S Jenkins Ave (Lloyd Noble Center)

    Observation Space (per agent):
        Box(low=0, high=1, shape=(36,), dtype=float32)
        See SteppableTrafficEngine._build_observations() for detailed layout.

    Action Space (per agent):
        Box(low=[-1, 0], high=[1, 1], shape=(2,), dtype=float32)
        - action[0]: Green split delta [-1,1] → [-15s, +15s] arterial green adjustment
        - action[1]: Phase hold probability [0,1] (>0.5 holds current phase)

    Reward:
        R_i = -α·queue² - δ·delay - β·spillback + γ·throughput
        (configurable via reward_* kwargs)

    Episode:
        One full game day: 8 hours (28800s) at dt=5s → 5760 steps.
        Configurable via episode_duration_s and dt_s.
    """

    # PettingZoo metadata
    metadata = {
        "name": "norman_traffic_v1",
        "render_modes": [],
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
        self,
        graph=None,
        gateways=None,
        sinks=None,
        demand_dataset=None,
        dt_s: float = 5.0,
        episode_duration_s: float = 28800.0,
        cav_penetration: float = 0.0,
        driver_compliance: float = 0.60,
        incident_shock: Optional[str] = None,
        enable_self_healing: bool = False,
        reward_queue_weight: float = 1.0,
        reward_delay_weight: float = 0.5,
        reward_spillback_weight: float = 5.0,
        reward_throughput_weight: float = 0.1,
        auto_load_network: bool = True,
    ):
        """
        Initialize the Norman Traffic Environment.

        If graph/gateways/sinks/demand_dataset are None and auto_load_network
        is True, the environment automatically loads the processed Norman network
        and game-day demand from data files (same loader as benchmark.py).

        Args:
            graph: Optional NetworkX MultiDiGraph. Auto-loaded if None.
            gateways: Optional list of Gateway objects. Auto-loaded if None.
            sinks: Optional list of ParkingSink objects. Auto-loaded if None.
            demand_dataset: Optional demand dict. Auto-loaded if None.
            dt_s: Simulation timestep in seconds.
            episode_duration_s: Episode length in seconds.
            cav_penetration: CAV fleet fraction [0, 1].
            driver_compliance: Human driver compliance [0, 1].
            incident_shock: Incident scenario string or None.
            enable_self_healing: Enable self-healing coordinator.
            reward_queue_weight: α in reward function.
            reward_delay_weight: δ in reward function.
            reward_spillback_weight: β in reward function.
            reward_throughput_weight: γ in reward function.
            auto_load_network: If True, load network from data files when not provided.
        """
        # Load network if not provided
        if graph is None and auto_load_network:
            graph, gateways, sinks, demand_dataset = self._auto_load()

        if graph is None:
            raise ValueError(
                "No graph provided and auto_load_network is False. "
                "Provide graph, gateways, sinks, and demand_dataset."
            )

        self._dt_s = dt_s
        self._episode_duration_s = episode_duration_s

        # Store construction args for engine recreation on reset
        self._engine_kwargs = dict(
            graph=graph,
            gateways=gateways,
            sinks=sinks,
            demand_dataset=demand_dataset,
            dt_s=dt_s,
            episode_duration_s=episode_duration_s,
            cav_penetration=cav_penetration,
            driver_compliance=driver_compliance,
            incident_shock=incident_shock,
            enable_self_healing=enable_self_healing,
            reward_queue_weight=reward_queue_weight,
            reward_delay_weight=reward_delay_weight,
            reward_spillback_weight=reward_spillback_weight,
            reward_throughput_weight=reward_throughput_weight,
        )

        # Create the simulation engine
        self._engine = SteppableTrafficEngine(**self._engine_kwargs)

        # PettingZoo agent management
        self.possible_agents: List[str] = list(self._engine.agent_ids)
        self.agents: List[str] = list(self.possible_agents)

        # Spaces are created lazily via observation_space() and action_space()
        self._obs_spaces = {
            agent: make_observation_space(OBS_DIM) for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: make_action_space(ACTION_DIM) for agent in self.possible_agents
        }

        # Episode tracking
        self._current_step = 0
        self._max_steps = round(episode_duration_s / dt_s)
        self._episode_count = 0

    @staticmethod
    def _auto_load():
        """Load the Norman network and demand from processed data files."""
        from soonergrid.research.benchmark import load_inputs
        network, demand, graph, gateways, sinks = load_inputs()
        return graph, gateways, sinks, demand

    # ── PettingZoo ParallelEnv API ──────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Returns:
            observations: Dict[agent_id -> observation array]
            infos: Dict[agent_id -> info dict]
        """
        self.agents = list(self.possible_agents)
        self._current_step = 0
        self._episode_count += 1

        obs_dict = self._engine.reset(seed=seed)

        # Convert to arrays compatible with spaces
        observations = {}
        infos: Dict[str, Any] = {}
        for agent in self.agents:
            observations[agent] = obs_dict.get(agent, [0.0] * OBS_DIM)
            infos[agent] = {"step": 0, "time_s": 0.0}

        return observations, infos

    def step(
        self, actions: Dict[str, Any]
    ) -> Tuple[
        Dict[str, Any],      # observations
        Dict[str, float],    # rewards
        Dict[str, bool],     # terminations
        Dict[str, bool],     # truncations
        Dict[str, Any],      # infos
    ]:
        """
        Execute one timestep for all agents simultaneously.

        Args:
            actions: Dict mapping agent_id to action array [green_delta, phase_hold].

        Returns:
            observations, rewards, terminations, truncations, infos
        """
        # Validate and format actions
        formatted_actions: Dict[str, List[float]] = {}
        for agent in self.agents:
            if agent in actions:
                act = actions[agent]
                if hasattr(act, 'tolist'):
                    act = act.tolist()
                elif not isinstance(act, list):
                    act = list(act)
                formatted_actions[agent] = act
            else:
                # Default: no adjustment, no phase hold
                formatted_actions[agent] = [0.0, 0.0]

        # Step the simulation
        obs_dict, rewards, terminated, truncated, info = self._engine.step(formatted_actions)

        self._current_step += 1

        # PettingZoo requires per-agent termination/truncation dicts
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}

        observations = {}
        infos: Dict[str, Any] = {}
        for agent in self.agents:
            observations[agent] = obs_dict.get(agent, [0.0] * OBS_DIM)
            infos[agent] = info

        # If episode ended, clear agents list (PettingZoo convention)
        if terminated or truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        """Returns the observation space for the given agent."""
        return self._obs_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        """Returns the action space for the given agent."""
        return self._action_spaces[agent]

    def render(self) -> None:
        """Rendering not implemented (use dashboard visualization)."""
        pass

    def close(self) -> None:
        """Clean up resources."""
        pass

    def state(self) -> Dict[str, Any]:
        """
        Returns a global state representation (optional PettingZoo method).

        Useful for centralized critics in CTDE (Centralized Training,
        Decentralized Execution) architectures like MAPPO.
        """
        vehicles = self._engine._vehicles
        edge_meta = self._engine._edge_meta
        link_states = self._engine._link_states

        # Global state: aggregate network statistics
        total_veh = sum(vehicles.values())
        total_queue = 0.0
        total_delay = 0.0
        n_spillback = 0

        for eid, state in link_states.items():
            meta = edge_meta.get(eid)
            if meta:
                total_queue += state.queue_length_m / max(1.0, meta["length_m"])
                total_delay += max(0.0, 1.0 - state.speed_mps / max(1.0, meta["free_speed_mps"]))
                if state.is_spillback:
                    n_spillback += 1

        n_links = max(1, len(link_states))
        t_s = self._current_step * self._dt_s

        return {
            "total_vehicles": total_veh,
            "avg_queue_ratio": total_queue / n_links,
            "avg_delay_ratio": total_delay / n_links,
            "spillback_count": n_spillback,
            "time_s": t_s,
            "time_hr": t_s / 3600.0,
            "boundary_queue": sum(self._engine._boundary_queue.values()),
            "generated": self._engine._generated,
            "exited": self._engine._exited,
        }

    def get_episode_summary(self) -> Dict[str, Any]:
        """Returns detailed episode summary from the engine."""
        return self._engine.get_episode_summary()

    # ── Utility Methods ─────────────────────────────────────────────────

    @property
    def num_agents(self) -> int:
        """Number of active agents."""
        return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        """Maximum number of agents."""
        return len(self.possible_agents)

    def agent_name_mapping(self) -> Dict[str, str]:
        """Maps agent IDs to human-readable intersection names."""
        return {d["agent_id"]: d["name"] for d in SIGNAL_AGENT_DEFS}

    def get_agent_intersection_info(self, agent_id: str) -> Dict[str, Any]:
        """Returns metadata about the intersection controlled by an agent."""
        for d in SIGNAL_AGENT_DEFS:
            if d["agent_id"] == agent_id:
                return {
                    "signal_id": d["signal_id"],
                    "name": d["name"],
                    "cycle_length_s": d["cycle_length_s"],
                    "base_arterial_green_s": d["base_arterial_green_s"],
                    "min_green_s": d["min_green_s"],
                    "max_green_s": d["max_green_s"],
                    "clearance_s": d["clearance_s"],
                    "n_controlled_edges": len(self._engine._agent_edge_map.get(agent_id, [])),
                }
        return {}

    def __str__(self) -> str:
        return (
            f"NormanTrafficEnv(agents={len(self.possible_agents)}, "
            f"dt={self._dt_s}s, episode={self._episode_duration_s/3600:.1f}h, "
            f"obs_dim={OBS_DIM}, action_dim={ACTION_DIM})"
        )

    def __repr__(self) -> str:
        return self.__str__()
