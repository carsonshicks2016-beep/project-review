"""
Comprehensive test suite for the SoonerGrid Gymnasium/PettingZoo RL Environment.

Tests cover:
    1. Environment construction and initialization
    2. Space definitions and validity
    3. Reset semantics and observation structure
    4. Step mechanics with random and deterministic actions
    5. Vehicle conservation across RL steps
    6. Reward computation and sign conventions
    7. Episode termination and truncation
    8. PettingZoo API contract compliance
    9. SteppableTrafficEngine internal consistency
    10. Multi-episode reset stability
    11. Edge cases (zero demand, single step, max actions)
"""

import unittest
import math
import networkx as nx

from soonergrid.physics.link_attributes import EdgeAttributes
from soonergrid.data.game_day_zones import Gateway, ParkingSink
from soonergrid.gym.steppable_engine import (
    SteppableTrafficEngine,
    SIGNAL_AGENT_DEFS,
    OBS_DIM,
    ACTION_DIM,
)
from soonergrid.gym.env import NormanTrafficEnv
from soonergrid.gym.spaces import FallbackBox, make_observation_space, make_action_space


def _make_test_graph():
    """
    Creates a minimal 4-node, 4-edge test network shaped like:

        (1) ──e1──> (2) ──e2──> (3) ──e3──> (4)
                     │
                     └──e4──> (4)

    Node 1: gateway injection point
    Node 4: parking sink
    Edge e1: "W Lindsey St" (primary) - matches sig_i35_lindsey agent
    Edge e2: "W Lindsey St" (primary) - matches sig_lindsey_berry agent  
    Edge e3: "W Lindsey St" (primary) - through route
    Edge e4: "S Jenkins Ave" (tertiary) - matches sig_lindsey_jenkins cross
    """
    G = nx.MultiDiGraph()
    G.add_node(1, x=-500.0, y=0.0)
    G.add_node(2, x=-200.0, y=0.0)
    G.add_node(3, x=100.0, y=0.0)
    G.add_node(4, x=300.0, y=0.0)

    edges = [
        EdgeAttributes(
            edge_id="e1", u=1, v=2, name="W Lindsey St",
            highway_type="primary", length_m=300.0, lanes=2,
            free_speed_mps=15.0, free_speed_mph=33.5,
            free_flow_time_s=20.0, capacity_vph=3600.0,
            jam_storage_veh=80, is_oneway=True, is_reversible=False,
        ),
        EdgeAttributes(
            edge_id="e2", u=2, v=3, name="W Lindsey St",
            highway_type="primary", length_m=300.0, lanes=2,
            free_speed_mps=15.0, free_speed_mph=33.5,
            free_flow_time_s=20.0, capacity_vph=3600.0,
            jam_storage_veh=80, is_oneway=True, is_reversible=False,
        ),
        EdgeAttributes(
            edge_id="e3", u=3, v=4, name="W Lindsey St",
            highway_type="primary", length_m=200.0, lanes=2,
            free_speed_mps=15.0, free_speed_mph=33.5,
            free_flow_time_s=13.3, capacity_vph=3600.0,
            jam_storage_veh=53, is_oneway=True, is_reversible=False,
        ),
        EdgeAttributes(
            edge_id="e4", u=2, v=4, name="S Jenkins Ave",
            highway_type="tertiary", length_m=250.0, lanes=1,
            free_speed_mps=11.2, free_speed_mph=25.0,
            free_flow_time_s=22.3, capacity_vph=700.0,
            jam_storage_veh=33, is_oneway=True, is_reversible=False,
        ),
    ]

    for attr in edges:
        G.add_edge(attr.u, attr.v, key=attr.edge_id, attr=attr)

    return G


def _make_test_fixtures():
    """Creates a complete set of test fixtures for the env."""
    graph = _make_test_graph()

    gateways = [
        Gateway(
            id="GW_I35_NORTH", name="Test I-35 Gateway",
            lat=35.295, lon=-97.485, corridor="I-35",
            primary_origin="North", hourly_inflow_weight=0.48,
            nearest_node=1, dist_to_node_m=0.0,
        ),
    ]

    sinks = [
        ParkingSink(
            id="SINK_TEST", name="Test Stadium Sink",
            lat=35.2059, lon=-97.4423, capacity_stalls=5000,
            zone_type="stadium_core", shuttle_served=False,
            description="Test sink", nearest_node=4, dist_to_node_m=0.0,
        ),
    ]

    demand = {
        "timeline": [
            {
                "time_hr": 0.0,
                "phase": "Pre-Game",
                "gateway_inflows": {"GW_I35_NORTH": 1000.0},
            },
            {
                "time_hr": 0.25,
                "phase": "Pre-Game Peak",
                "gateway_inflows": {"GW_I35_NORTH": 2000.0},
            },
        ],
    }

    return graph, gateways, sinks, demand


class TestFallbackSpaces(unittest.TestCase):
    """Tests for FallbackBox space implementation."""

    def test_construction(self):
        box = FallbackBox(low=[0.0, 0.0], high=[1.0, 1.0], shape=(2,))
        self.assertEqual(box.shape, (2,))
        self.assertEqual(len(box.low), 2)
        self.assertEqual(len(box.high), 2)

    def test_sample_within_bounds(self):
        box = FallbackBox(low=[-1.0, 0.0], high=[1.0, 1.0], shape=(2,))
        for _ in range(50):
            sample = box.sample()
            self.assertEqual(len(sample), 2)
            self.assertGreaterEqual(sample[0], -1.0)
            self.assertLessEqual(sample[0], 1.0)
            self.assertGreaterEqual(sample[1], 0.0)
            self.assertLessEqual(sample[1], 1.0)

    def test_contains(self):
        box = FallbackBox(low=[0.0, 0.0], high=[1.0, 1.0], shape=(2,))
        self.assertTrue(box.contains([0.5, 0.5]))
        self.assertTrue(box.contains([0.0, 0.0]))
        self.assertTrue(box.contains([1.0, 1.0]))
        self.assertFalse(box.contains([-0.1, 0.5]))
        self.assertFalse(box.contains([0.5]))
        self.assertFalse(box.contains([0.5, 0.5, 0.5]))

    def test_seed_reproducibility(self):
        box = FallbackBox(low=[0.0], high=[1.0], shape=(1,))
        box.seed(42)
        s1 = box.sample()
        box.seed(42)
        s2 = box.sample()
        self.assertEqual(s1, s2)

    def test_observation_space_factory(self):
        space = make_observation_space(OBS_DIM)
        self.assertEqual(space.shape, (OBS_DIM,))
        sample = space.sample()
        self.assertEqual(len(sample), OBS_DIM)

    def test_action_space_factory(self):
        space = make_action_space(ACTION_DIM)
        self.assertEqual(space.shape, (ACTION_DIM,))
        sample = space.sample()
        self.assertEqual(len(sample), ACTION_DIM)


class TestSteppableEngine(unittest.TestCase):
    """Tests for the SteppableTrafficEngine core."""

    def setUp(self):
        graph, gateways, sinks, demand = _make_test_fixtures()
        self.engine = SteppableTrafficEngine(
            graph=graph,
            gateways=gateways,
            sinks=sinks,
            demand_dataset=demand,
            dt_s=5.0,
            episode_duration_s=900.0,  # 15 minutes
        )

    def test_agent_ids(self):
        """Verify all 7 signal agents are defined."""
        self.assertEqual(len(self.engine.agent_ids), 7)
        self.assertIn("sig_i35_lindsey", self.engine.agent_ids)
        self.assertIn("sig_sh9_jenkins", self.engine.agent_ids)

    def test_obs_dim(self):
        self.assertEqual(self.engine.obs_dim, OBS_DIM)
        self.assertEqual(OBS_DIM, 36)

    def test_action_dim(self):
        self.assertEqual(self.engine.action_dim, ACTION_DIM)
        self.assertEqual(ACTION_DIM, 2)

    def test_reset_returns_observations(self):
        obs = self.engine.reset()
        self.assertIsInstance(obs, dict)
        for agent_id in self.engine.agent_ids:
            self.assertIn(agent_id, obs)
            self.assertEqual(len(obs[agent_id]), OBS_DIM)

    def test_initial_observations_bounded(self):
        """All observation values should be in [0, 1] at t=0."""
        obs = self.engine.reset()
        for agent_id, obs_vec in obs.items():
            for i, val in enumerate(obs_vec):
                self.assertGreaterEqual(val, -0.01,
                    f"Agent {agent_id} obs[{i}]={val} below 0")
                self.assertLessEqual(val, 1.01,
                    f"Agent {agent_id} obs[{i}]={val} above 1")

    def test_step_returns_correct_structure(self):
        self.engine.reset()
        actions = {aid: [0.0, 0.0] for aid in self.engine.agent_ids}
        obs, rewards, terminated, truncated, info = self.engine.step(actions)

        self.assertIsInstance(obs, dict)
        self.assertIsInstance(rewards, dict)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_step_observations_per_agent(self):
        self.engine.reset()
        actions = {aid: [0.0, 0.0] for aid in self.engine.agent_ids}
        obs, _, _, _, _ = self.engine.step(actions)
        for agent_id in self.engine.agent_ids:
            self.assertIn(agent_id, obs)
            self.assertEqual(len(obs[agent_id]), OBS_DIM)

    def test_step_rewards_per_agent(self):
        self.engine.reset()
        actions = {aid: [0.0, 0.0] for aid in self.engine.agent_ids}
        _, rewards, _, _, _ = self.engine.step(actions)
        for agent_id in self.engine.agent_ids:
            self.assertIn(agent_id, rewards)
            self.assertIsInstance(rewards[agent_id], float)

    def test_vehicle_conservation_multi_step(self):
        """Run 100 steps and verify conservation holds at each step."""
        self.engine.reset()
        actions = {aid: [0.0, 0.0] for aid in self.engine.agent_ids}
        for step in range(100):
            obs, rewards, terminated, truncated, info = self.engine.step(actions)
            # Conservation is checked inside step() and raises ArithmeticError
            # If we get here, conservation holds
            self.assertLessEqual(
                info["max_conservation_residual"], 1e-3,
                f"Conservation residual too large at step {step}"
            )

    def test_truncation_at_episode_end(self):
        """Episode truncates when total_steps is reached."""
        # dt=5s, duration=900s -> 180 steps
        self.engine.reset()
        actions = {aid: [0.0, 0.0] for aid in self.engine.agent_ids}
        for step in range(179):
            _, _, terminated, truncated, _ = self.engine.step(actions)
            self.assertFalse(truncated, f"Premature truncation at step {step}")
            self.assertFalse(terminated)

        # Step 180 should truncate
        _, _, terminated, truncated, _ = self.engine.step(actions)
        self.assertTrue(truncated, "Expected truncation at final step")

    def test_green_split_action_clamped(self):
        """Extreme green split adjustments should be clamped to [min, max]."""
        self.engine.reset()

        # Max positive delta
        actions = {aid: [1.0, 0.0] for aid in self.engine.agent_ids}
        self.engine.step(actions)
        for agent_id in self.engine.agent_ids:
            max_green = self.engine._agent_defs[agent_id]["max_green_s"]
            self.assertLessEqual(
                self.engine._active_art_green[agent_id], max_green + 0.01
            )

        # Max negative delta
        actions = {aid: [-1.0, 0.0] for aid in self.engine.agent_ids}
        for _ in range(10):
            self.engine.step(actions)
        for agent_id in self.engine.agent_ids:
            min_green = self.engine._agent_defs[agent_id]["min_green_s"]
            self.assertGreaterEqual(
                self.engine._active_art_green[agent_id], min_green - 0.01
            )

    def test_random_actions_no_crash(self):
        """Engine should not crash under random actions for a full episode."""
        self.engine.reset()
        import random
        rng = random.Random(42)
        actions = {}
        for step in range(180):
            for aid in self.engine.agent_ids:
                actions[aid] = [rng.uniform(-1, 1), rng.uniform(0, 1)]
            obs, rewards, terminated, truncated, info = self.engine.step(actions)
            if terminated or truncated:
                break

    def test_reset_clears_state(self):
        """After reset, all vehicles and counters should be zero."""
        self.engine.reset()
        actions = {aid: [0.0, 0.0] for aid in self.engine.agent_ids}
        for _ in range(50):
            self.engine.step(actions)

        # Verify some vehicles exist
        total = sum(self.engine._vehicles.values())
        self.assertGreater(total, 0, "No vehicles injected during 50 steps")

        # Reset and verify clean state
        self.engine.reset()
        total_after = sum(self.engine._vehicles.values())
        self.assertEqual(total_after, 0.0, "Vehicles not cleared on reset")
        self.assertEqual(self.engine._step_idx, 0)
        self.assertEqual(self.engine._generated, 0.0)
        self.assertEqual(self.engine._exited, 0.0)

    def test_multi_episode_stability(self):
        """Engine should support multiple reset/episode cycles."""
        for ep in range(3):
            self.engine.reset()
            actions = {aid: [0.0, 0.0] for aid in self.engine.agent_ids}
            for step in range(30):
                self.engine.step(actions)

    def test_episode_summary(self):
        self.engine.reset()
        actions = {aid: [0.0, 0.0] for aid in self.engine.agent_ids}
        for _ in range(20):
            self.engine.step(actions)
        summary = self.engine.get_episode_summary()
        self.assertIn("conservation", summary)
        self.assertIn("episode_steps", summary)
        self.assertEqual(summary["episode_steps"], 20)


class TestNormanTrafficEnv(unittest.TestCase):
    """Tests for the PettingZoo-compatible environment wrapper."""

    def setUp(self):
        graph, gateways, sinks, demand = _make_test_fixtures()
        self.env = NormanTrafficEnv(
            graph=graph,
            gateways=gateways,
            sinks=sinks,
            demand_dataset=demand,
            dt_s=5.0,
            episode_duration_s=900.0,
            auto_load_network=False,
        )

    def test_possible_agents(self):
        self.assertEqual(len(self.env.possible_agents), 7)
        for agent in self.env.possible_agents:
            self.assertIsInstance(agent, str)
            self.assertTrue(agent.startswith("sig_"))

    def test_observation_space(self):
        for agent in self.env.possible_agents:
            space = self.env.observation_space(agent)
            self.assertEqual(space.shape, (OBS_DIM,))
            sample = space.sample()
            self.assertEqual(len(sample), OBS_DIM)

    def test_action_space(self):
        for agent in self.env.possible_agents:
            space = self.env.action_space(agent)
            self.assertEqual(space.shape, (ACTION_DIM,))
            sample = space.sample()
            self.assertEqual(len(sample), ACTION_DIM)

    def test_reset_returns_obs_and_infos(self):
        obs, infos = self.env.reset()
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(infos, dict)
        for agent in self.env.possible_agents:
            self.assertIn(agent, obs)
            self.assertIn(agent, infos)

    def test_step_pettingzoo_api(self):
        """Step returns (obs, rewards, terminations, truncations, infos)."""
        self.env.reset()
        actions = {
            agent: self.env.action_space(agent).sample()
            for agent in self.env.agents
        }
        obs, rewards, terms, truncs, infos = self.env.step(actions)

        self.assertIsInstance(obs, dict)
        self.assertIsInstance(rewards, dict)
        self.assertIsInstance(terms, dict)
        self.assertIsInstance(truncs, dict)
        self.assertIsInstance(infos, dict)

        for agent in self.env.possible_agents:
            self.assertIn(agent, terms)
            self.assertIsInstance(terms[agent], bool)
            self.assertIn(agent, truncs)
            self.assertIsInstance(truncs[agent], bool)

    def test_agents_cleared_on_truncation(self):
        """PettingZoo convention: agents list cleared when episode ends."""
        self.env.reset()
        actions = {
            agent: [0.0, 0.0] for agent in self.env.agents
        }
        for _ in range(180):  # Full episode
            obs, rewards, terms, truncs, infos = self.env.step(actions)
            if any(truncs.values()):
                break
            actions = {agent: [0.0, 0.0] for agent in self.env.agents}

        self.assertEqual(len(self.env.agents), 0, "Agents not cleared after truncation")

    def test_agents_restored_on_reset(self):
        """After episode ends and reset, agents should be restored."""
        self.env.reset()
        actions = {agent: [0.0, 0.0] for agent in self.env.agents}
        for _ in range(180):
            self.env.step(actions)
            if not self.env.agents:
                break

        obs, infos = self.env.reset()
        self.assertEqual(len(self.env.agents), 7)

    def test_state_method(self):
        """Global state should return network-wide diagnostics."""
        self.env.reset()
        actions = {agent: [0.0, 0.0] for agent in self.env.agents}
        self.env.step(actions)

        state = self.env.state()
        self.assertIn("total_vehicles", state)
        self.assertIn("avg_queue_ratio", state)
        self.assertIn("time_s", state)

    def test_agent_name_mapping(self):
        mapping = self.env.agent_name_mapping()
        self.assertEqual(len(mapping), 7)
        self.assertIn("sig_i35_lindsey", mapping)
        self.assertIn("I-35", mapping["sig_i35_lindsey"])

    def test_get_agent_intersection_info(self):
        info = self.env.get_agent_intersection_info("sig_i35_lindsey")
        self.assertIn("signal_id", info)
        self.assertEqual(info["signal_id"], "SIG_I35_LINDSEY_SPUI")
        self.assertIn("cycle_length_s", info)
        self.assertEqual(info["cycle_length_s"], 90.0)

    def test_str_repr(self):
        s = str(self.env)
        self.assertIn("NormanTrafficEnv", s)
        self.assertIn("agents=7", s)

    def test_episode_summary(self):
        self.env.reset()
        actions = {agent: [0.0, 0.0] for agent in self.env.agents}
        for _ in range(20):
            self.env.step(actions)
        summary = self.env.get_episode_summary()
        self.assertIn("conservation", summary)

    def test_num_agents_property(self):
        self.env.reset()
        self.assertEqual(self.env.num_agents, 7)
        self.assertEqual(self.env.max_num_agents, 7)


class TestRewardProperties(unittest.TestCase):
    """Tests for reward function behavior."""

    def setUp(self):
        graph, gateways, sinks, demand = _make_test_fixtures()
        self.engine = SteppableTrafficEngine(
            graph=graph,
            gateways=gateways,
            sinks=sinks,
            demand_dataset=demand,
            dt_s=5.0,
            episode_duration_s=900.0,
        )

    def test_rewards_finite(self):
        """All rewards must be finite real numbers."""
        self.engine.reset()
        actions = {aid: [0.0, 0.0] for aid in self.engine.agent_ids}
        for _ in range(50):
            _, rewards, _, _, _ = self.engine.step(actions)
            for agent_id, r in rewards.items():
                self.assertTrue(math.isfinite(r),
                    f"Non-finite reward {r} for agent {agent_id}")

    def test_empty_network_zero_penalty(self):
        """At t=0 with no vehicles, rewards should be near zero."""
        self.engine.reset()
        actions = {aid: [0.0, 0.0] for aid in self.engine.agent_ids}
        _, rewards, _, _, _ = self.engine.step(actions)
        for agent_id, r in rewards.items():
            # With very low demand in first step, reward should be small
            self.assertGreater(r, -10.0,
                f"Unexpectedly large negative reward {r} at step 0")


class TestZeroDemand(unittest.TestCase):
    """Edge case: zero demand should not crash."""

    def test_zero_demand_episode(self):
        graph, gateways, sinks, _ = _make_test_fixtures()
        demand = {
            "timeline": [
                {"time_hr": 0.0, "phase": "Empty", "gateway_inflows": {"GW_I35_NORTH": 0.0}},
            ],
        }
        engine = SteppableTrafficEngine(
            graph=graph, gateways=gateways, sinks=sinks,
            demand_dataset=demand, dt_s=5.0, episode_duration_s=100.0,
        )
        engine.reset()
        actions = {aid: [0.0, 0.0] for aid in engine.agent_ids}
        for _ in range(20):
            obs, rewards, terminated, truncated, info = engine.step(actions)
            # With zero demand, no vehicles should exist
            self.assertAlmostEqual(info["total_vehicles_on_network"], 0.0, places=5)


class TestIncidentScenario(unittest.TestCase):
    """Test that incident scenarios don't crash the RL env."""

    def test_shock_collision(self):
        graph, gateways, sinks, demand = _make_test_fixtures()
        engine = SteppableTrafficEngine(
            graph=graph, gateways=gateways, sinks=sinks,
            demand_dataset=demand, dt_s=5.0, episode_duration_s=300.0,
            incident_shock="SHOCK_COLLISION",
            enable_self_healing=True,
        )
        engine.reset()
        actions = {aid: [0.0, 0.0] for aid in engine.agent_ids}
        for _ in range(60):
            obs, rewards, terminated, truncated, info = engine.step(actions)
            if terminated or truncated:
                break


if __name__ == "__main__":
    unittest.main()
