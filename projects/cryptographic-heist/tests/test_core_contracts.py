from __future__ import annotations

import math
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from crypt_heist.adversarial import run_adversarial_eval, run_authentication_curriculum_eval
from crypt_heist.comms import CommsChannel, VOCABULARY
from crypt_heist.config import get_car
from crypt_heist.control_pool import load_control_pool, select_control_opponent
from crypt_heist.curriculum import CurriculumStage, next_stage, plan_curriculum_from_manifest
from crypt_heist.cycle import run_information_cycle
from crypt_heist.env import CryptHeistParallelEnv, scripted_random_actions
from crypt_heist.env_validation import run_env_validation
from crypt_heist.evidence import build_evidence_bundle, verify_evidence_bundle
from crypt_heist.imitation import collect_evader_dataset, train_evader_imitation
from crypt_heist.jamming import (
    EvaderJammerController,
    collect_counterfactual_jammer_dataset,
    collect_jammer_dataset,
    load_jammer_policy,
    train_counterfactual_jammer_policy,
    train_jammer_policy,
)
from crypt_heist.league import evaluate_checkpoint_set
from crypt_heist.marl import EvaderPolicy, PursuerPolicy
from crypt_heist.observations import AGENT_NAMES, OBS_SIZE, build_observation
from crypt_heist.operational_validation import run_operational_validation
from crypt_heist.policy_runtime import EvaderCheckpointController, PursuerCheckpointController, PursuerTeamCheckpointController
from crypt_heist.physics import Controls, Vehicle
from crypt_heist.physics_validation import run_physics_validation
from crypt_heist.ppo import train_evader_ppo, train_pursuer_ppo, train_pursuer_team_ppo
from crypt_heist.radio import PursuerRadioController, collect_radio_dataset, load_radio_policy, train_radio_policy
from crypt_heist.research_report import build_v2_research_report
from crypt_heist.rewards import RewardSnapshot, RewardSystem, RewardWeights
from crypt_heist.replay import (
    compare_replay_fingerprints,
    load_replay,
    record_scripted_replay,
    ReplayRecorder,
    replay_diagnostic_report,
    replay_fingerprint,
    summarize_replay,
    validate_replay,
)
from crypt_heist.scenarios import ScenarioThresholds, run_acceptance_suite, run_checkpoint_acceptance_suite, score_acceptance_manifest
from crypt_heist.scanner import ScannerDecoderRuntime, collect_scanner_dataset, train_scanner_decoder
from crypt_heist.selfplay import ControlCheckpointSet, _control_promotion_decision, load_active_control_set, run_self_play_cycle
from crypt_heist.sim import HeistSim
from crypt_heist.spectator_validation import run_spectator_validation


class CoreContractTests(unittest.TestCase):
    def test_deterministic_scripted_trace(self):
        a = HeistSim(seed=44)
        b = HeistSim(seed=44)
        for _ in range(240):
            a.step()
            b.step()
        av = [(round(x.vehicle.x, 6), round(x.vehicle.y, 6), round(x.vehicle.yaw, 6)) for x in a.agents]
        bv = [(round(x.vehicle.x, 6), round(x.vehicle.y, 6), round(x.vehicle.yaw, 6)) for x in b.agents]
        self.assertEqual(av, bv)
        self.assertEqual(a.channel.jamming_budget, b.channel.jamming_budget)

    def test_observation_contract(self):
        sim = HeistSim(seed=4)
        for agent in AGENT_NAMES:
            obs = build_observation(sim, agent)
            self.assertEqual(obs.shape, (OBS_SIZE,))
            self.assertTrue(np.isfinite(obs).all())

    def test_parallel_env_contract(self):
        env = CryptHeistParallelEnv(seed=5, max_cycles=10)
        obs, infos = env.reset(seed=5)
        self.assertEqual(set(obs), set(AGENT_NAMES))
        self.assertEqual(obs["evader_0"].shape, (OBS_SIZE,))
        actions = scripted_random_actions(env)
        obs, rewards, terms, truncs, infos = env.step(actions)
        self.assertEqual(set(rewards), set(AGENT_NAMES))
        self.assertIn("reward_components", infos["evader_0"])
        self.assertIn("decoder_accuracy", infos["pursuer_0"]["reward_components"])
        self.assertIn("spoof_susceptibility", infos["evader_0"]["reward_components"])

    def test_parallel_env_counterfactual_probe(self):
        env = CryptHeistParallelEnv(
            seed=7,
            max_cycles=3,
            control_repeat=1,
            counterfactual_interval=1,
            counterfactual_horizon_steps=3,
            counterfactual_evader_weight=0.25,
            counterfactual_pursuer_weight=0.25,
        )
        env.reset(seed=7)
        actions = scripted_random_actions(env)
        obs, rewards, terms, truncs, infos = env.step(actions)
        components = infos["evader_0"]["reward_components"]
        self.assertEqual(components["counterfactual_probe_active"], 1)
        self.assertGreaterEqual(components["counterfactual_deception"], 0.0)
        self.assertGreaterEqual(components["counterfactual_evader_bonus"], 0.0)
        self.assertGreaterEqual(components["counterfactual_pursuer_penalty"], 0.0)

    def test_env_validation_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "env_validation.json"
            manifest = run_env_validation(seed=8, steps=3, out=out)
            self.assertTrue(out.exists())
            self.assertTrue(manifest["passed"])
            self.assertEqual(manifest["checks_passed"], manifest["checks"])
            by_name = {row["name"]: row for row in manifest["results"]}
            self.assertTrue(by_name["pettingzoo_parallel_api"]["passed"])
            self.assertTrue(by_name["seeded_env_determinism"]["passed"])

    def test_replay_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sample.jsonl"
            record_scripted_replay(path, seed=6, steps=30)
            replay = load_replay(path)
            validate_replay(replay)
            frame = replay[1]
            for key in ("actions", "collisions", "rewards", "reward_components", "camera"):
                self.assertIn(key, frame)
            self.assertEqual(len(frame["actions"]), 6)
            self.assertIn("EVADER", frame["actions"])
            self.assertEqual(len(frame["actions"]["EVADER"]["control"]), 4)
            self.assertEqual(set(frame["rewards"]), {"EVADER", "P1", "P2", "P3", "P4", "P5"})
            self.assertEqual(set(frame["reward_components"]), {"EVADER", "P1", "P2", "P3", "P4", "P5"})
            self.assertIn("decoder_accuracy", frame["reward_components"]["EVADER"])
            self.assertIn("spoof_susceptibility", frame["reward_components"]["EVADER"])
            self.assertIn("auth_penalty", frame["reward_components"]["P1"])
            self.assertEqual(len(frame["camera"]["focus"]), 2)
            self.assertIn(frame["camera"]["event"], {"chase", "impact", "waypoint", "capture"})
            summary = summarize_replay(replay)
            self.assertEqual(summary["frames"], 30)
            self.assertEqual(summary["agents"], 6)
            self.assertEqual(summary["action_frames"], 30)
            self.assertEqual(summary["camera_hint_frames"], 30)
            self.assertEqual(summary["reward_frames"], 30)
            self.assertIn("avg_evader_reward", summary)
            self.assertIn("avg_pursuer_auth_penalty", summary)
            self.assertIn("avg_decoder_accuracy", summary)
            self.assertEqual(summary["reward_agent_coverage"], 1.0)
            self.assertIn("radio_word_entropy", summary)
            self.assertIn("pursuer_word_entropy", summary)
            self.assertIn("radio_spoof_ratio", summary)
            self.assertIn("jam_events", summary)
            self.assertIn("cipher_rotations", summary)
            self.assertGreaterEqual(summary["radio_word_entropy"], 0.0)
            self.assertGreaterEqual(summary["confidence_range"], 0.0)
            self.assertIn("avg_confidence", summary)
            self.assertIn("avg_evader_speed", summary)
            self.assertEqual(len(summary["state_sha256"]), 64)
            self.assertEqual(summary["state_sha256"], replay_fingerprint(replay)["state_sha256"])

            twin = Path(td) / "sample-twin.jsonl"
            record_scripted_replay(twin, seed=6, steps=30)
            comparison = compare_replay_fingerprints(replay, load_replay(twin))
            self.assertTrue(comparison["matched"])
            self.assertTrue(comparison["state_matched"])

    def test_replay_diagnostic_report_contract(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "report.jsonl"
            record_scripted_replay(path, seed=13, steps=60)
            replay = load_replay(path)
            report = replay_diagnostic_report(path, replay)
            self.assertEqual(report["kind"], "replay_diagnostic_report")
            self.assertIn("summary", report)
            self.assertIn("sections", report)
            self.assertIn("chase", report["sections"])
            self.assertIn("information_warfare", report["sections"])
            self.assertIn("reward_trace", report["sections"])
            self.assertIn("replay_contract", report["sections"])
            self.assertIn("notable_events", report)
            self.assertIn("diagnostics", report)
            self.assertEqual(report["sections"]["replay_contract"]["action_frames"], 60)
            self.assertEqual(report["sections"]["reward_trace"]["reward_frames"], 60)
            self.assertEqual(report["sections"]["reward_trace"]["reward_agent_coverage"], 1.0)
            self.assertGreaterEqual(report["sections"]["information_warfare"]["radio_word_entropy"], 0.0)
            self.assertEqual(len(report["sections"]["replay_contract"]["state_sha256"]), 64)

    def test_replay_records_requested_actions(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "external-actions.jsonl"
            sim = HeistSim(seed=9)
            actions = {
                "evader_0": {
                    "control": [0.45, 0.8, 1.0],
                    "jam": 1,
                    "spoof_tokens": [1, 2, 3, 4, 5],
                    "target_mask": [1, 0, 1, 0, 0],
                },
                "pursuer_0": {
                    "control": [-0.2, 0.5, -1.0],
                    "tokens": [3, 4, 5, 6, 7, 8],
                },
            }
            with ReplayRecorder(path, seed=9, dt=sim.sim.dt) as recorder:
                sim.step(actions=actions)
                recorder.record(sim.snapshot())
            replay = load_replay(path)
            validate_replay(replay)
            frame = replay[1]
            self.assertEqual(frame["actions"]["EVADER"]["source"], "external")
            self.assertEqual(frame["actions"]["EVADER"]["requested_control"], [0.45, 0.8, 1.0])
            self.assertEqual(frame["actions"]["EVADER"]["target_mask"], [1, 0, 1, 0, 0])
            self.assertTrue(frame["actions"]["EVADER"]["jam"])
            self.assertEqual(frame["actions"]["P1"]["tokens"], [3, 4, 5, 6, 7, 8])
            self.assertEqual(frame["actions"]["P2"]["source"], "scripted")
            self.assertIn("EVADER", frame["rewards"])
            self.assertIn("information_reward", frame["reward_components"]["EVADER"])

    def test_replay_time_survives_capture_reset(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "capture-reset.jsonl"
            sim = HeistSim(seed=26, reset_on_capture=True)
            ev = sim.evader.vehicle
            ev.vx = 0.0
            ev.vy = 0.0
            for i, pursuer in enumerate(sim.pursuers[:3]):
                pursuer.vehicle.x = ev.x + float(i)
                pursuer.vehicle.y = ev.y
                pursuer.vehicle.vx = 0.0
                pursuer.vehicle.vy = 0.0
            sim.capture_timer = 1.25
            with ReplayRecorder(path, seed=26, dt=sim.sim.dt) as recorder:
                for _ in range(8):
                    sim.step()
                    recorder.record(sim.snapshot())
            replay = load_replay(path)
            validate_replay(replay)
            times = [frame["time"] for frame in replay[1:]]
            self.assertEqual(times, sorted(times))
            self.assertGreaterEqual(summarize_replay(replay)["captures"], 1)

    def test_acceptance_scenarios_record_replay_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = run_acceptance_suite(
                scenarios=["downtown_chase", "roadblock", "spoof_burst", "cipher_shift"],
                seed=31,
                steps=260,
                record_dir=root / "replays",
                out=root / "acceptance.json",
                thresholds=ScenarioThresholds(
                    downtown_min_radio_events=6,
                    downtown_min_avg_speed=1.0,
                    roadblock_min_boxed_frames=80,
                    roadblock_max_capture_time=2.5,
                    cipher_min_recovery=0.0,
                ),
            )
            self.assertTrue((root / "acceptance.json").exists())
            self.assertTrue(manifest["passed"])
            self.assertGreater(manifest["score"]["overall_score"], 0.0)
            self.assertEqual(score_acceptance_manifest(manifest)["required_score"], 1.0)
            self.assertIn("evader_control_score", manifest["score"])
            self.assertIn("pursuer_team_control_score", manifest["score"])
            self.assertIn("diagnostics", manifest)
            self.assertEqual(len(manifest["scenarios"]), 4)
            by_name = {scenario["name"]: scenario for scenario in manifest["scenarios"]}
            self.assertGreaterEqual(by_name["roadblock"]["metrics"]["capture_events"], 1)
            self.assertGreaterEqual(by_name["spoof_burst"]["replay_summary"]["spoofed_radio_events"], 5)
            self.assertGreaterEqual(by_name["spoof_burst"]["replay_summary"]["jam_events"], 1)
            self.assertGreaterEqual(by_name["spoof_burst"]["replay_summary"]["max_jam_confidence_drop"], 0.0)
            self.assertGreaterEqual(by_name["cipher_shift"]["metrics"]["cipher_rotations"], 1)
            self.assertGreaterEqual(by_name["cipher_shift"]["replay_summary"]["cipher_rotations"], 1)
            self.assertGreaterEqual(by_name["cipher_shift"]["replay_summary"]["avg_cipher_confidence_drop"], 0.0)
            self.assertEqual(by_name["downtown_chase"]["milestone_targets"], 1)
            self.assertIn("diagnostics", by_name["downtown_chase"])
            self.assertIn("reason", by_name["downtown_chase"]["primary_diagnostic"])
            for scenario in manifest["scenarios"]:
                replay = load_replay(scenario["replay"])
                validate_replay(replay)
                self.assertEqual(summarize_replay(replay)["frames"], scenario["metrics"]["frames"])

            strict = run_acceptance_suite(
                scenarios=["downtown_chase"],
                seed=11,
                steps=1200,
                record_dir=None,
                out=None,
                thresholds=ScenarioThresholds(downtown_min_waypoints=3),
            )
            self.assertTrue(strict["passed"])
            self.assertIsNone(strict["scenarios"][0]["replay"])
            self.assertGreaterEqual(strict["scenarios"][0]["metrics"]["waypoints_hit"], 3)
            self.assertEqual(strict["scenarios"][0]["metrics"]["frames"], 1200)
            self.assertEqual(strict["diagnostics"]["count"], 0)

    def test_jamming_budget_and_spoof_labels(self):
        channel = CommsChannel(np.random.default_rng(1))
        ok = channel.inject_jam(1.0, payload=[0, 1, 2, 3, 4])
        self.assertTrue(ok)
        self.assertEqual(channel.jamming_budget, 6)
        self.assertEqual(len([event for event in channel.events if event.spoofed]), 5)
        self.assertTrue(all(event.word in VOCABULARY for event in channel.events))

    def test_evader_handbrake_rotates_more_and_sheds_speed(self):
        def run(car: str):
            veh = Vehicle(get_car(car))
            veh.reset(speed=24.0)
            for _ in range(180):
                veh.step(Controls(steer=1.0, throttle=0.35, handbrake=1.0, clutch=1.0))
            return veh

        evader = run("evader")
        pursuer = run("pursuer")
        self.assertGreater(abs(evader.yaw), abs(pursuer.yaw) * 1.05)
        self.assertLess(evader.speed, pursuer.speed * 0.6)
        self.assertTrue(math.isfinite(evader.x))
        self.assertTrue(math.isfinite(pursuer.x))

    def test_physics_validation_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "physics_validation.json"
            manifest = run_physics_validation(
                seed=11,
                turn_steps=180,
                trace_steps=120,
                long_steps=240,
                out=out,
            )
            self.assertTrue(out.exists())
            self.assertTrue(manifest["passed"])
            self.assertEqual(manifest["checks_passed"], manifest["checks"])
            by_name = {row["name"]: row for row in manifest["results"]}
            self.assertGreater(by_name["handbrake_asymmetry"]["metrics"]["yaw_ratio"], 1.08)
            self.assertGreater(by_name["handbrake_asymmetry"]["metrics"]["radius_ratio"], 1.35)
            self.assertTrue(by_name["collision_recovery"]["metrics"]["clear_of_padded_building"])

    def test_spectator_validation_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "spectator_validation.json"
            manifest = run_spectator_validation(
                replay="replays/control_acceptance_active/downtown_chase_seed11.jsonl",
                pygame_frames=2,
                out=out,
            )
            self.assertTrue(out.exists())
            self.assertTrue(manifest["passed"])
            self.assertEqual(manifest["checks_passed"], manifest["checks"])
            by_name = {row["name"]: row for row in manifest["results"]}
            buckets = [plan["bucket"] for plan in by_name["audio_confidence_mapping"]["metrics"]["plans"]]
            self.assertEqual(buckets, ["low", "mid", "high"])
            self.assertTrue(by_name["web_replay_payload_contract"]["metrics"]["has_predictions"])

    def test_operational_validation_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "operational_readiness.json"
            manifest = run_operational_validation(out=out)
            self.assertTrue(out.exists())
            self.assertTrue(manifest["passed"])
            self.assertEqual(manifest["checks_passed"], manifest["checks"])
            by_name = {row["name"]: row for row in manifest["results"]}
            self.assertTrue(by_name["scenario_acceptance_gates"]["passed"])
            self.assertTrue(by_name["marl_environment_contract"]["passed"])
            self.assertTrue(by_name["scenario_gated_self_play"]["passed"])
            self.assertTrue(by_name["differentiable_english_token_comms"]["passed"])
            self.assertEqual(
                by_name["differentiable_english_token_comms"]["metrics"]["message_len"],
                6,
            )

    def test_evidence_bundle_contract(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "evidence_bundle.json"
            bundle = build_evidence_bundle(out=out)
            self.assertTrue(out.exists())
            self.assertEqual(bundle["kind"], "crypt_heist_evidence_bundle")
            self.assertTrue(bundle["passed"])
            self.assertTrue(bundle["command_registry"]["exists"])
            self.assertIn("configs/commands.json", bundle["command_registry"]["path"])
            self.assertGreaterEqual(bundle["summary"]["operational_checks_passed"], 11)
            self.assertEqual(bundle["summary"]["active_required_checks_passed"], 16)
            self.assertEqual(bundle["summary"]["checkpoints_existing"], bundle["summary"]["checkpoints_expected"])
            self.assertGreaterEqual(bundle["summary"]["active_replay_reports"], 4)
            self.assertEqual(
                bundle["summary"]["active_replay_reports_nominal"],
                bundle["summary"]["active_replay_reports"],
            )
            first = bundle["active_replays"][0]
            self.assertTrue(first["valid"])
            self.assertIn("state_sha256", first["summary"])
            self.assertIn("score", first["report"])
            verification = verify_evidence_bundle(out, out=Path(td) / "verification.json")
            self.assertTrue(verification["passed"])
            self.assertEqual(verification["checks_passed"], verification["checks"])

            tampered_path = Path(td) / "tampered_bundle.json"
            tampered = json.loads(out.read_text(encoding="utf-8"))
            first_checkpoint = next(iter(tampered["checkpoints"].values()))
            first_checkpoint["sha256"] = "0" * 64
            tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
            tampered_verification = verify_evidence_bundle(tampered_path)
            self.assertFalse(tampered_verification["passed"])
            self.assertTrue(tampered_verification["failures"])

    def test_v2_research_report_contract(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            league_path = root / "league.json"
            replay_path = root / "replay_report.json"
            spectator_path = root / "spectator.json"
            operational_path = root / "operational.json"
            evidence_path = root / "evidence_verification.json"
            out = root / "v2_report.json"
            markdown = root / "v2_report.md"
            league_path.write_text(json.dumps({
                "seeds": [11, 12, 13, 14, 15],
                "steps": 1800,
                "scores": {
                    "overall_score": 0.71,
                    "pursuer_security_score": 0.73,
                    "evader_pressure_score": 0.60,
                    "adversarial_balance_score": 0.86,
                },
                "aggregates": {
                    "mean_spoof_susceptibility": 0.43,
                    "mean_deception_lift": 2.0,
                    "mean_confidence_damage": 0.03,
                    "mean_spoof_event_lift": 10.0,
                    "mean_pursuer_deviation": 45.0,
                },
                "per_seed": [
                    {"metrics": {"spoof_susceptibility_counterfactual": value}}
                    for value in (0.41, 0.42, 0.43, 0.44, 0.45)
                ],
            }), encoding="utf-8")
            replay_path.write_text(json.dumps({
                "score": {"status": "nominal", "overall": 0.96},
                "sections": {
                    "chase": {"waypoints_hit": 5},
                    "information_warfare": {"jam_events": 1, "radio_events": 500},
                },
            }), encoding="utf-8")
            spectator_path.write_text(json.dumps({
                "passed": True,
                "checks_passed": 6,
                "checks": 6,
                "score": 1.0,
                "results": [
                    {"name": "audio_confidence_mapping", "passed": True},
                    {"name": "web_cockpit_static_contract", "passed": True},
                    {"name": "threejs_replay_renderer_contract", "passed": True},
                    {"name": "web_replay_payload_contract", "passed": True},
                    {"name": "dashboard_replay_deep_links", "passed": True},
                ],
            }), encoding="utf-8")
            operational_path.write_text(json.dumps({
                "passed": True,
                "checks_passed": 11,
                "checks": 11,
                "score": 1.0,
            }), encoding="utf-8")
            evidence_path.write_text(json.dumps({
                "passed": True,
                "checks_passed": 38,
                "checks": 38,
                "failures": [],
            }), encoding="utf-8")
            report = build_v2_research_report(
                league_manifest=league_path,
                replay_report=replay_path,
                spectator_validation=spectator_path,
                operational_readiness=operational_path,
                evidence_verification=evidence_path,
                out=out,
                markdown_out=markdown,
            )
            self.assertTrue(out.exists())
            self.assertTrue(markdown.exists())
            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "research_grade_candidate")
            self.assertEqual(report["checks_passed"], report["checks"])
            self.assertIn("Cryptographic Heist Engine V2", markdown.read_text(encoding="utf-8"))

    def test_marl_policy_shapes(self):
        pursuer = PursuerPolicy(OBS_SIZE, vocab_size=len(VOCABULARY), hidden=64)
        pout = pursuer(torch.zeros(2, 3, OBS_SIZE), agent_id=torch.tensor([0, 1]))
        self.assertEqual(tuple(pout["control"].shape), (2, 3))
        self.assertEqual(tuple(pout["token_logits"].shape), (2, 6, len(VOCABULARY)))
        self.assertEqual(tuple(pout["relaxed_tokens"].shape), (2, 6, len(VOCABULARY)))
        self.assertEqual(tuple(pout["token_ids"].shape), (2, 6))
        self.assertTrue(torch.allclose(pout["relaxed_tokens"].sum(dim=-1), torch.ones(2, 6)))
        class_weights = torch.linspace(0.0, 1.0, len(VOCABULARY))
        loss = (pout["relaxed_tokens"] * class_weights).sum()
        loss.backward()
        self.assertIsNotNone(pursuer.words.weight.grad)
        self.assertGreater(float(pursuer.words.weight.grad.abs().sum()), 0.0)
        with self.assertRaises(ValueError):
            pursuer(torch.zeros(1, 1, OBS_SIZE), tau=0.0)
        evader = EvaderPolicy(OBS_SIZE, len(VOCABULARY), hidden=64)
        eout = evader(torch.zeros(2, OBS_SIZE), torch.zeros(2, 3, len(VOCABULARY)))
        self.assertEqual(tuple(eout["future_xy"].shape), (2, 5, 2))
        self.assertEqual(tuple(eout["spoof_logits"].shape), (2, 5, len(VOCABULARY)))

    def test_curriculum_order(self):
        self.assertEqual(
            next_stage(CurriculumStage.SCRIPTED_SANDBOX),
            CurriculumStage.EVADER_WAYPOINTS,
        )

    def test_acceptance_diagnostics_route_to_curriculum_actions(self):
        manifest = {
            "diagnostics": {
                "top": [
                    {
                        "scenario": "downtown_chase",
                        "owner": "evader",
                        "reason": "waypoint_milestone_gap",
                        "severity": "medium",
                        "evidence": {"waypoints_hit": 1, "final_waypoint_distance": 42.0},
                    },
                    {
                        "scenario": "spoof_burst",
                        "owner": "information_warfare",
                        "reason": "jam_not_deceptive",
                        "severity": "high",
                        "evidence": {"deception_score": 0.0},
                    },
                ]
            },
            "scenarios": [],
        }
        plan = plan_curriculum_from_manifest(manifest, source="logs/acceptance_scenarios.json")
        self.assertEqual(plan["status"], "needs_training")
        commands = {action["command_id"] for action in plan["actions"]}
        self.assertIn("train_evader_ppo", commands)
        self.assertIn("train_counterfactual_jammer_policy", commands)
        top = plan["actions"][0]
        self.assertEqual(top["lane"], "counterfactual_jammer")
        self.assertIn("jam_not_deceptive", top["diagnostic_reasons"])

        nominal = plan_curriculum_from_manifest({"diagnostics": {"top": []}, "scenarios": []})
        self.assertEqual(nominal["status"], "nominal")
        self.assertEqual(nominal["actions"][0]["command_id"], "run_self_play_cycle_scenario_gated")

    def test_information_reward_terms_shape_training_rewards(self):
        sim = HeistSim(seed=8)
        sim.step()
        sim.decoder_predictions = {
            p.name: (float(p.vehicle.x), float(p.vehicle.y))
            for p in sim.pursuers
        }
        before = RewardSnapshot.from_sim(sim)
        after = RewardSnapshot.from_sim(sim)
        rewards, components = RewardSystem(
            weights=RewardWeights(
                decoder_accuracy_alpha=0.25,
                spoof_susceptibility_gamma=0.50,
                evader_spoof_reward=0.35,
            ),
            prediction_horizon_steps=0,
        ).compute(before, after, done=False)
        self.assertGreater(components["decoder_accuracy"], 0.98)
        self.assertGreater(components["decoder_accuracy_penalty"], 0.20)
        self.assertGreater(components["pursuer_auth_penalty"], 0.20)
        self.assertLess(rewards["pursuer_0"], 0.0)

        sim.pursuers[0].spoof_timer = 1.0
        sim.pursuers[0].spoof_offset = np.asarray([85.0, 0.0], dtype=np.float32)
        spoofed = RewardSnapshot.from_sim(sim)
        rewards, components = RewardSystem(
            weights=RewardWeights(spoof_susceptibility_gamma=0.50, evader_spoof_reward=0.35),
            prediction_horizon_steps=0,
        ).compute(before, spoofed, done=False)
        self.assertGreater(components["spoof_susceptibility"], 0.95)
        self.assertGreater(components["spoof_susceptibility_penalty"], 0.45)
        self.assertGreater(components["evader_information_reward"], 0.30)
        self.assertGreater(rewards["evader_0"], 0.0)

    def test_imitation_checkpoint_round_trip(self):
        obs, actions = collect_evader_dataset(seed=9, steps=12)
        self.assertEqual(obs.shape, (12, OBS_SIZE))
        self.assertEqual(actions.shape, (12, 3))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "evader.pt"
            result = train_evader_imitation(path, seed=9, steps=32, epochs=1, batch_size=16, hidden=32)
            self.assertTrue(path.exists())
            self.assertEqual(result.samples, 32)
            sim = HeistSim(seed=9)
            controller = EvaderCheckpointController(path)
            action = controller.action(sim)
            self.assertIn("evader_0", action)
            self.assertEqual(tuple(action["evader_0"]["control"].shape), (3,))

    def test_ppo_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "evader_ppo.pt"
            result = train_evader_ppo(
                path,
                seed=12,
                updates=1,
                steps_per_update=16,
                train_epochs=1,
                minibatch_size=8,
                max_cycles=16,
                control_repeat=2,
                hidden=32,
            )
            self.assertTrue(path.exists())
            self.assertEqual(result.steps, 16)
            self.assertIn("counterfactual_deception", result.stats[-1])
            sim = HeistSim(seed=12)
            controller = EvaderCheckpointController(path)
            self.assertEqual(controller.metadata["kind"], "evader_ppo")
            action = controller.action(sim)
            self.assertIn("evader_0", action)
            self.assertEqual(tuple(action["evader_0"]["control"].shape), (3,))

    def test_ppo_can_warm_start_from_imitation_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            imitation_path = root / "evader_imitation.pt"
            ppo_path = root / "evader_ppo.pt"
            train_evader_imitation(
                imitation_path,
                seed=18,
                steps=32,
                epochs=1,
                batch_size=16,
                hidden=24,
            )
            result = train_evader_ppo(
                ppo_path,
                seed=18,
                updates=1,
                steps_per_update=8,
                train_epochs=1,
                minibatch_size=4,
                max_cycles=8,
                control_repeat=1,
                hidden=24,
                init_checkpoint=imitation_path,
                validation_steps=16,
                counterfactual_interval=0,
            )
            self.assertTrue(ppo_path.exists())
            self.assertEqual(result.steps, 8)
            controller = EvaderCheckpointController(ppo_path)
            self.assertEqual(controller.metadata["init"]["kind"], "evader_imitation")
            self.assertEqual(controller.metadata["config"]["init_checkpoint"], str(imitation_path))
            self.assertIn("actor_mean.0.weight", controller.metadata["init"]["loaded_tensors"])
            self.assertIsNotNone(controller.metadata["validation"])
            self.assertIn("score", controller.metadata["validation"])

    def test_checkpoint_acceptance_can_use_learned_evader_only(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "evader_imitation.pt"
            train_evader_imitation(path, seed=19, steps=24, epochs=1, batch_size=8, hidden=24)
            manifest = run_checkpoint_acceptance_suite(
                evader_checkpoint=path,
                scenarios=["downtown_chase"],
                seed=19,
                steps=12,
                record_dir=None,
                out=None,
                thresholds=ScenarioThresholds(downtown_min_waypoints=0),
            )
            self.assertEqual(manifest["requested_scenarios"], ["downtown_chase"])
            self.assertEqual(len(manifest["scenarios"]), 1)
            self.assertIn("evader_control_score", manifest["score"])

    def test_pursuer_ppo_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pursuer_ppo.pt"
            result = train_pursuer_ppo(
                path,
                agent_name="pursuer_0",
                seed=13,
                updates=1,
                steps_per_update=16,
                train_epochs=1,
                minibatch_size=8,
                max_cycles=16,
                control_repeat=2,
                hidden=32,
            )
            self.assertTrue(path.exists())
            self.assertEqual(result.steps, 16)
            sim = HeistSim(seed=13)
            controller = PursuerCheckpointController(path)
            self.assertEqual(controller.metadata["kind"], "pursuer_ppo")
            self.assertEqual(controller.agent_name, "pursuer_0")
            action = controller.action(sim)
            self.assertIn("pursuer_0", action)
            self.assertEqual(tuple(action["pursuer_0"]["control"].shape), (3,))

    def test_pursuer_team_ppo_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pursuer_team_ppo.pt"
            result = train_pursuer_team_ppo(
                path,
                seed=14,
                updates=1,
                steps_per_update=8,
                train_epochs=1,
                minibatch_size=10,
                max_cycles=8,
                control_repeat=2,
                hidden=32,
            )
            self.assertTrue(path.exists())
            self.assertEqual(result.steps, 40)
            sim = HeistSim(seed=14)
            controller = PursuerTeamCheckpointController(path)
            self.assertEqual(controller.metadata["kind"], "pursuer_team_ppo")
            self.assertEqual(len(controller.agent_names), 5)
            action = controller.action(sim)
            self.assertEqual(set(action), {f"pursuer_{i}" for i in range(5)})
            self.assertEqual(tuple(action["pursuer_0"]["control"].shape), (3,))

    def test_ppo_can_train_against_checkpoint_opponent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evader_path = root / "evader_ppo.pt"
            pursuer_path = root / "pursuer_vs_evader.pt"
            train_evader_ppo(
                evader_path,
                seed=22,
                updates=1,
                steps_per_update=8,
                train_epochs=1,
                minibatch_size=4,
                max_cycles=8,
                control_repeat=1,
                hidden=24,
                counterfactual_interval=0,
            )
            result = train_pursuer_ppo(
                pursuer_path,
                agent_name="pursuer_0",
                seed=23,
                updates=1,
                steps_per_update=8,
                train_epochs=1,
                minibatch_size=4,
                max_cycles=8,
                control_repeat=1,
                hidden=24,
                counterfactual_interval=0,
                opponent_evader_checkpoint=evader_path,
            )
            self.assertTrue(pursuer_path.exists())
            self.assertEqual(result.stats[-1]["opponent_count"], 1)
            controller = PursuerCheckpointController(pursuer_path)
            self.assertEqual(controller.metadata["config"]["opponent_evader_checkpoint"], str(evader_path))

    def test_scanner_decoder_checkpoint_round_trip(self):
        tokens, targets = collect_scanner_dataset(seed=15, steps=24, horizon_steps=4, window=8, sample_every=2)
        self.assertEqual(tokens.shape, (12, 8))
        self.assertEqual(targets.shape, (12, 5, 2))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "scanner.pt"
            result = train_scanner_decoder(
                path,
                seed=15,
                steps=32,
                horizon_steps=4,
                window=8,
                sample_every=2,
                epochs=1,
                batch_size=8,
                embed=8,
                hidden=16,
            )
            self.assertTrue(path.exists())
            self.assertEqual(result.samples, 16)
            sim = HeistSim(seed=15)
            runtime = ScannerDecoderRuntime(path)
            sim.step()
            predictions = runtime.apply(sim)
            self.assertEqual(set(predictions), {f"P{i + 1}" for i in range(5)})
            self.assertTrue(0.0 <= sim.channel.confidence <= 1.0)

    def test_radio_policy_checkpoint_round_trip(self):
        obs, agent_ids, tokens = collect_radio_dataset(seed=16, steps=120)
        self.assertEqual(obs.shape[1], OBS_SIZE)
        self.assertEqual(agent_ids.ndim, 1)
        self.assertEqual(tokens.shape[1], 6)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "radio.pt"
            result = train_radio_policy(
                path,
                seed=16,
                steps=120,
                epochs=1,
                batch_size=4,
                hidden=32,
            )
            self.assertTrue(path.exists())
            self.assertGreaterEqual(result.samples, 2)
            model, metadata = load_radio_policy(path)
            self.assertEqual(metadata["kind"], "pursuer_radio_policy")
            sample_obs = torch.as_tensor(obs[:2], dtype=torch.float32)
            sample_ids = torch.as_tensor(agent_ids[:2], dtype=torch.long)
            relaxed = model.relaxed_tokens(sample_obs, sample_ids, tau=0.9, hard=True)
            self.assertEqual(tuple(relaxed.shape), (2, 6, len(VOCABULARY)))
            self.assertTrue(torch.allclose(relaxed.sum(dim=-1), torch.ones(2, 6)))
            class_weights = torch.linspace(0.0, 1.0, len(VOCABULARY))
            loss = (relaxed * class_weights).sum()
            loss.backward()
            self.assertIsNotNone(model.net[-1].weight.grad)
            self.assertGreater(float(model.net[-1].weight.grad.abs().sum()), 0.0)
            with self.assertRaises(ValueError):
                model.relaxed_tokens(sample_obs, sample_ids, tau=0.0)
            sim = HeistSim(seed=16)
            controller = PursuerRadioController(path)
            action = controller.action(sim)
            self.assertEqual(set(action), {f"pursuer_{i}" for i in range(5)})
            sim.step(actions=action)
            self.assertGreaterEqual(len(sim.channel.events), 6)

    def test_jammer_policy_checkpoint_round_trip(self):
        obs, triggers, spoof, targets = collect_jammer_dataset(seed=17, steps=48)
        self.assertEqual(obs.shape[1], OBS_SIZE)
        self.assertEqual(triggers.shape, (48,))
        self.assertEqual(spoof.shape, (48, 5))
        self.assertEqual(targets.shape, (48, 5))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "jammer.pt"
            result = train_jammer_policy(
                path,
                seed=17,
                steps=64,
                epochs=1,
                batch_size=16,
                hidden=32,
            )
            self.assertTrue(path.exists())
            self.assertEqual(result.samples, 64)
            model, metadata = load_jammer_policy(path)
            self.assertEqual(metadata["kind"], "evader_jammer_policy")
            sample_obs = torch.as_tensor(obs[:2], dtype=torch.float32)
            relaxed = model.relaxed_spoof_tokens(sample_obs, tau=0.9, hard=True)
            self.assertEqual(tuple(relaxed.shape), (2, 5, len(VOCABULARY)))
            sim = HeistSim(seed=17)
            sim.jam_cooldown = 0.0
            controller = EvaderJammerController(path)
            action = controller.action(sim)
            self.assertIn("evader_0", action)
            self.assertEqual(tuple(action["evader_0"]["spoof_tokens"].shape), (5,))
            sim.step(actions=action)
            self.assertTrue(0 <= sim.channel.jamming_budget <= 7)

    def test_counterfactual_jammer_checkpoint_round_trip(self):
        obs, triggers, spoof, targets, weights, rewards = collect_counterfactual_jammer_dataset(
            seed=21,
            steps=8,
            sample_every=4,
            horizon_steps=4,
            positive_threshold=0.01,
        )
        self.assertEqual(obs.shape[1], OBS_SIZE)
        self.assertEqual(triggers.shape, (2,))
        self.assertEqual(spoof.shape, (2, 5))
        self.assertEqual(targets.shape, (2, 5))
        self.assertEqual(weights.shape, (2,))
        self.assertEqual(rewards.shape, (2,))
        self.assertTrue(np.all(weights >= 1.0))
        self.assertTrue(np.all((rewards >= 0.0) & (rewards <= 1.0)))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "jammer_counterfactual.pt"
            result = train_counterfactual_jammer_policy(
                path,
                seed=21,
                steps=12,
                sample_every=4,
                horizon_steps=4,
                positive_threshold=0.01,
                epochs=1,
                batch_size=2,
                hidden=24,
            )
            self.assertTrue(path.exists())
            self.assertEqual(result.samples, 3)
            self.assertEqual(result.training_mode, "counterfactual")
            self.assertGreaterEqual(result.mean_counterfactual_reward, 0.0)
            model, metadata = load_jammer_policy(path)
            self.assertEqual(metadata["kind"], "evader_jammer_policy")
            self.assertEqual(metadata["training_mode"], "counterfactual")
            sim = HeistSim(seed=21)
            controller = EvaderJammerController(path)
            action = controller.action(sim)
            self.assertIn("evader_0", action)
            self.assertEqual(tuple(action["evader_0"]["spoof_tokens"].shape), (5,))

    def test_adversarial_stack_records_replay_and_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            scanner_path = root / "scanner.pt"
            radio_path = root / "radio.pt"
            jammer_path = root / "jammer.pt"
            replay_path = root / "adversarial.jsonl"
            train_scanner_decoder(
                scanner_path,
                seed=18,
                steps=20,
                horizon_steps=3,
                window=6,
                sample_every=2,
                epochs=1,
                batch_size=4,
                embed=8,
                hidden=24,
            )
            train_radio_policy(
                radio_path,
                seed=18,
                steps=70,
                epochs=1,
                batch_size=4,
                hidden=24,
            )
            train_jammer_policy(
                jammer_path,
                seed=18,
                steps=36,
                epochs=1,
                batch_size=8,
                hidden=24,
            )
            result = run_adversarial_eval(
                radio_checkpoint=radio_path,
                scanner_checkpoint=scanner_path,
                jammer_checkpoint=jammer_path,
                seed=18,
                steps=72,
                record=replay_path,
            )
            self.assertTrue(replay_path.exists())
            replay = load_replay(replay_path)
            validate_replay(replay)
            metrics = result.metrics
            self.assertEqual(metrics["frames"], 72)
            self.assertGreater(metrics["unique_radio_events"], 0)
            self.assertIn("avg_decoder_error", metrics)
            self.assertIn("spoof_susceptibility_proxy", metrics)
            self.assertIn("unique_spoofed_radio_events", metrics["replay_summary"])
            paired = run_authentication_curriculum_eval(
                radio_checkpoint=radio_path,
                scanner_checkpoint=scanner_path,
                jammer_checkpoint=jammer_path,
                seed=18,
                steps=48,
                record_prefix=root / "auth_pair",
            )
            self.assertTrue((root / "auth_pair_jammed.jsonl").exists())
            self.assertTrue((root / "auth_pair_baseline.jsonl").exists())
            self.assertGreaterEqual(paired.metrics["spoof_event_lift"], 0)
            self.assertEqual(paired.baseline.metrics["unique_spoofed_events"], 0)
            self.assertIn("mean_pursuer_deviation", paired.metrics["trajectory"])
            self.assertIn("pursuer_auth_penalty_proxy", paired.metrics)
            league = evaluate_checkpoint_set(
                name="tiny",
                radio=radio_path,
                scanner=scanner_path,
                jammer=jammer_path,
                seeds=[18],
                steps=36,
                record_prefix=root / "league",
                out=root / "league_manifest.json",
                promotion_threshold=0.0,
                promote=True,
                active_dir=root / "active",
            )
            self.assertTrue((root / "league_manifest.json").exists())
            self.assertTrue((root / "active" / "manifest.json").exists())
            self.assertTrue((root / "active" / "radio_policy.pt").exists())
            self.assertIn("overall_score", league.scores)
            self.assertTrue(league.promoted)

    def test_information_cycle_trains_scores_and_promotes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = run_information_cycle(
                name="cycle tiny",
                seed=19,
                checkpoint_dir=root / "checkpoints",
                log_dir=root / "logs",
                scanner_steps=20,
                scanner_epochs=1,
                scanner_batch_size=4,
                scanner_horizon_steps=3,
                scanner_window=6,
                scanner_sample_every=2,
                scanner_embed=8,
                radio_steps=70,
                radio_epochs=1,
                radio_batch_size=4,
                jammer_steps=36,
                jammer_epochs=1,
                jammer_batch_size=8,
                hidden=24,
                league_seeds=[19],
                league_steps=36,
                promotion_threshold=0.0,
                promote=True,
                active_dir=root / "active",
            )
            self.assertTrue(Path(result.manifest).exists())
            self.assertTrue((root / "active" / "manifest.json").exists())
            self.assertTrue(Path(result.checkpoints["scanner"]).exists())
            self.assertIn("scanner", result.training)
            self.assertIn("overall_score", result.league.scores)
            self.assertTrue(result.league.promoted)
            self.assertEqual(result.promotion_decision["reason"], "first_active_stack")
            blocked = run_information_cycle(
                name="cycle blocked",
                seed=20,
                checkpoint_dir=root / "checkpoints_blocked",
                log_dir=root / "logs_blocked",
                scanner_steps=20,
                scanner_epochs=1,
                scanner_batch_size=4,
                scanner_horizon_steps=3,
                scanner_window=6,
                scanner_sample_every=2,
                scanner_embed=8,
                radio_steps=70,
                radio_epochs=1,
                radio_batch_size=4,
                jammer_steps=36,
                jammer_epochs=1,
                jammer_batch_size=8,
                hidden=24,
                league_seeds=[20],
                league_steps=36,
                promotion_threshold=0.0,
                improvement_margin=2.0,
                promote=True,
                active_dir=root / "active",
            )
            self.assertTrue(blocked.promotion_decision["incumbent_found"])
            self.assertEqual(blocked.promotion_decision["reason"], "did_not_improve_over_incumbent")
            self.assertFalse(blocked.league.promoted)

    def test_self_play_cycle_trains_and_promotes_control_checkpoints(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = run_self_play_cycle(
                name="self tiny",
                seed=24,
                checkpoint_dir=root / "checkpoints",
                log_dir=root / "logs",
                active_control_dir=root / "control_active",
                control_pool_dir=root / "control_pool",
                active_info_dir=root / "info_active",
                evader_updates=1,
                team_updates=1,
                steps_per_update=6,
                max_cycles=6,
                control_repeat=1,
                train_epochs=1,
                evader_minibatch_size=3,
                team_minibatch_size=6,
                hidden=24,
                counterfactual_interval=0,
                control_eval_seeds=[24],
                control_eval_steps=8,
                control_record_prefix=root / "replays" / "selfplay",
                train_information=False,
            )
            self.assertTrue(Path(result.manifest).exists())
            self.assertTrue(Path(result.checkpoints["evader"]).exists())
            self.assertTrue(Path(result.checkpoints["pursuer_team"]).exists())
            self.assertIsNotNone(result.control_evaluation)
            self.assertIn("overall_score", result.control_evaluation["scores"])
            self.assertIsNotNone(result.control_league)
            self.assertIn("worst_case_score", result.control_league["scores"])
            self.assertEqual(result.control_league["scores"]["pool_depth_score"], 0.0)
            self.assertTrue(result.control_promotion_decision["promoted"])
            self.assertEqual(result.control_promotion_decision["reason"], "first_active_control_pair")
            self.assertTrue(Path(result.control_evaluation["manifest"]).exists())
            self.assertTrue((root / "replays" / "selfplay_self_tiny_seed24.jsonl").exists())
            self.assertIsNotNone(result.control_pool_entry)
            self.assertTrue(Path(result.control_pool_entry["evader"]).exists())
            self.assertTrue(Path(result.control_pool_entry["pursuer_team"]).exists())
            pool = load_control_pool(root / "control_pool")
            self.assertEqual(len(pool), 1)
            sampled = select_control_opponent(
                pool_dir=root / "control_pool",
                kind="pursuer_team",
                seed=99,
            )
            self.assertEqual(sampled.source, "pool")
            self.assertTrue(Path(sampled.checkpoint).exists())
            self.assertIsNone(result.information_cycle)
            active = load_active_control_set(root / "control_active")
            self.assertTrue(Path(active.evader).exists())
            self.assertTrue(Path(active.pursuer_team).exists())
            self.assertIn("evader", result.training)
            self.assertIn("pursuer_team", result.training)

            next_result = run_self_play_cycle(
                name="self tiny two",
                seed=34,
                checkpoint_dir=root / "checkpoints_two",
                log_dir=root / "logs_two",
                active_control_dir=root / "control_active",
                control_pool_dir=root / "control_pool",
                active_info_dir=root / "info_active",
                evader_updates=1,
                team_updates=1,
                steps_per_update=4,
                evader_init_checkpoint=result.checkpoints["evader"],
                evader_validation_steps=2,
                evader_validation_seed=34,
                max_cycles=4,
                control_repeat=1,
                train_epochs=1,
                evader_minibatch_size=2,
                team_minibatch_size=5,
                hidden=24,
                counterfactual_interval=0,
                evaluate_control=False,
                train_information=False,
            )
            self.assertEqual(next_result.control_opponents["evader_training"]["source"], "pool")
            self.assertEqual(next_result.control_opponents["pursuer_team_training"]["source"], "pool")
            self.assertTrue(next_result.control_promotion_decision["promoted"])
            self.assertEqual(next_result.control_promotion_decision["reason"], "control_evaluation_skipped")
            self.assertEqual(len(load_control_pool(root / "control_pool")), 2)
            evader_ckpt = torch.load(next_result.checkpoints["evader"], map_location="cpu", weights_only=False)
            self.assertEqual(evader_ckpt["init"]["path"], result.checkpoints["evader"])
            self.assertEqual(evader_ckpt["validation"]["steps"], 2)
            self.assertEqual(evader_ckpt["validation"]["seed"], 34)

    def test_control_promotion_decision_can_advance_one_faction(self):
        pool_entry = type("PoolEntry", (), {})()
        pool_entry.control_evaluation = {
            "scores": {
                "overall_score": 0.50,
                "evader_generalization_score": 0.35,
                "team_resilience_score": 0.80,
            }
        }
        fake_league = type("League", (), {
            "to_dict": lambda self: {
                "scores": {
                    "overall_score": 0.58,
                    "evader_generalization_score": 0.72,
                    "team_resilience_score": 0.70,
                }
            }
        })()
        decision = _control_promotion_decision(
            promote_control=True,
            evaluate_control=True,
            active_before=ControlCheckpointSet(evader="active_evader.pt", pursuer_team="active_team.pt"),
            pool_before=[pool_entry],
            control_result=None,
            control_league=fake_league,
            threshold=0.0,
            improvement_margin=0.05,
            evader_threshold=None,
            evader_margin=None,
            team_threshold=None,
            team_margin=None,
        )
        self.assertTrue(decision["promoted"])
        self.assertTrue(decision["evader_promoted"])
        self.assertFalse(decision["pursuer_team_promoted"])
        self.assertEqual(decision["reason"], "evader_improved_over_pool")
        self.assertGreater(decision["evader_candidate_score"], decision["evader_required_score"])
        self.assertLess(decision["pursuer_team_candidate_score"], decision["pursuer_team_required_score"])

        stricter = _control_promotion_decision(
            promote_control=True,
            evaluate_control=True,
            active_before=ControlCheckpointSet(evader="active_evader.pt", pursuer_team="active_team.pt"),
            pool_before=[pool_entry],
            control_result=None,
            control_league=fake_league,
            threshold=0.0,
            improvement_margin=0.0,
            evader_threshold=0.75,
            evader_margin=0.0,
            team_threshold=0.0,
            team_margin=0.05,
        )
        self.assertFalse(stricter["evader_promoted"])
        self.assertFalse(stricter["pursuer_team_promoted"])
        self.assertEqual(stricter["reason"], "no_control_component_improved")
        self.assertEqual(stricter["evader_threshold"], 0.75)

        incomplete = _control_promotion_decision(
            promote_control=True,
            evaluate_control=True,
            active_before=ControlCheckpointSet(),
            pool_before=[],
            control_result=None,
            control_league=fake_league,
            threshold=0.0,
            improvement_margin=0.0,
            evader_threshold=0.0,
            evader_margin=0.0,
            team_threshold=0.95,
            team_margin=0.0,
        )
        self.assertFalse(incomplete["promoted"])
        self.assertEqual(incomplete["reason"], "below_component_threshold")

        scenario_blocked = type("League", (), {
            "to_dict": lambda self: {
                "scores": {
                    "overall_score": 0.95,
                    "evader_generalization_score": 0.95,
                    "team_resilience_score": 0.95,
                    "scenario_acceptance_score": 0.40,
                }
            }
        })()
        blocked = _control_promotion_decision(
            promote_control=True,
            evaluate_control=True,
            active_before=ControlCheckpointSet(evader="active_evader.pt", pursuer_team="active_team.pt"),
            pool_before=[],
            control_result=None,
            control_league=scenario_blocked,
            threshold=0.0,
            improvement_margin=0.0,
            scenario_threshold=0.75,
        )
        self.assertFalse(blocked["promoted"])
        self.assertEqual(blocked["reason"], "below_scenario_acceptance_threshold")
        self.assertEqual(blocked["scenario_acceptance_score"], 0.40)

        failed_manifest_gate = type("League", (), {
            "to_dict": lambda self: {
                "scores": {
                    "overall_score": 0.95,
                    "evader_generalization_score": 0.95,
                    "team_resilience_score": 0.95,
                    "scenario_acceptance_score": 0.95,
                    "scenario_evader_acceptance_score": 0.95,
                    "scenario_team_acceptance_score": 0.95,
                },
                "scenario_acceptance": {
                    "passed": False,
                    "required_checks_passed": 15,
                    "required_checks": 16,
                    "milestone_targets_passed": 0,
                    "milestone_targets": 1,
                    "score": {
                        "required_score": 0.9375,
                        "milestone_score": 0.0,
                    },
                },
            }
        })()
        failed_manifest_decision = _control_promotion_decision(
            promote_control=True,
            evaluate_control=True,
            active_before=ControlCheckpointSet(evader="active_evader.pt", pursuer_team="active_team.pt"),
            pool_before=[],
            control_result=None,
            control_league=failed_manifest_gate,
            threshold=0.0,
            improvement_margin=0.0,
            evader_scenario_threshold=0.75,
            team_scenario_threshold=0.75,
        )
        self.assertFalse(failed_manifest_decision["promoted"])
        self.assertEqual(failed_manifest_decision["reason"], "scenario_acceptance_failed")
        self.assertEqual(failed_manifest_decision["scenario_required_checks_passed"], 15)
        self.assertEqual(failed_manifest_decision["scenario_required_checks"], 16)

        split_gate = type("League", (), {
            "to_dict": lambda self: {
                "scores": {
                    "overall_score": 0.95,
                    "evader_generalization_score": 0.95,
                    "team_resilience_score": 0.95,
                    "scenario_acceptance_score": 0.85,
                    "scenario_evader_acceptance_score": 0.40,
                    "scenario_team_acceptance_score": 0.92,
                }
            }
        })()
        split_decision = _control_promotion_decision(
            promote_control=True,
            evaluate_control=True,
            active_before=ControlCheckpointSet(evader="active_evader.pt", pursuer_team="active_team.pt"),
            pool_before=[],
            control_result=None,
            control_league=split_gate,
            threshold=0.0,
            improvement_margin=0.0,
            evader_scenario_threshold=0.75,
            team_scenario_threshold=0.75,
        )
        self.assertTrue(split_decision["promoted"])
        self.assertFalse(split_decision["evader_promoted"])
        self.assertTrue(split_decision["pursuer_team_promoted"])
        self.assertEqual(split_decision["reason"], "pursuer_team_improved_over_pool")
        self.assertEqual(split_decision["evader_scenario_acceptance_score"], 0.40)
        self.assertEqual(split_decision["pursuer_team_scenario_acceptance_score"], 0.92)


if __name__ == "__main__":
    unittest.main()
