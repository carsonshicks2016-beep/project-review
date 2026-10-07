from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest

import command_center.server as server
from command_center.server import app
from crypt_heist.replay import record_scripted_replay


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_dashboard_index_and_registry(self):
        index = self.client.get("/")
        self.assertEqual(index.status_code, 200)
        registry = self.client.get("/api/commands")
        self.assertEqual(registry.status_code, 200)
        data = registry.get_json()
        ids = {cmd["id"] for cmd in data["commands"]}
        self.assertIn("viewer", ids)
        self.assertIn("record_replay", ids)
        self.assertIn("verify_replay_fidelity", ids)
        self.assertIn("replay_report", ids)
        self.assertIn("verify_physics", ids)
        self.assertIn("verify_env", ids)
        self.assertIn("verify_spectator", ids)
        self.assertIn("verify_operational_readiness", ids)
        self.assertIn("build_evidence_bundle", ids)
        self.assertIn("verify_evidence_bundle", ids)
        self.assertIn("train_smoke", ids)
        self.assertIn("train_evader_ppo", ids)
        self.assertIn("train_pursuer_ppo", ids)
        self.assertIn("train_pursuer_team_ppo", ids)
        self.assertIn("train_evader_ppo_self_play", ids)
        self.assertIn("train_pursuer_ppo_self_play", ids)
        self.assertIn("train_pursuer_team_ppo_self_play", ids)
        self.assertIn("train_scanner_decoder", ids)
        self.assertIn("train_radio_policy", ids)
        self.assertIn("train_jammer_policy", ids)
        self.assertIn("train_counterfactual_jammer_policy", ids)
        self.assertIn("run_information_cycle", ids)
        self.assertIn("run_self_play_cycle", ids)
        self.assertIn("eval_control_pair", ids)
        self.assertIn("eval_control_league", ids)
        self.assertIn("list_control_pool", ids)
        self.assertIn("watch_evader_checkpoint_live", ids)
        self.assertIn("watch_pursuer_checkpoint_live", ids)
        self.assertIn("watch_pursuer_team_checkpoint_live", ids)
        self.assertIn("watch_scanner_decoder_live", ids)
        self.assertIn("watch_radio_policy_live", ids)
        self.assertIn("watch_jammer_policy_live", ids)
        self.assertIn("watch_adversarial_stack", ids)
        self.assertIn("watch_adversarial_stack_live", ids)
        self.assertIn("eval_authentication_curriculum", ids)
        self.assertIn("eval_checkpoint_league", ids)
        self.assertIn("eval_v2_checkpoint_league", ids)
        self.assertIn("build_v2_report", ids)
        self.assertIn("eval_evader_checkpoint_acceptance", ids)
        self.assertIn("eval_pursuer_team_checkpoint_acceptance", ids)
        self.assertIn("eval_active_control_acceptance", ids)
        self.assertIn("plan_curriculum_from_acceptance", ids)
        self.assertIn("unit_tests", ids)

    def test_dashboard_lists_configs_and_replays(self):
        configs = self.client.get("/api/configs")
        self.assertEqual(configs.status_code, 200)
        names = {cfg["name"] for cfg in configs.get_json()}
        self.assertIn("commands.json", names)
        self.assertIn("simulation.json", names)
        replays = self.client.get("/api/replays")
        self.assertEqual(replays.status_code, 200)
        self.assertIsInstance(replays.get_json(), list)
        metrics = self.client.get("/api/metrics")
        self.assertEqual(metrics.status_code, 200)
        self.assertIn("avg_confidence", metrics.get_json())
        self.assertIn("avg_evader_reward", metrics.get_json())
        self.assertIn("avg_pursuer_auth_penalty", metrics.get_json())
        self.assertIn("avg_radio_word_entropy", metrics.get_json())
        self.assertIn("jam_events", metrics.get_json())
        self.assertIn("cipher_rotations", metrics.get_json())

    def test_dashboard_readiness_api_routes_next_step(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_acceptance = server.ACCEPTANCE_MANIFEST_PATH
            old_strict = server.STRICT_ACCEPTANCE_MANIFEST_PATH
            old_evader_acceptance = server.EVADER_CHECKPOINT_ACCEPTANCE_PATH
            old_pursuer_team_acceptance = server.PURSUER_TEAM_CHECKPOINT_ACCEPTANCE_PATH
            old_active_control_acceptance = server.ACTIVE_CONTROL_ACCEPTANCE_PATH
            old_checkpoint_league = server.CHECKPOINT_LEAGUE_MANIFEST_PATH
            old_active_information = server.ACTIVE_INFORMATION_MANIFEST_PATH
            old_replay_fidelity = server.REPLAY_FIDELITY_MANIFEST_PATH
            old_physics_validation = server.PHYSICS_VALIDATION_MANIFEST_PATH
            old_env_validation = server.ENV_VALIDATION_MANIFEST_PATH
            old_spectator_validation = server.SPECTATOR_VALIDATION_MANIFEST_PATH
            old_operational_validation = server.OPERATIONAL_READINESS_MANIFEST_PATH
            old_self_play = server.SELF_PLAY_MANIFEST_PATH
            old_plan = server.CURRICULUM_PLAN_PATH
            old_history = server.CURRICULUM_HISTORY_PATH
            old_checkpoints = server.READINESS_CHECKPOINTS
            server.ACCEPTANCE_MANIFEST_PATH = root / "acceptance.json"
            server.STRICT_ACCEPTANCE_MANIFEST_PATH = root / "strict.json"
            server.EVADER_CHECKPOINT_ACCEPTANCE_PATH = root / "evader_checkpoint_acceptance.json"
            server.PURSUER_TEAM_CHECKPOINT_ACCEPTANCE_PATH = root / "pursuer_team_checkpoint_acceptance.json"
            server.ACTIVE_CONTROL_ACCEPTANCE_PATH = root / "control_acceptance_active.json"
            server.CHECKPOINT_LEAGUE_MANIFEST_PATH = root / "checkpoint_league_manifest.json"
            server.ACTIVE_INFORMATION_MANIFEST_PATH = root / "active_information.json"
            server.REPLAY_FIDELITY_MANIFEST_PATH = root / "replay_fidelity.json"
            server.PHYSICS_VALIDATION_MANIFEST_PATH = root / "physics_validation.json"
            server.ENV_VALIDATION_MANIFEST_PATH = root / "env_validation.json"
            server.SPECTATOR_VALIDATION_MANIFEST_PATH = root / "spectator_validation.json"
            server.OPERATIONAL_READINESS_MANIFEST_PATH = root / "operational_readiness.json"
            server.SELF_PLAY_MANIFEST_PATH = root / "selfplay.json"
            server.CURRICULUM_PLAN_PATH = root / "curriculum_plan.json"
            server.CURRICULUM_HISTORY_PATH = root / "curriculum_history.json"
            existing_checkpoint = root / "evader_ppo.pt"
            existing_checkpoint.write_bytes(b"checkpoint")
            missing_checkpoint = root / "radio_policy.pt"
            server.READINESS_CHECKPOINTS = (
                {
                    "id": "evader_ppo",
                    "label": "Evader PPO driver",
                    "path": existing_checkpoint,
                    "command_id": "train_evader_ppo",
                    "lane": "evader_waypoint_driver",
                    "required_for": "learned control",
                },
                {
                    "id": "radio_policy",
                    "label": "Pursuer radio policy",
                    "path": missing_checkpoint,
                    "command_id": "train_radio_policy",
                    "lane": "radio_authentication_probe",
                    "required_for": "learned English-token comms",
                },
            )
            acceptance = {
                "passed": True,
                "required_checks_passed": 16,
                "required_checks": 16,
                "milestone_targets_passed": 0,
                "milestone_targets": 1,
                "score": {
                    "overall_score": 0.86,
                    "required_score": 1.0,
                    "milestone_score": 0.0,
                },
                "diagnostics": {
                    "count": 1,
                    "top": [
                        {
                            "scenario": "downtown_chase",
                            "severity": "medium",
                            "reason": "waypoint_milestone_gap",
                        }
                    ],
                },
                "scenarios": [],
            }
            plan = {
                "version": 1,
                "status": "needs_training",
                "diagnostic_count": 1,
                "summary": "Top diagnostic is downtown_chase:medium:waypoint_milestone_gap.",
                "actions": [
                    {
                        "lane": "evader_waypoint_driver",
                        "command_id": "train_evader_ppo",
                        "default_args": "--updates 1 --steps-per-update 8",
                    }
                ],
            }
            server.ACCEPTANCE_MANIFEST_PATH.write_text(json.dumps(acceptance), encoding="utf-8")
            server.EVADER_CHECKPOINT_ACCEPTANCE_PATH.write_text(json.dumps(acceptance), encoding="utf-8")
            server.PURSUER_TEAM_CHECKPOINT_ACCEPTANCE_PATH.write_text(json.dumps(acceptance), encoding="utf-8")
            server.ACTIVE_CONTROL_ACCEPTANCE_PATH.write_text(json.dumps(acceptance), encoding="utf-8")
            information = {
                "scores": {
                    "overall_score": 0.71,
                    "pursuer_security_score": 0.73,
                    "evader_pressure_score": 0.60,
                    "adversarial_balance_score": 0.86,
                },
                "aggregates": {
                    "mean_deception_lift": 2.0,
                    "mean_spoof_event_lift": 10.0,
                    "mean_spoof_susceptibility": 0.43,
                },
                "checkpoint_set": {
                    "radio": "checkpoints/radio_policy.pt",
                    "scanner": "checkpoints/scanner_decoder.pt",
                    "jammer": "checkpoints/jammer_policy.pt",
                },
            }
            server.CHECKPOINT_LEAGUE_MANIFEST_PATH.write_text(json.dumps(information), encoding="utf-8")
            server.ACTIVE_INFORMATION_MANIFEST_PATH.write_text(json.dumps(information), encoding="utf-8")
            fidelity = {
                "valid": True,
                "passed": True,
                "replay": "replays/acceptance/downtown_chase_seed11.jsonl",
                "summary": {"frames": 1200, "schema_version": 1},
                "fingerprint": {
                    "schema_version": 1,
                    "frames": 1200,
                    "state_sha256": "a" * 64,
                    "radio_sha256": "b" * 64,
                    "full_sha256": "c" * 64,
                },
                "scripted_determinism": {
                    "matched": True,
                    "seed": 11,
                    "steps": 1200,
                },
            }
            server.REPLAY_FIDELITY_MANIFEST_PATH.write_text(json.dumps(fidelity), encoding="utf-8")
            physics = {
                "passed": True,
                "checks_passed": 5,
                "checks": 5,
                "score": 1.0,
                "results": [
                    {
                        "name": "deterministic_trace",
                        "passed": True,
                        "metrics": {"max_state_delta": 0.0},
                    },
                    {
                        "name": "handbrake_asymmetry",
                        "passed": True,
                        "metrics": {
                            "yaw_ratio": 1.22,
                            "radius_ratio": 2.04,
                            "speed_ratio": 0.44,
                        },
                    },
                ],
            }
            server.PHYSICS_VALIDATION_MANIFEST_PATH.write_text(json.dumps(physics), encoding="utf-8")
            env_validation = {
                "passed": True,
                "checks_passed": 6,
                "checks": 6,
                "score": 1.0,
                "results": [
                    {"name": "pettingzoo_parallel_api", "passed": True},
                    {"name": "seeded_env_determinism", "passed": True},
                ],
            }
            server.ENV_VALIDATION_MANIFEST_PATH.write_text(json.dumps(env_validation), encoding="utf-8")
            spectator = {
                "passed": True,
                "checks_passed": 5,
                "checks": 5,
                "score": 1.0,
                "replay": "replays/acceptance/downtown_chase_seed11.jsonl",
                "results": [
                    {
                        "name": "audio_confidence_mapping",
                        "passed": True,
                        "metrics": {
                            "plans": [
                                {"bucket": "low"},
                                {"bucket": "mid"},
                                {"bucket": "high"},
                            ],
                        },
                    },
                    {
                        "name": "web_replay_payload_contract",
                        "passed": True,
                        "metrics": {
                            "frame_count": 1200,
                            "returned_frames": 16,
                        },
                    },
                    {
                        "name": "dashboard_replay_deep_links",
                        "passed": True,
                        "metrics": {
                            "valid_replays": 35,
                            "with_viewer_links": 35,
                        },
                    },
                    {
                        "name": "pygame_headless_smoke",
                        "passed": True,
                        "metrics": {"duration": 0.4},
                    },
                ],
            }
            server.SPECTATOR_VALIDATION_MANIFEST_PATH.write_text(json.dumps(spectator), encoding="utf-8")
            operational = {
                "passed": True,
                "checks_passed": 10,
                "checks": 10,
                "score": 1.0,
                "summary": "Operational readiness evidence is complete.",
                "results": [],
            }
            server.OPERATIONAL_READINESS_MANIFEST_PATH.write_text(json.dumps(operational), encoding="utf-8")
            server.CURRICULUM_PLAN_PATH.write_text(json.dumps(plan), encoding="utf-8")
            try:
                response = self.client.get("/api/readiness")
                self.assertEqual(response.status_code, 200)
                data = response.get_json()
                self.assertEqual(data["status"], "needs_training")
                self.assertEqual(data["recommended_command"]["command_id"], "train_evader_ppo")
                self.assertEqual(data["acceptance"]["required_score"], 1.0)
                self.assertEqual(data["acceptance"]["milestone_score"], 0.0)
                self.assertTrue(data["learned_control"]["evader_checkpoint_acceptance"]["exists"])
                self.assertTrue(data["learned_control"]["active_control_acceptance"]["exists"])
                self.assertTrue(data["learned_information"]["checkpoint_league"]["passed"])
                self.assertTrue(data["learned_information"]["active_information"]["passed"])
                self.assertTrue(data["replay_fidelity"]["passed"])
                self.assertEqual(data["replay_fidelity"]["state_sha256_short"], "a" * 16)
                self.assertTrue(data["physics_validation"]["passed"])
                self.assertEqual(data["physics_validation"]["checks_passed"], 5)
                self.assertTrue(data["env_validation"]["passed"])
                self.assertTrue(data["env_validation"]["pettingzoo_parallel_api"])
                self.assertTrue(data["spectator_validation"]["passed"])
                self.assertEqual(data["spectator_validation"]["audio_buckets"], ["low", "mid", "high"])
                self.assertEqual(data["spectator_validation"]["valid_replays"], 35)
                self.assertTrue(data["operational_validation"]["passed"])
                self.assertEqual(data["operational_validation"]["checks_passed"], 10)
                self.assertEqual(data["checkpoints"]["existing"], 1)
                self.assertEqual(data["checkpoints"]["missing"], 1)
                self.assertTrue(any(blocker["area"] == "curriculum" for blocker in data["blockers"]))
            finally:
                server.ACCEPTANCE_MANIFEST_PATH = old_acceptance
                server.STRICT_ACCEPTANCE_MANIFEST_PATH = old_strict
                server.EVADER_CHECKPOINT_ACCEPTANCE_PATH = old_evader_acceptance
                server.PURSUER_TEAM_CHECKPOINT_ACCEPTANCE_PATH = old_pursuer_team_acceptance
                server.ACTIVE_CONTROL_ACCEPTANCE_PATH = old_active_control_acceptance
                server.CHECKPOINT_LEAGUE_MANIFEST_PATH = old_checkpoint_league
                server.ACTIVE_INFORMATION_MANIFEST_PATH = old_active_information
                server.REPLAY_FIDELITY_MANIFEST_PATH = old_replay_fidelity
                server.PHYSICS_VALIDATION_MANIFEST_PATH = old_physics_validation
                server.ENV_VALIDATION_MANIFEST_PATH = old_env_validation
                server.SPECTATOR_VALIDATION_MANIFEST_PATH = old_spectator_validation
                server.OPERATIONAL_READINESS_MANIFEST_PATH = old_operational_validation
                server.SELF_PLAY_MANIFEST_PATH = old_self_play
                server.CURRICULUM_PLAN_PATH = old_plan
                server.CURRICULUM_HISTORY_PATH = old_history
                server.READINESS_CHECKPOINTS = old_checkpoints

    def test_dashboard_readiness_routes_failed_self_play_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_acceptance = server.ACCEPTANCE_MANIFEST_PATH
            old_strict = server.STRICT_ACCEPTANCE_MANIFEST_PATH
            old_evader_acceptance = server.EVADER_CHECKPOINT_ACCEPTANCE_PATH
            old_pursuer_team_acceptance = server.PURSUER_TEAM_CHECKPOINT_ACCEPTANCE_PATH
            old_active_control_acceptance = server.ACTIVE_CONTROL_ACCEPTANCE_PATH
            old_checkpoint_league = server.CHECKPOINT_LEAGUE_MANIFEST_PATH
            old_active_information = server.ACTIVE_INFORMATION_MANIFEST_PATH
            old_replay_fidelity = server.REPLAY_FIDELITY_MANIFEST_PATH
            old_physics_validation = server.PHYSICS_VALIDATION_MANIFEST_PATH
            old_env_validation = server.ENV_VALIDATION_MANIFEST_PATH
            old_spectator_validation = server.SPECTATOR_VALIDATION_MANIFEST_PATH
            old_operational_validation = server.OPERATIONAL_READINESS_MANIFEST_PATH
            old_self_play = server.SELF_PLAY_MANIFEST_PATH
            old_plan = server.CURRICULUM_PLAN_PATH
            old_checkpoints = server.READINESS_CHECKPOINTS
            server.ACCEPTANCE_MANIFEST_PATH = root / "acceptance.json"
            server.STRICT_ACCEPTANCE_MANIFEST_PATH = root / "strict.json"
            server.EVADER_CHECKPOINT_ACCEPTANCE_PATH = root / "evader_checkpoint_acceptance.json"
            server.PURSUER_TEAM_CHECKPOINT_ACCEPTANCE_PATH = root / "pursuer_team_checkpoint_acceptance.json"
            server.ACTIVE_CONTROL_ACCEPTANCE_PATH = root / "control_acceptance_active.json"
            server.CHECKPOINT_LEAGUE_MANIFEST_PATH = root / "checkpoint_league_manifest.json"
            server.ACTIVE_INFORMATION_MANIFEST_PATH = root / "active_information.json"
            server.REPLAY_FIDELITY_MANIFEST_PATH = root / "replay_fidelity.json"
            server.PHYSICS_VALIDATION_MANIFEST_PATH = root / "physics_validation.json"
            server.ENV_VALIDATION_MANIFEST_PATH = root / "env_validation.json"
            server.SPECTATOR_VALIDATION_MANIFEST_PATH = root / "spectator_validation.json"
            server.OPERATIONAL_READINESS_MANIFEST_PATH = root / "operational_readiness.json"
            server.SELF_PLAY_MANIFEST_PATH = root / "selfplay.json"
            server.CURRICULUM_PLAN_PATH = root / "curriculum_plan.json"
            server.READINESS_CHECKPOINTS = ()
            acceptance = {
                "passed": True,
                "required_checks_passed": 16,
                "required_checks": 16,
                "milestone_targets_passed": 1,
                "milestone_targets": 1,
                "score": {
                    "overall_score": 1.0,
                    "required_score": 1.0,
                    "milestone_score": 1.0,
                },
                "diagnostics": {"count": 0, "top": []},
                "scenarios": [],
            }
            information = {
                "scores": {
                    "overall_score": 0.71,
                    "pursuer_security_score": 0.73,
                    "evader_pressure_score": 0.60,
                    "adversarial_balance_score": 0.86,
                },
                "aggregates": {
                    "mean_deception_lift": 2.0,
                    "mean_spoof_event_lift": 10.0,
                    "mean_spoof_susceptibility": 0.43,
                },
            }
            fidelity = {
                "valid": True,
                "passed": True,
                "summary": {"frames": 1200, "schema_version": 1},
                "fingerprint": {
                    "schema_version": 1,
                    "frames": 1200,
                    "state_sha256": "a" * 64,
                    "radio_sha256": "b" * 64,
                },
                "scripted_determinism": {"matched": True},
            }
            physics = {
                "passed": True,
                "checks_passed": 5,
                "checks": 5,
                "score": 1.0,
                "results": [],
            }
            spectator = {
                "passed": True,
                "checks_passed": 5,
                "checks": 5,
                "score": 1.0,
                "results": [],
            }
            plan = {
                "version": 1,
                "status": "nominal",
                "diagnostic_count": 0,
                "summary": "All acceptance diagnostics are nominal.",
                "actions": [
                    {
                        "lane": "scenario_gated_self_play",
                        "command_id": "run_self_play_cycle_scenario_gated",
                        "default_args": "--name selfplay_scenario",
                    }
                ],
            }
            self_play = {
                "name": "selfplay_scenario",
                "seed": 11,
                "control_promotion_decision": {
                    "promoted": False,
                    "reason": "scenario_acceptance_failed",
                    "candidate_score": 0.44,
                    "scenario_acceptance_present": True,
                    "scenario_acceptance_passed": False,
                    "scenario_required_checks_passed": 15,
                    "scenario_required_checks": 16,
                },
                "control_league": {
                    "scores": {
                        "overall_score": 0.44,
                        "scenario_acceptance_score": 0.80,
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
                        "diagnostics": {
                            "count": 1,
                            "top": [
                                {
                                    "scenario": "downtown_chase",
                                    "severity": "high",
                                    "reason": "speed_collapse",
                                    "owner": "evader",
                                }
                            ],
                        },
                    },
                },
            }
            for path in (
                server.ACCEPTANCE_MANIFEST_PATH,
                server.STRICT_ACCEPTANCE_MANIFEST_PATH,
                server.EVADER_CHECKPOINT_ACCEPTANCE_PATH,
                server.PURSUER_TEAM_CHECKPOINT_ACCEPTANCE_PATH,
                server.ACTIVE_CONTROL_ACCEPTANCE_PATH,
            ):
                path.write_text(json.dumps(acceptance), encoding="utf-8")
            server.CHECKPOINT_LEAGUE_MANIFEST_PATH.write_text(json.dumps(information), encoding="utf-8")
            server.ACTIVE_INFORMATION_MANIFEST_PATH.write_text(json.dumps(information), encoding="utf-8")
            server.REPLAY_FIDELITY_MANIFEST_PATH.write_text(json.dumps(fidelity), encoding="utf-8")
            server.PHYSICS_VALIDATION_MANIFEST_PATH.write_text(json.dumps(physics), encoding="utf-8")
            env_validation = {
                "passed": True,
                "checks_passed": 6,
                "checks": 6,
                "score": 1.0,
                "results": [
                    {"name": "pettingzoo_parallel_api", "passed": True},
                    {"name": "seeded_env_determinism", "passed": True},
                ],
            }
            server.ENV_VALIDATION_MANIFEST_PATH.write_text(json.dumps(env_validation), encoding="utf-8")
            server.SPECTATOR_VALIDATION_MANIFEST_PATH.write_text(json.dumps(spectator), encoding="utf-8")
            operational = {
                "passed": True,
                "checks_passed": 10,
                "checks": 10,
                "score": 1.0,
                "summary": "Operational readiness evidence is complete.",
                "results": [],
            }
            server.OPERATIONAL_READINESS_MANIFEST_PATH.write_text(json.dumps(operational), encoding="utf-8")
            server.CURRICULUM_PLAN_PATH.write_text(json.dumps(plan), encoding="utf-8")
            server.SELF_PLAY_MANIFEST_PATH.write_text(json.dumps(self_play), encoding="utf-8")
            try:
                response = self.client.get("/api/readiness")
                self.assertEqual(response.status_code, 200)
                data = response.get_json()
                self.assertEqual(data["status"], "self_play_needs_repair")
                self.assertEqual(data["self_play"]["status"], "failed")
                self.assertEqual(data["self_play"]["repair_action"]["command_id"], "train_evader_ppo")
                self.assertEqual(data["recommended_command"]["source"], "self_play")
                self.assertEqual(data["recommended_command"]["command_id"], "train_evader_ppo")
                self.assertTrue(any(blocker["area"] == "self_play" for blocker in data["blockers"]))
            finally:
                server.ACCEPTANCE_MANIFEST_PATH = old_acceptance
                server.STRICT_ACCEPTANCE_MANIFEST_PATH = old_strict
                server.EVADER_CHECKPOINT_ACCEPTANCE_PATH = old_evader_acceptance
                server.PURSUER_TEAM_CHECKPOINT_ACCEPTANCE_PATH = old_pursuer_team_acceptance
                server.ACTIVE_CONTROL_ACCEPTANCE_PATH = old_active_control_acceptance
                server.CHECKPOINT_LEAGUE_MANIFEST_PATH = old_checkpoint_league
                server.ACTIVE_INFORMATION_MANIFEST_PATH = old_active_information
                server.REPLAY_FIDELITY_MANIFEST_PATH = old_replay_fidelity
                server.PHYSICS_VALIDATION_MANIFEST_PATH = old_physics_validation
                server.ENV_VALIDATION_MANIFEST_PATH = old_env_validation
                server.SPECTATOR_VALIDATION_MANIFEST_PATH = old_spectator_validation
                server.OPERATIONAL_READINESS_MANIFEST_PATH = old_operational_validation
                server.SELF_PLAY_MANIFEST_PATH = old_self_play
                server.CURRICULUM_PLAN_PATH = old_plan
                server.READINESS_CHECKPOINTS = old_checkpoints

    def test_dashboard_readiness_routes_evidence_bundle_gates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_acceptance = server.ACCEPTANCE_MANIFEST_PATH
            old_strict = server.STRICT_ACCEPTANCE_MANIFEST_PATH
            old_evader_acceptance = server.EVADER_CHECKPOINT_ACCEPTANCE_PATH
            old_pursuer_team_acceptance = server.PURSUER_TEAM_CHECKPOINT_ACCEPTANCE_PATH
            old_active_control_acceptance = server.ACTIVE_CONTROL_ACCEPTANCE_PATH
            old_checkpoint_league = server.CHECKPOINT_LEAGUE_MANIFEST_PATH
            old_active_information = server.ACTIVE_INFORMATION_MANIFEST_PATH
            old_replay_fidelity = server.REPLAY_FIDELITY_MANIFEST_PATH
            old_physics_validation = server.PHYSICS_VALIDATION_MANIFEST_PATH
            old_env_validation = server.ENV_VALIDATION_MANIFEST_PATH
            old_spectator_validation = server.SPECTATOR_VALIDATION_MANIFEST_PATH
            old_operational_validation = server.OPERATIONAL_READINESS_MANIFEST_PATH
            old_evidence_bundle = server.EVIDENCE_BUNDLE_PATH
            old_evidence_verification = server.EVIDENCE_BUNDLE_VERIFICATION_PATH
            old_self_play = server.SELF_PLAY_MANIFEST_PATH
            old_plan = server.CURRICULUM_PLAN_PATH
            old_checkpoints = server.READINESS_CHECKPOINTS
            server.ACCEPTANCE_MANIFEST_PATH = root / "acceptance.json"
            server.STRICT_ACCEPTANCE_MANIFEST_PATH = root / "strict.json"
            server.EVADER_CHECKPOINT_ACCEPTANCE_PATH = root / "evader_checkpoint_acceptance.json"
            server.PURSUER_TEAM_CHECKPOINT_ACCEPTANCE_PATH = root / "pursuer_team_checkpoint_acceptance.json"
            server.ACTIVE_CONTROL_ACCEPTANCE_PATH = root / "control_acceptance_active.json"
            server.CHECKPOINT_LEAGUE_MANIFEST_PATH = root / "checkpoint_league_manifest.json"
            server.ACTIVE_INFORMATION_MANIFEST_PATH = root / "active_information.json"
            server.REPLAY_FIDELITY_MANIFEST_PATH = root / "replay_fidelity.json"
            server.PHYSICS_VALIDATION_MANIFEST_PATH = root / "physics_validation.json"
            server.ENV_VALIDATION_MANIFEST_PATH = root / "env_validation.json"
            server.SPECTATOR_VALIDATION_MANIFEST_PATH = root / "spectator_validation.json"
            server.OPERATIONAL_READINESS_MANIFEST_PATH = root / "operational_readiness.json"
            server.EVIDENCE_BUNDLE_PATH = root / "evidence_bundle.json"
            server.EVIDENCE_BUNDLE_VERIFICATION_PATH = root / "evidence_bundle_verification.json"
            server.SELF_PLAY_MANIFEST_PATH = root / "selfplay.json"
            server.CURRICULUM_PLAN_PATH = root / "curriculum_plan.json"
            server.READINESS_CHECKPOINTS = ()

            acceptance = {
                "passed": True,
                "required_checks_passed": 16,
                "required_checks": 16,
                "milestone_targets_passed": 1,
                "milestone_targets": 1,
                "score": {
                    "overall_score": 1.0,
                    "required_score": 1.0,
                    "milestone_score": 1.0,
                },
                "diagnostics": {"count": 0, "top": []},
                "scenarios": [],
            }
            information = {
                "scores": {
                    "overall_score": 0.72,
                    "pursuer_security_score": 0.74,
                    "evader_pressure_score": 0.61,
                    "adversarial_balance_score": 0.86,
                },
                "aggregates": {
                    "mean_deception_lift": 2.1,
                    "mean_spoof_event_lift": 10.0,
                    "mean_spoof_susceptibility": 0.42,
                },
            }
            fidelity = {
                "valid": True,
                "passed": True,
                "summary": {"frames": 1200, "schema_version": 1},
                "fingerprint": {
                    "schema_version": 1,
                    "frames": 1200,
                    "state_sha256": "a" * 64,
                    "radio_sha256": "b" * 64,
                },
                "scripted_determinism": {"matched": True},
            }
            physics = {
                "passed": True,
                "checks_passed": 5,
                "checks": 5,
                "score": 1.0,
                "results": [],
            }
            env_validation = {
                "passed": True,
                "checks_passed": 6,
                "checks": 6,
                "score": 1.0,
                "results": [
                    {"name": "pettingzoo_parallel_api", "passed": True},
                    {"name": "seeded_env_determinism", "passed": True},
                ],
            }
            spectator = {
                "passed": True,
                "checks_passed": 5,
                "checks": 5,
                "score": 1.0,
                "results": [],
            }
            operational = {
                "passed": True,
                "checks_passed": 11,
                "checks": 11,
                "score": 1.0,
                "summary": "Operational readiness evidence is complete.",
                "results": [],
            }
            plan = {
                "version": 1,
                "status": "nominal",
                "diagnostic_count": 0,
                "summary": "All acceptance diagnostics are nominal.",
                "actions": [],
            }
            for path in (
                server.ACCEPTANCE_MANIFEST_PATH,
                server.STRICT_ACCEPTANCE_MANIFEST_PATH,
                server.EVADER_CHECKPOINT_ACCEPTANCE_PATH,
                server.PURSUER_TEAM_CHECKPOINT_ACCEPTANCE_PATH,
                server.ACTIVE_CONTROL_ACCEPTANCE_PATH,
            ):
                path.write_text(json.dumps(acceptance), encoding="utf-8")
            server.CHECKPOINT_LEAGUE_MANIFEST_PATH.write_text(json.dumps(information), encoding="utf-8")
            server.ACTIVE_INFORMATION_MANIFEST_PATH.write_text(json.dumps(information), encoding="utf-8")
            server.REPLAY_FIDELITY_MANIFEST_PATH.write_text(json.dumps(fidelity), encoding="utf-8")
            server.PHYSICS_VALIDATION_MANIFEST_PATH.write_text(json.dumps(physics), encoding="utf-8")
            server.ENV_VALIDATION_MANIFEST_PATH.write_text(json.dumps(env_validation), encoding="utf-8")
            server.SPECTATOR_VALIDATION_MANIFEST_PATH.write_text(json.dumps(spectator), encoding="utf-8")
            server.OPERATIONAL_READINESS_MANIFEST_PATH.write_text(json.dumps(operational), encoding="utf-8")
            server.CURRICULUM_PLAN_PATH.write_text(json.dumps(plan), encoding="utf-8")

            try:
                response = self.client.get("/api/readiness")
                self.assertEqual(response.status_code, 200)
                data = response.get_json()
                self.assertEqual(data["status"], "evidence_bundle_failing")
                self.assertEqual(data["recommended_command"]["command_id"], "build_evidence_bundle")
                self.assertEqual(data["evidence_bundle"]["status"], "missing")
                self.assertTrue(any(blocker["area"] == "evidence_bundle" for blocker in data["blockers"]))

                bundle = {
                    "version": 1,
                    "kind": "crypt_heist_evidence_bundle",
                    "created_at": "2026-06-30T00:00:00",
                    "passed": True,
                    "summary": {
                        "operational_passed": True,
                        "operational_checks_passed": 11,
                        "operational_checks": 11,
                        "active_acceptance_passed": True,
                        "active_required_checks_passed": 16,
                        "active_required_checks": 16,
                        "active_milestone_targets_passed": 1,
                        "active_milestone_targets": 1,
                        "active_replay_reports": 4,
                        "active_replay_reports_nominal": 4,
                        "checkpoints_existing": 10,
                        "checkpoints_expected": 10,
                    },
                }
                server.EVIDENCE_BUNDLE_PATH.write_text(json.dumps(bundle), encoding="utf-8")
                data = self.client.get("/api/readiness").get_json()
                self.assertEqual(data["status"], "evidence_verification_failing")
                self.assertEqual(data["recommended_command"]["command_id"], "verify_evidence_bundle")
                self.assertTrue(data["evidence_bundle"]["passed"])
                self.assertEqual(data["evidence_bundle_verification"]["status"], "missing")

                verification = {
                    "version": 1,
                    "kind": "crypt_heist_evidence_bundle_verification",
                    "bundle": str(server.EVIDENCE_BUNDLE_PATH),
                    "bundle_created_at": bundle["created_at"],
                    "passed": True,
                    "checks_passed": 38,
                    "checks": 38,
                    "failures": [],
                    "results": [],
                }
                server.EVIDENCE_BUNDLE_VERIFICATION_PATH.write_text(json.dumps(verification), encoding="utf-8")
                data = self.client.get("/api/readiness").get_json()
                self.assertEqual(data["status"], "operational")
                self.assertTrue(data["evidence_bundle"]["passed"])
                self.assertTrue(data["evidence_bundle_verification"]["passed"])
                self.assertEqual(data["evidence_bundle_verification"]["checks_passed"], 38)
            finally:
                server.ACCEPTANCE_MANIFEST_PATH = old_acceptance
                server.STRICT_ACCEPTANCE_MANIFEST_PATH = old_strict
                server.EVADER_CHECKPOINT_ACCEPTANCE_PATH = old_evader_acceptance
                server.PURSUER_TEAM_CHECKPOINT_ACCEPTANCE_PATH = old_pursuer_team_acceptance
                server.ACTIVE_CONTROL_ACCEPTANCE_PATH = old_active_control_acceptance
                server.CHECKPOINT_LEAGUE_MANIFEST_PATH = old_checkpoint_league
                server.ACTIVE_INFORMATION_MANIFEST_PATH = old_active_information
                server.REPLAY_FIDELITY_MANIFEST_PATH = old_replay_fidelity
                server.PHYSICS_VALIDATION_MANIFEST_PATH = old_physics_validation
                server.ENV_VALIDATION_MANIFEST_PATH = old_env_validation
                server.SPECTATOR_VALIDATION_MANIFEST_PATH = old_spectator_validation
                server.OPERATIONAL_READINESS_MANIFEST_PATH = old_operational_validation
                server.EVIDENCE_BUNDLE_PATH = old_evidence_bundle
                server.EVIDENCE_BUNDLE_VERIFICATION_PATH = old_evidence_verification
                server.SELF_PLAY_MANIFEST_PATH = old_self_play
                server.CURRICULUM_PLAN_PATH = old_plan
                server.READINESS_CHECKPOINTS = old_checkpoints

    def test_dashboard_readiness_button_runs_recommended_command(self):
        script = (server.ROOT / "command_center/static/app.js").read_text(encoding="utf-8")
        self.assertIn('recommended.source === "curriculum"', script)
        self.assertIn('recommended.command_id ? \'<button id="run-next-readiness"', script)
        self.assertIn('recommended.lane || next.lane || "n/a"', script)
        self.assertIn('await runCommand(recommended.command_id, recommended.default_args || "")', script)
        self.assertIn("envStatusLine", script)
        self.assertIn("operationalStatusLine", script)
        self.assertIn("evidenceStatusLine", script)
        self.assertIn("evidenceVerificationLine", script)

    def test_dashboard_runs_registered_command(self):
        start = self.client.post("/api/run", json={"id": "compileall", "args": ""})
        self.assertEqual(start.status_code, 200)
        job = start.get_json()
        job_id = job["id"]
        deadline = time.time() + 12.0
        while time.time() < deadline:
            current = self.client.get(f"/api/jobs/{job_id}").get_json()
            if current["status"] != "running":
                break
            time.sleep(0.1)
        self.assertEqual(current["status"], "succeeded")
        self.assertIn("compileall", " ".join(current["argv"]))

    def test_dashboard_replay_validate_action(self):
        record_scripted_replay("replays/dashboard-test.jsonl", seed=8, steps=8)
        start = self.client.post(
            "/api/run_replay_action",
            json={"action": "validate", "replay": "dashboard-test.jsonl"},
        )
        self.assertEqual(start.status_code, 200)
        job = start.get_json()
        deadline = time.time() + 12.0
        while time.time() < deadline:
            current = self.client.get(f"/api/jobs/{job['id']}").get_json()
            if current["status"] != "running":
                break
            time.sleep(0.1)
        self.assertEqual(current["status"], "succeeded")
        self.assertIn("avg_confidence", current["log"])

    def test_dashboard_nested_replay_index_and_action(self):
        path = Path("replays/dashboard-nested/nested-test.jsonl")
        record_scripted_replay(path, seed=9, steps=8)
        try:
            replays = self.client.get("/api/replays")
            self.assertEqual(replays.status_code, 200)
            items = replays.get_json()
            self.assertTrue(any(item["name"] == "dashboard-nested/nested-test.jsonl" for item in items))
            nested = next(item for item in items if item["name"] == "dashboard-nested/nested-test.jsonl")
            self.assertIn("avg_evader_reward", nested)
            self.assertIn("avg_pursuer_auth_penalty", nested)
            self.assertIn("radio_word_entropy", nested)
            self.assertIn("jam_events", nested)
            self.assertIn("cipher_rotations", nested)

            start = self.client.post(
                "/api/run_replay_action",
                json={"action": "validate", "replay": "replays/dashboard-nested/nested-test.jsonl"},
            )
            self.assertEqual(start.status_code, 200)
            job = start.get_json()
            deadline = time.time() + 12.0
            while time.time() < deadline:
                current = self.client.get(f"/api/jobs/{job['id']}").get_json()
                if current["status"] != "running":
                    break
                time.sleep(0.1)
            self.assertEqual(current["status"], "succeeded")
            self.assertIn("nested-test.jsonl", " ".join(current["argv"]))

            report_path = Path("logs/replay_reports/dashboard-nested_nested-test_report.json")
            if report_path.exists():
                report_path.unlink()
            start = self.client.post(
                "/api/run_replay_action",
                json={"action": "report", "replay": "replays/dashboard-nested/nested-test.jsonl"},
            )
            self.assertEqual(start.status_code, 200)
            job = start.get_json()
            deadline = time.time() + 12.0
            while time.time() < deadline:
                current = self.client.get(f"/api/jobs/{job['id']}").get_json()
                if current["status"] != "running":
                    break
                time.sleep(0.1)
            self.assertEqual(current["status"], "succeeded")
            self.assertIn("replay_report", current["log"])
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["kind"], "replay_diagnostic_report")
            self.assertIn("information_warfare", report["sections"])
        finally:
            if path.exists():
                path.unlink()
            report_path = Path("logs/replay_reports/dashboard-nested_nested-test_report.json")
            if report_path.exists():
                report_path.unlink()
            if path.parent.exists():
                try:
                    path.parent.rmdir()
                except OSError:
                    pass

    def test_dashboard_replay_payload_api_samples_valid_replay(self):
        path = Path("replays/dashboard-payload/payload-test.jsonl")
        record_scripted_replay(path, seed=10, steps=20)
        try:
            response = self.client.get(
                "/api/replay",
                query_string={"path": "replays/dashboard-payload/payload-test.jsonl", "max_frames": 5},
            )
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(data["name"], "dashboard-payload/payload-test.jsonl")
            self.assertEqual(data["frame_count"], 20)
            self.assertLessEqual(data["returned_frames"], 5)
            self.assertGreaterEqual(data["stride"], 1)
            self.assertIn("summary", data)
            self.assertIn("avg_evader_reward", data["summary"])
            self.assertIn("avg_pursuer_auth_penalty", data["summary"])
            self.assertIn("avg_decoder_accuracy", data["summary"])
            self.assertIn("radio_word_entropy", data["summary"])
            self.assertIn("avg_jam_confidence_drop", data["summary"])
            self.assertIn("avg_cipher_confidence_recovery", data["summary"])
            self.assertIn("agents", data["frames"][0])
            self.assertIn("radio", data["frames"][0])
            self.assertIn("actions", data["frames"][0])
            self.assertIn("rewards", data["frames"][0])
            self.assertIn("reward_components", data["frames"][0])
            self.assertIn("camera", data["frames"][0])
            self.assertIn("EVADER", data["frames"][0]["rewards"])
            self.assertIn("auth_penalty", data["frames"][0]["reward_components"]["P1"])

            rejected = self.client.get("/api/replay", query_string={"path": "../outside.jsonl"})
            self.assertIn(rejected.status_code, {400, 404})
        finally:
            if path.exists():
                path.unlink()
            if path.parent.exists():
                try:
                    path.parent.rmdir()
                except OSError:
                    pass

    def test_dashboard_curriculum_plan_api_and_action(self):
        with tempfile.TemporaryDirectory() as td:
            old_path = server.CURRICULUM_PLAN_PATH
            old_history = server.CURRICULUM_HISTORY_PATH
            server.CURRICULUM_PLAN_PATH = Path(td) / "curriculum_plan.json"
            server.CURRICULUM_HISTORY_PATH = Path(td) / "curriculum_history.json"
            plan = {
                "version": 1,
                "status": "needs_training",
                "diagnostic_count": 1,
                "summary": "Top diagnostic is downtown_chase:medium:waypoint_milestone_gap.",
                "actions": [
                    {
                        "priority": 44,
                        "lane": "evader_waypoint_driver",
                        "stage": "evader_waypoints",
                        "owner": "evader",
                        "command_id": "compileall",
                        "default_args": "crypt_heist/curriculum.py",
                        "objective": "Compile the routed repair target.",
                        "rationale": "Dashboard curriculum action smoke test.",
                        "diagnostic_reasons": ["waypoint_milestone_gap"],
                        "scenarios": ["downtown_chase"],
                        "evidence": {},
                        "followup_command_ids": ["eval_acceptance_scenarios"],
                        "tags": ["test"],
                    }
                ],
            }
            server.CURRICULUM_PLAN_PATH.write_text(json.dumps(plan), encoding="utf-8")
            try:
                response = self.client.get("/api/curriculum_plan")
                self.assertEqual(response.status_code, 200)
                data = response.get_json()
                self.assertTrue(data["exists"])
                self.assertEqual(data["plan"]["actions"][0]["command_id"], "compileall")
                self.assertIn("history", data)

                start = self.client.post("/api/run_curriculum_action", json={"index": 0})
                self.assertEqual(start.status_code, 200)
                job = start.get_json()
                self.assertEqual(job["command_id"], "compileall")
                self.assertIn("curriculum.py", " ".join(job["argv"]))
                deadline = time.time() + 12.0
                while time.time() < deadline:
                    current = self.client.get(f"/api/jobs/{job['id']}").get_json()
                    if current["status"] != "running":
                        break
                    time.sleep(0.1)
                self.assertEqual(current["status"], "succeeded")
                history = json.loads(server.CURRICULUM_HISTORY_PATH.read_text(encoding="utf-8"))
                self.assertEqual(history["events"][-1]["kind"], "repair")
                self.assertEqual(history["events"][-1]["lane"], "evader_waypoint_driver")
                response = self.client.get("/api/curriculum_plan")
                lane_history = response.get_json()["history"]["by_lane"]["evader_waypoint_driver"]
                self.assertEqual(lane_history["latest_repair"]["command_id"], "compileall")
            finally:
                server.CURRICULUM_PLAN_PATH = old_path
                server.CURRICULUM_HISTORY_PATH = old_history

    def test_dashboard_curriculum_followup_action(self):
        with tempfile.TemporaryDirectory() as td:
            old_path = server.CURRICULUM_PLAN_PATH
            old_history = server.CURRICULUM_HISTORY_PATH
            server.CURRICULUM_PLAN_PATH = Path(td) / "curriculum_plan.json"
            server.CURRICULUM_HISTORY_PATH = Path(td) / "curriculum_history.json"
            plan = {
                "version": 1,
                "status": "needs_training",
                "diagnostic_count": 1,
                "summary": "Top diagnostic is downtown_chase:medium:waypoint_milestone_gap.",
                "actions": [
                    {
                        "priority": 44,
                        "lane": "evader_waypoint_driver",
                        "stage": "evader_waypoints",
                        "owner": "evader",
                        "command_id": "train_evader_ppo",
                        "default_args": "--updates 1 --steps-per-update 8",
                        "objective": "Repair waypoint control.",
                        "rationale": "Dashboard curriculum follow-up smoke test.",
                        "diagnostic_reasons": ["waypoint_milestone_gap"],
                        "scenarios": ["downtown_chase"],
                        "evidence": {},
                        "followup_command_ids": ["compileall"],
                        "tags": ["test"],
                    }
                ],
            }
            server.CURRICULUM_PLAN_PATH.write_text(json.dumps(plan), encoding="utf-8")
            try:
                blocked = self.client.post(
                    "/api/run_curriculum_followup",
                    json={"index": 0, "command_id": "unit_tests"},
                )
                self.assertEqual(blocked.status_code, 400)

                start = self.client.post(
                    "/api/run_curriculum_followup",
                    json={"index": 0, "command_id": "compileall"},
                )
                self.assertEqual(start.status_code, 200)
                job = start.get_json()
                self.assertEqual(job["command_id"], "compileall")
                self.assertIn("Follow-up", job["label"])
                self.assertIn("crypt_heist", " ".join(job["argv"]))
                deadline = time.time() + 12.0
                while time.time() < deadline:
                    current = self.client.get(f"/api/jobs/{job['id']}").get_json()
                    if current["status"] != "running":
                        break
                    time.sleep(0.1)
                self.assertEqual(current["status"], "succeeded")
                history = json.loads(server.CURRICULUM_HISTORY_PATH.read_text(encoding="utf-8"))
                self.assertEqual(history["events"][-1]["kind"], "followup")
                self.assertEqual(history["events"][-1]["command_id"], "compileall")
            finally:
                server.CURRICULUM_PLAN_PATH = old_path
                server.CURRICULUM_HISTORY_PATH = old_history

    def test_acceptance_job_auto_updates_curriculum_plan(self):
        with tempfile.TemporaryDirectory(dir=server.ROOT) as td:
            root = Path(td)
            manifest_path = root / "acceptance.json"
            plan_path = root / "curriculum_plan.json"
            history_path = root / "curriculum_history.json"
            old_path = server.CURRICULUM_PLAN_PATH
            old_history = server.CURRICULUM_HISTORY_PATH
            server.CURRICULUM_PLAN_PATH = plan_path
            server.CURRICULUM_HISTORY_PATH = history_path
            manifest = {
                "diagnostics": {
                    "top": [
                        {
                            "scenario": "downtown_chase",
                            "owner": "evader",
                            "reason": "waypoint_milestone_gap",
                            "severity": "medium",
                            "evidence": {"waypoints_hit": 1},
                        }
                    ]
                },
                "scenarios": [],
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            try:
                job = server.Job(
                    id="acceptance-test",
                    command_id="eval_acceptance_scenarios",
                    label="Acceptance Scenarios",
                    argv=[
                        "python3",
                        "scripts/eval_acceptance_scenarios.py",
                        "--out",
                        str(manifest_path.relative_to(server.ROOT)),
                    ],
                    cwd=str(server.ROOT),
                )
                lines = server._postprocess_completed_job(job)
                self.assertTrue(plan_path.exists())
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                self.assertEqual(plan["status"], "needs_training")
                self.assertEqual(plan["actions"][0]["command_id"], "train_evader_ppo")
                self.assertTrue(any("curriculum plan updated" in line for line in lines))
                history = json.loads(history_path.read_text(encoding="utf-8"))
                self.assertEqual(history["events"][-1]["kind"], "acceptance")
                self.assertEqual(history["events"][-1]["diagnostic_trend"], "unknown")

                manifest["diagnostics"]["top"] = []
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                lines = server._postprocess_completed_job(job)
                history = json.loads(history_path.read_text(encoding="utf-8"))
                self.assertEqual(history["events"][-1]["diagnostic_count"], 0)
                self.assertEqual(history["events"][-1]["diagnostic_trend"], "improved")
            finally:
                server.CURRICULUM_PLAN_PATH = old_path
                server.CURRICULUM_HISTORY_PATH = old_history


if __name__ == "__main__":
    unittest.main()
