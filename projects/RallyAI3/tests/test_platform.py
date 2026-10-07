import os
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from rallylab import core
from rallylab.api import CourseSpec, EvaluationSpec, RunSpec, app
from rallylab.clone import cloning_schedule, update_budget
from rallylab.worker import (BENCHMARK_SEEDS, checkpoint_trainer_state,
                             benchmark_suite_audit, procedural_excluded_seeds, tune_demonstration_updates,
                             tune_short_run_updates, trainer_step_rate)


class PlatformTests(unittest.TestCase):
    def setUp(self):
        self.previous = core.DATA
        self.temp = tempfile.TemporaryDirectory()
        core.DATA = Path(self.temp.name)

    def tearDown(self):
        core.DATA = self.previous
        self.temp.cleanup()

    def test_atomic_manifest_and_empty_result(self):
        manifest = core.DATA / "courses/abc/manifest.json"
        core.write(manifest, {"schema": 1, "id": "abc"})
        self.assertEqual(core.read(manifest)["id"], "abc")
        self.assertEqual(list(manifest.parent.glob("*.tmp")), [])
        result = core.summary([], expected=20)
        self.assertIsNone(result["best"])
        self.assertEqual(result["attempts"], 0)
        self.assertFalse(result["complete"])

    def test_time_attack_counts_only_valid_finishes(self):
        rows = [
            {"outcome": "Finished", "seconds": 75.2, "valid": True},
            {"outcome": "Finished", "seconds": 70.0, "valid": False},
            {"outcome": "HitObstacle", "seconds": 22.0, "valid": False},
        ]
        result = core.summary(rows, expected=3)
        self.assertEqual(result["best"], 75.2)
        self.assertEqual(result["median"], 75.2)
        self.assertEqual(result["finishes"], 1)
        self.assertEqual(result["finish_interval"][0] <= 1/3 <= result["finish_interval"][1], True)
        self.assertTrue(result["complete"])

    def test_http_state_and_loopback_guard(self):
        client = TestClient(app)
        response = client.get("/api/state")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["jobs"], [])
        blocked = client.get("/api/state", headers={"host": "remote.example"})
        self.assertEqual(blocked.status_code, 403)

    def test_episode_ingestion_waits_for_complete_jsonl_lines_and_is_idempotent(self):
        folder = core.DATA / "jobs/run-a/episodes"
        folder.mkdir(parents=True)
        path = folder / "episodes-0-session.jsonl"
        first = {"outcome": "FellOff", "reward": -2.0, "waypoints": 3, "target": 10, "seed": 44, "station": 80, "off": 5}
        path.write_bytes((json.dumps(first) + "\n").encode() + b'{"outcome":"RolledOver","reward":-4')
        core.ingest_episode_events("run-a")
        self.assertEqual(core.indexed_summary("run-a")["attempts"], 1)
        core.ingest_episode_events("run-a")
        self.assertEqual(core.indexed_summary("run-a")["attempts"], 1)

        with path.open("ab") as stream:
            stream.write(b',"waypoints":5,"target":10,"seed":44}\n')
        core.ingest_episode_events("run-a")
        summary = core.indexed_summary("run-a")
        self.assertEqual(summary["attempts"], 2)
        self.assertEqual(summary["outcomes"], {"FellOff": 1, "RolledOver": 1})
        self.assertEqual([row["outcome"] for row in core.indexed_episodes("run-a")], ["FellOff", "RolledOver"])

    def test_run_telemetry_snapshot_and_cursor_updates_are_durable(self):
        folder = core.DATA / "jobs/run-a"
        folder.mkdir(parents=True)
        now = core.time.time()
        with core.db() as conn:
            conn.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?)", ("run-a", "training", "running", now, now,
                json.dumps({"name": "Telemetry smoke", "mode": "specialist", "steps": 10000}),
                json.dumps({"trainer_step": 1200, "steps_per_second": 80, "cpu_percent": 25, "memory_bytes": 1024, "children": 2, "heartbeat": now})))
        episode_path = folder / "episodes/episodes-0-session.jsonl"
        episode_path.parent.mkdir(parents=True)
        episode_path.write_text(json.dumps({"outcome": "TimedOut", "reward": 2.5, "waypoints": 4, "target": 10, "seed": 9}) + "\n")
        core.append_telemetry("run-a", "resource", {"cpu_percent": 25, "memory_bytes": 1024, "children": 2, "steps_per_second": 80})
        core.append_telemetry("run-a", "scalar", {"tag": "Environment/Cumulative Reward", "step": 1200, "value": 2.5, "wall_time": now})

        client = TestClient(app)
        snapshot = client.get("/api/jobs/run-a/telemetry")
        self.assertEqual(snapshot.status_code, 200)
        data = snapshot.json()
        self.assertFalse(data["incremental"])
        self.assertEqual(data["summary"]["episodes_completed"], 1)
        self.assertEqual(data["history"]["episodes"][0]["seed"], 9)
        self.assertEqual(data["history"]["resources"][0]["steps_per_second"], 80)
        cursor = data["cursor"]

        core.append_telemetry("run-a", "scalar", {"tag": "Losses/Policy Loss", "step": 1300, "value": -0.02, "wall_time": now + 1})
        delta = client.get(f"/api/jobs/run-a/telemetry?cursor={cursor}")
        self.assertTrue(delta.json()["incremental"])
        self.assertEqual([(event["kind"], event.get("tag")) for event in delta.json()["events"]], [("scalar", "Losses/Policy Loss")])
        again = client.get(f"/api/jobs/run-a/telemetry?cursor={delta.json()['cursor']}")
        self.assertEqual(again.json()["events"], [])

    def test_scalar_and_checkpoint_telemetry_are_deduplicated(self):
        now = core.time.time()
        with core.db() as conn:
            conn.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?)", ("run-a", "training", "running", now, now, "{}", "{}"))
        scalar = {"tag": "Environment/Cumulative Reward", "step": 200, "value": 1.25}
        checkpoint = {"id": "checkpoint-hash", "step": 200}
        self.assertIsNotNone(core.append_telemetry("run-a", "scalar", scalar))
        self.assertIsNone(core.append_telemetry("run-a", "scalar", scalar))
        self.assertIsNotNone(core.append_telemetry("run-a", "checkpoint", checkpoint))
        self.assertIsNone(core.append_telemetry("run-a", "checkpoint", checkpoint))
        snapshot = core.telemetry_snapshot("run-a")
        self.assertEqual(len(snapshot["history"]["scalars"]), 1)
        self.assertEqual(len(snapshot["history"]["checkpoints"]), 1)

    def test_unavailable_cpu_sample_does_not_fall_back_to_a_false_zero(self):
        now = core.time.time()
        with core.db() as conn:
            conn.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?)", ("run-a", "training", "running", now, now, "{}",
                json.dumps({"cpu_percent": 0.0, "memory_bytes": 2048, "children": 1, "heartbeat": now})))
        core.capture_job_samples("run-a", min_gap=0, cpu_percent=None)
        sample = core.telemetry_snapshot("run-a")["history"]["resources"][0]
        self.assertIsNone(sample["cpu_percent"])
        self.assertEqual(sample["source"], "dashboard-sampler")
        self.assertEqual(sample["memory_bytes"], 2048)

    def test_polling_stopped_job_does_not_fabricate_fresh_resource_sample(self):
        now = core.time.time()
        with core.db() as conn:
            conn.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?)", ("run-stopped", "training", "stopped", now - 60, now, "{}",
                json.dumps({"cpu_percent": 0.0, "memory_bytes": 2048, "children": 1, "heartbeat": now - 60})))
        client = TestClient(app)
        response = client.get("/api/jobs/run-stopped/telemetry")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["history"]["resources"], [])

    def test_stop_force_and_resume_controls_use_expected_lifecycle_states(self):
        now = core.time.time()
        with core.db() as conn:
            conn.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?)", ("run-stop", "training", "running", now, now,
                json.dumps({"name": "Control smoke"}), "{}"))
            conn.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?)", ("run-resume", "training", "stopped", now, now,
                json.dumps({"name": "Resume smoke"}), "{}"))
        (core.DATA / "jobs/run-stop").mkdir(parents=True)
        trainer_state = core.DATA / "jobs/run-resume/trainer/checkpoint.pt"
        trainer_state.parent.mkdir(parents=True)
        trainer_state.write_bytes(b"test trainer state")

        client = TestClient(app)
        stopped = client.post("/api/jobs/run-stop/stop")
        self.assertEqual(stopped.status_code, 200)
        self.assertEqual(stopped.json()["state"], "stopping")
        self.assertTrue((core.DATA / "jobs/run-stop/stop").exists())
        forced = client.post("/api/jobs/run-stop/force")
        self.assertEqual(forced.status_code, 200)
        self.assertTrue((core.DATA / "jobs/run-stop/force").exists())

        with patch("rallylab.api.core.spawn") as spawn:
            resumed = client.post("/api/jobs/run-resume/resume")
        self.assertEqual(resumed.status_code, 200)
        self.assertEqual(resumed.json()["state"], "queued")
        spawn.assert_called_once_with("run-resume")

    def test_training_episode_summary_does_not_claim_valid_time_attack_results(self):
        now = core.time.time()
        with core.db() as conn:
            conn.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?)", ("training-a", "training", "running", now, now,
                json.dumps({"name": "Specialist", "mode": "specialist", "steps": 5000}),
                json.dumps({"trainer_step": 800})))
        output = core.DATA / "jobs/training-a/episodes/episodes-0-session.jsonl"
        output.parent.mkdir(parents=True)
        output.write_text(json.dumps({"outcome": "Finished", "seconds": 42, "valid": False, "reward": 7.0}) + "\n")
        state = __import__("rallylab.api", fromlist=["state"]).state()
        run = next(item for item in state["jobs"] if item["id"] == "training-a")
        self.assertEqual(run["detail"]["trainer_step"], 800)
        self.assertEqual(run["summary"]["episodes_completed"], 1)
        self.assertEqual(run["summary"]["finishes"], 0)
        self.assertIsNone(run["summary"]["best"])

    def test_invalid_training_request_is_rejected_before_launch(self):
        response = TestClient(app).post("/api/runs", json={"name": "bad", "mode": "specialist", "course": "missing"})
        self.assertIn(response.status_code, (409, 422))
        self.assertEqual(core.jobs(), [])

    def test_viewer_request_is_exactly_one_attempt_even_when_default_is_twenty(self):
        with patch("rallylab.api.require_checkpoint"), \
             patch("rallylab.api.require_course"), \
             patch("rallylab.api.core.create", return_value={"id": "viewer-smoke"}) as create:
            response = TestClient(app).post("/api/viewers", json={
                "checkpoint": "checkpoint-id", "courses": ["course-id"]
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], "viewer-smoke")
        spec = create.call_args.args[1]
        self.assertEqual(spec["attempts"], 1)

    def test_course_spec_keeps_explicitly_disabled_scenery(self):
        spec = CourseSpec(name="Crests / no hazards", seed=41200, family="crests", rocks=0, scenery=False)
        self.assertFalse(spec.model_dump()["scenery"])

    def test_no_hazards_specialist_requires_zero_object_audit(self):
        valid = {"id": "clean-course", "suite": "library", "review": "reviewed", "rocks": 0,
                 "scenery": False, "definition": {"scenery": False, "obstacleObjects": 0}}
        with patch("rallylab.api.core.courses", return_value=[valid]), \
             patch("rallylab.api.core.course_asset_valid", return_value=True), \
             patch("rallylab.api.core.build_status", return_value={"status": "current"}), \
             patch("rallylab.api.core.create", return_value={"id": "queued"}) as create:
            response = TestClient(app).post("/api/runs", json={
                "name": "Crests no-hazards audit", "mode": "specialist", "course": "clean-course"
            })
        self.assertEqual(response.status_code, 200)
        create.assert_called_once()

        unaudited = {**valid, "definition": {"scenery": False}}
        with patch("rallylab.api.core.courses", return_value=[unaudited]), \
             patch("rallylab.api.core.course_asset_valid", return_value=True), \
             patch("rallylab.api.core.build_status", return_value={"status": "current"}), \
             patch("rallylab.api.core.create") as create:
            response = TestClient(app).post("/api/runs", json={
                "name": "Reject unaudited course", "mode": "specialist", "course": "clean-course"
            })
        self.assertEqual(response.status_code, 409)
        self.assertIn("zero-obstacle generation audit", response.json()["detail"])
        create.assert_not_called()

    def test_legacy_template_rock_course_cannot_start_new_specialist_run(self):
        legacy = {"id": "old-course", "suite": "library", "review": "reviewed",
                  "legacyObstacleLayout": True}
        with patch("rallylab.api.core.courses", return_value=[legacy]), \
             patch("rallylab.api.core.course_asset_valid", return_value=True), \
             patch("rallylab.api.core.build_status", return_value={"status": "current"}), \
             patch("rallylab.api.core.create") as create:
            response = TestClient(app).post("/api/runs", json={
                "name": "Must not train on stale template rocks", "mode": "specialist", "course": "old-course"
            })
        self.assertEqual(response.status_code, 409)
        self.assertIn("stale template road rocks", response.json()["detail"])
        create.assert_not_called()

    def test_legacy_course_run_cannot_resume(self):
        core.write(core.DATA / "courses/old-course/manifest.json", {
            "id": "old-course", "legacyObstacleLayout": True
        })
        state = core.DATA / "jobs/legacy-run/trainer/legacy-run/RallyDriver/checkpoint.pt"
        state.parent.mkdir(parents=True)
        state.write_bytes(b"saved trainer state")
        now = core.time.time()
        with core.db() as conn:
            conn.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?)", (
                "legacy-run", "training", "stopped", now, now,
                json.dumps({"mode": "specialist", "course": "old-course"}), "{}"
            ))
        with patch("rallylab.api.core.spawn") as spawn:
            response = TestClient(app).post("/api/jobs/legacy-run/resume")
        self.assertEqual(response.status_code, 409)
        self.assertIn("cannot resume", response.json()["detail"])
        self.assertEqual(core.job("legacy-run")["state"], "stopped")
        spawn.assert_not_called()

    def test_demo_recording_requires_a_control_probe(self):
        response = TestClient(app).post("/api/evaluations", json={
            "checkpoint": "checkpoint", "courses": ["course"], "recordDemonstration": True
        })
        self.assertEqual(response.status_code, 409)
        self.assertEqual(core.jobs(), [])

    def test_reference_recording_requires_reviewed_course_and_current_build(self):
        core.write(core.DATA / "courses/course-a/manifest.json", {
            "id": "course-a", "suite": "library", "review": "reviewed"
        })
        with patch("rallylab.api.core.build_status", return_value={"status": "current"}), \
             patch("rallylab.api.core.course_asset_valid", return_value=True), \
             patch("rallylab.api.core.create", return_value={"id": "demo-job", "kind": "demonstration"}) as create:
            response = TestClient(app).post("/api/demonstrations/record", json={"course": "course-a"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(create.call_args.args[0], "demonstration")
        self.assertEqual(create.call_args.args[1]["course"], "course-a")

    def test_training_rejects_modified_demonstration_artifacts(self):
        contract = {"continuous": 2, "observationWidths": [84, 84, 44]}
        demo = core.DATA / "demonstrations/demo-id/teacher.demo"
        demo.parent.mkdir(parents=True)
        demo.write_bytes(b"demo artifact")
        core.write(demo.parent / "manifest.json", {
            "id": "demo-id", "path": str(demo), "sha256": core.sha(demo), "contract": contract
        })
        payload = {"name": "BC smoke", "mode": "specialist", "course": "course-a",
                   "demonstrations": ["demo-id"]}
        with patch("rallylab.api.core.build_status", return_value={"status": "current", "contract": contract}), \
             patch("rallylab.api.core.courses", return_value=[{"id": "course-a", "suite": "library", "review": "reviewed"}]), \
             patch("rallylab.api.core.course_asset_valid", return_value=True), \
             patch("rallylab.api.core.create", return_value={"id": "run-id"}) as create:
            accepted = TestClient(app).post("/api/runs", json=payload)
            self.assertEqual(accepted.status_code, 200)
            demo.write_bytes(b"modified artifact")
            rejected = TestClient(app).post("/api/runs", json=payload)
        self.assertEqual(create.call_count, 1)
        self.assertEqual(rejected.status_code, 409)

    def test_short_training_budgets_trigger_optimizer_updates_without_changing_long_runs(self):
        behavior = {"hyperparameters": {"batch_size": 2048, "buffer_size": 20480}}
        tune_short_run_updates(behavior, 4096)
        self.assertEqual(behavior["hyperparameters"]["batch_size"], 512)
        self.assertEqual(behavior["hyperparameters"]["buffer_size"], 1024)

        full_budget = {"hyperparameters": {"batch_size": 2048, "buffer_size": 20480}}
        tune_short_run_updates(full_budget, 100000)
        self.assertEqual(full_budget["hyperparameters"], {"batch_size": 2048, "buffer_size": 20480})

    def test_trainer_throughput_retains_last_nonzero_interval_between_summaries(self):
        self.assertEqual(trainer_step_rate(20000, 40000, 15), 20000 / 15)
        self.assertIsNone(trainer_step_rate(40000, 40000, 15))
        self.assertIsNone(trainer_step_rate(40000, 30000, 15))

    def test_demonstrations_receive_optimizer_updates_during_the_cloning_window(self):
        behavior = {"hyperparameters": {"batch_size": 2048, "buffer_size": 20480}}
        tune_demonstration_updates(behavior, 10000)
        self.assertEqual(behavior["hyperparameters"]["batch_size"], 512)
        self.assertEqual(behavior["hyperparameters"]["buffer_size"], 512)

    def test_isolated_clone_budget_maps_to_complete_optimizer_updates(self):
        self.assertEqual(update_budget(10000, 512), 20)
        self.assertEqual(update_budget(512, 512), 1)
        self.assertEqual(update_budget(1000, 0), 1000)
        self.assertEqual(cloning_schedule(10000, 2048), {
            "batch_size": 512, "buffer_size": 512, "samples_per_update": 512,
            "epochs": 3, "updates": 20})
        self.assertEqual(cloning_schedule(1000, 64)["updates"], 16)

    def test_generalist_sampling_reserves_all_benchmark_seeds_before_suite_creation(self):
        saved = [
            {"seed": 123, "suite": "library"},
            {"seed": 456, "suite": "development"},
            {"seed": 789, "suite": "held-out"},
        ]
        excluded = procedural_excluded_seeds(saved)
        reserved = {seed for seeds in BENCHMARK_SEEDS.values() for seed in seeds}
        self.assertEqual(len(reserved), 60)
        self.assertTrue(reserved.issubset(excluded))
        self.assertTrue({456, 789}.issubset(excluded))
        self.assertNotIn(123, excluded)

    def test_benchmark_suite_audit_requires_balanced_unique_course_assets(self):
        courses = []
        for suite, seeds in BENCHMARK_SEEDS.items():
            for seed in seeds:
                bundle = core.DATA / "benchmark-fixtures" / f"{suite}-{seed}.bundle"
                bundle.parent.mkdir(parents=True, exist_ok=True)
                bundle.write_bytes(f"asset-{suite}-{seed}".encode())
        for suite, seeds in BENCHMARK_SEEDS.items():
            for index, seed in enumerate(seeds):
                definition = {"resolvedHash": f"geometry-{suite}-{seed}"}
                bundle = core.DATA / "benchmark-fixtures" / f"{suite}-{seed}.bundle"
                manifest = {
                    "suite": suite, "seed": seed,
                    "family": ("gentle", "technical", "crests")[index // 10],
                    "rocks": 0 if index % 2 == 0 else 2,
                    "source": f"source-{suite}-{seed}", "definition": definition,
                    "bundle": str(bundle), "bundle_hash": core.sha(bundle),
                }
                manifest["id"] = core.course_identity(manifest)
                courses.append(manifest)
        audit = benchmark_suite_audit(courses)
        self.assertTrue(audit["complete"], audit["errors"])
        self.assertEqual(audit["total"], 60)
        for suite in BENCHMARK_SEEDS:
            self.assertEqual(audit["suites"][suite]["courses"], 30)
            self.assertEqual(audit["suites"][suite]["families"], {
                "gentle": 10, "technical": 10, "crests": 10})
            self.assertEqual(audit["suites"][suite]["obstacles"], {0: 15, 2: 15})

        incomplete = benchmark_suite_audit(courses[:-1])
        self.assertFalse(incomplete["complete"])
        self.assertTrue(any("missing 1 reserved courses" in error for error in incomplete["errors"]))

        duplicated = [dict(course, definition=dict(course["definition"])) for course in courses]
        duplicated[-1]["definition"]["resolvedHash"] = duplicated[0]["definition"]["resolvedHash"]
        self.assertTrue(any("geometry hashes are not unique" in error
                            for error in benchmark_suite_audit(duplicated)["errors"]))

    def test_checkpoint_initialization_uses_exact_manifest_trainer_state(self):
        exact = core.DATA / "jobs/run-a/trainer/pretraining/RallyDriver/checkpoint.pt"
        exact.parent.mkdir(parents=True)
        exact.write_bytes(b"exact clone state")
        manifest = {"run": "run-a", "step": 2000, "trainer_state": str(exact)}
        self.assertEqual(checkpoint_trainer_state(manifest), exact)
        legacy = {"run": "run-a", "step": 5000}
        with patch("rallylab.worker.DATA", core.DATA):
            self.assertEqual(checkpoint_trainer_state(legacy),
                             core.DATA / "jobs/run-a/trainer/run-a/RallyDriver/RallyDriver-5000.pt")

    def test_resume_allows_only_the_known_inference_initialization_change(self):
        contract = {"continuous": 2, "observationWidths": [84, 84, 44]}
        old_hash = "b2082150e30791c20cfb1272693371315dcb0af596d4267faef9b2cc93cf2d6d"
        new_hash = "d65c768e92644fa1b19593c4bef1e7f8cc2152d22ec2958681016dd295977e81"
        prior = {"contract": contract, "source": {"hash": "old", "files": {"Assets/Core/ML/LabRuntime.cs": old_hash}}}
        current = {"contract": contract, "source": {"hash": "new", "files": {"Assets/Core/ML/LabRuntime.cs": new_hash}}}
        self.assertEqual(core.resume_build_compatibility(prior, current), (True, ["Assets/Core/ML/LabRuntime.cs"]))
        changed_physics = {"contract": contract, "source": {"hash": "newer", "files": {
            "Assets/Core/ML/LabRuntime.cs": new_hash, "Assets/Core/Physics/VehicleController.cs": "changed"}}}
        self.assertFalse(core.resume_build_compatibility(prior, changed_physics)[0])
        changed_contract = {**current, "contract": {"continuous": 1}}
        self.assertFalse(core.resume_build_compatibility(prior, changed_contract)[0])

    def test_course_review_preserves_asset_identity(self):
        bundle = core.DATA / "courses/abc/course"
        bundle.parent.mkdir(parents=True)
        bundle.write_bytes(b"frozen")
        manifest = {"name": "Seeded road", "suite": "library", "review": "unreviewed",
                    "bundle": str(bundle), "bundle_hash": core.sha(bundle), "source": "source-hash",
                    "definition": {"resolvedHash": "geometry-hash"}}
        manifest["id"] = core.course_identity(manifest)
        core.write(bundle.parent / "manifest.json", manifest)
        self.assertTrue(core.course_asset_valid(manifest))
        client = TestClient(app)
        reviewed = client.post(f"/api/courses/{manifest['id']}/review")
        self.assertEqual(reviewed.status_code, 200)
        self.assertEqual(reviewed.json()["id"], manifest["id"])
        renamed = client.patch(f"/api/courses/{manifest['id']}/name", json={"name": "West ridge"})
        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(renamed.json()["id"], manifest["id"])
        self.assertEqual(core.sha(bundle), manifest["bundle_hash"])

    def test_course_review_rejects_rewritten_bundle_under_old_identity(self):
        bundle = core.DATA / "courses/course-a/course"
        bundle.parent.mkdir(parents=True)
        bundle.write_bytes(b"frozen")
        manifest = {"name": "Seeded road", "suite": "library", "review": "unreviewed",
                    "bundle": str(bundle), "bundle_hash": core.sha(bundle), "source": "source-hash",
                    "definition": {"resolvedHash": "geometry-hash"}}
        manifest["id"] = core.course_identity(manifest)
        bundle.write_bytes(b"changed")
        core.write(bundle.parent / "manifest.json", manifest)
        response = TestClient(app).post(f"/api/courses/{manifest['id']}/review")
        self.assertEqual(response.status_code, 409)
        self.assertFalse(core.read(bundle.parent / "manifest.json")["review"] == "reviewed")

    def test_evaluation_requires_each_course_attempt(self):
        folder = core.DATA / "jobs/eval-run"
        output = folder / "attempts/course-a"
        output.mkdir(parents=True)
        records = [{"courseId": "course-a", "outcome": "Finished", "seconds": 55.0, "valid": True, "attempt": 1},
                   {"courseId": "course-a", "outcome": "Finished", "seconds": 54.0, "valid": True, "attempt": 2}]
        (output / "episodes-worker.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
        now = core.time.time()
        with core.db() as conn:
            conn.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?)", ("eval-run", "evaluation", "running", now, now,
                json.dumps({"courses": ["course-a", "course-b"], "attempts": 1}), "{}"))
        result = __import__("rallylab.api", fromlist=["state"]).state()["jobs"][0]["summary"]
        self.assertEqual(result["course_attempts"], {"course-a": 2, "course-b": 0})
        self.assertFalse(result["complete"])

    def test_completed_reference_job_reports_one_requested_attempt(self):
        folder = core.DATA / "jobs/demo-run"
        output = folder / "attempts/course-a"
        output.mkdir(parents=True)
        record = {"courseId": "course-a", "outcome": "Finished", "seconds": 55.0,
                  "valid": True, "attempt": 1}
        (output / "episodes-editor.jsonl").write_text(json.dumps(record) + "\n")
        now = core.time.time()
        with core.db() as conn:
            conn.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?)", ("demo-run", "demonstration", "completed", now, now, "{}", "{}"))
        result = __import__("rallylab.api", fromlist=["state"]).state()["jobs"][0]["summary"]
        self.assertEqual(result["requested"], 1)
        self.assertTrue(result["complete"])

    def test_control_probe_is_explicit_and_disabled_by_default(self):
        base = {"checkpoint": "checkpoint", "courses": ["course"]}
        self.assertFalse(EvaluationSpec(**base).controlProbe)
        probe = EvaluationSpec(**base, controlProbe=True, controlMode="waypoint-follow", controlProbeSeconds=20)
        self.assertTrue(probe.controlProbe)
        self.assertEqual(probe.controlMode, "waypoint-follow")
        self.assertEqual(probe.controlProbeSeconds, 20)
        self.assertFalse(probe.recordDemonstration)
        full_course_probe = EvaluationSpec(**base, controlProbe=True, controlMode="waypoint-follow", controlProbeSeconds=120)
        self.assertEqual(full_course_probe.controlProbeSeconds, 120)
        with self.assertRaises(ValidationError):
            EvaluationSpec(**base, controlProbe=True, controlProbeSeconds=121)
        teacher = EvaluationSpec(**base, controlProbe=True, controlMode="waypoint-follow", recordDemonstration=True)
        self.assertTrue(teacher.recordDemonstration)
        with self.assertRaises(ValidationError):
            EvaluationSpec(**base, controlProbe=True, controlProbeSeconds=121)
        steer = EvaluationSpec(**base, controlProbe=True, controlMode="steering-step", controlSteer=-0.1, controlTargetSpeed=8)
        self.assertEqual(steer.controlSteer, -0.1)
        fixed = EvaluationSpec(**base, controlProbe=True, controlMode="fixed-input", controlDrive=0.04)
        self.assertEqual(fixed.controlDrive, 0.04)
        with self.assertRaises(ValidationError):
            EvaluationSpec(**base, controlProbe=True, controlMode="fixed-input", controlDrive=1.1)

    def test_starting_gear_is_explicit_and_preserves_neutral_default(self):
        self.assertEqual(RunSpec(name="neutral baseline").startingGear, "neutral")
        self.assertEqual(RunSpec(name="first gear", startingGear="first").startingGear, "first")
        self.assertEqual(RunSpec(name="time attack", reward="time-attack-v1").reward, "time-attack-v1")
        self.assertEqual(RunSpec(name="clone first", demonstrations=["demo"]).imitationMode, "isolated")
        self.assertEqual(RunSpec(name="joint clone", demonstrations=["demo"], imitationMode="joint").imitationMode, "joint")
        self.assertEqual(EvaluationSpec(checkpoint="c", courses=["course"]).startingGear, "neutral")
        self.assertEqual(EvaluationSpec(checkpoint="c", courses=["course"], startingGear="first").startingGear, "first")
        with self.assertRaises(ValidationError):
            EvaluationSpec(checkpoint="c", courses=["course"], startingGear="reverse")


if __name__ == "__main__":
    unittest.main()
