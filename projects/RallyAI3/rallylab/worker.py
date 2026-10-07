"""Durable, single-job supervisor. The API process is not its parent lifecycle."""
from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

import psutil
import yaml

from .core import (ROOT, DATA, UNITY, PYTHON, append_telemetry, build_status, checkpoints, course_asset_valid, courses, demonstrations, episodes,
                   fingerprint, job, read, resume_build_compatibility, sha, summary, update, write)


class Cancelled(Exception):
    pass


BENCHMARK_SEEDS = {
    "development": tuple(range(910000, 910030)),
    "held-out": tuple(range(920000, 920030)),
}


def procedural_excluded_seeds(saved_courses):
    reserved = {seed for seeds in BENCHMARK_SEEDS.values() for seed in seeds}
    reserved.update(course["seed"] for course in saved_courses if course.get("suite") != "library")
    return sorted(reserved)


def benchmark_suite_audit(saved_courses):
    expected = {}
    for suite, seeds in BENCHMARK_SEEDS.items():
        expected[suite] = {
            seed: (("gentle", "technical", "crests")[index // 10], 0 if index % 2 == 0 else 2)
            for index, seed in enumerate(seeds)
        }

    errors = []
    selected = [course for course in saved_courses if course.get("suite") in expected]
    suite_counts = {}
    identities = []
    geometry_hashes = []
    for suite, seed_map in expected.items():
        rows = [course for course in selected if course.get("suite") == suite]
        by_seed = {}
        for course in rows:
            seed = course.get("seed")
            if seed in by_seed:
                errors.append(f"{suite} has duplicate seed {seed}")
            by_seed[seed] = course
        missing = sorted(set(seed_map) - set(by_seed))
        extra = sorted(seed for seed in by_seed if seed not in seed_map)
        if missing:
            errors.append(f"{suite} is missing {len(missing)} reserved courses")
        if extra:
            errors.append(f"{suite} contains {len(extra)} unreserved seeds")
        family_counts = {family: 0 for family in ("gentle", "technical", "crests")}
        obstacle_counts = {0: 0, 2: 0}
        for seed, course in by_seed.items():
            contract = seed_map.get(seed)
            if contract and (course.get("family"), course.get("rocks")) != contract:
                errors.append(f"{suite} seed {seed} has the wrong layout or obstacle condition")
            if course.get("family") in family_counts:
                family_counts[course["family"]] += 1
            if course.get("rocks") in obstacle_counts:
                obstacle_counts[course["rocks"]] += 1
            if course.get("id"):
                identities.append(course["id"])
            else:
                errors.append(f"{suite} seed {seed} has no immutable course identity")
            resolved_hash = course.get("definition", {}).get("resolvedHash")
            if resolved_hash:
                geometry_hashes.append(resolved_hash)
            else:
                errors.append(f"{suite} seed {seed} has no resolved geometry hash")
            if not course_asset_valid(course):
                errors.append(f"{suite} seed {seed} failed its source, definition, or bundle hash check")
        suite_counts[suite] = {"courses": len(rows), "families": family_counts, "obstacles": obstacle_counts}

    if len(identities) != len(set(identities)):
        errors.append("Benchmark course identities are not unique")
    if len(geometry_hashes) != len(set(geometry_hashes)):
        errors.append("Benchmark geometry hashes are not unique")
    return {"complete": not errors, "total": len(selected), "suites": suite_counts, "errors": errors}


def trainer_step_rate(previous_step, current_step, elapsed):
    if current_step <= previous_step:
        return None
    return (current_step - previous_step) / max(1.0, elapsed)


def run_process(identifier, args, env=None, interruptible=True, on_tick=None):
    folder = DATA / "jobs" / identifier
    with (folder / "process.log").open("a") as log:
        process = subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        update(identifier, "running", child_pid=process.pid, child_created=psutil.Process(process.pid).create_time())
        stopping = False
        next_tick = 0.0
        next_resource_sample = 0.0
        cpu_watch = {}
        while process.poll() is None:
            if interruptible and (folder / "stop").exists() and not stopping:
                # Signal the trainer first; it owns orderly worker shutdown and saves in finally.
                process.send_signal(signal.SIGINT)
                stopping = True
                update(identifier, "stopping", stop_requested=time.time())
            if (folder / "force").exists():
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise Cancelled("Forced termination; latest checkpoint may not include recent steps")
            if on_tick and time.time() >= next_tick:
                on_tick()
                next_tick = time.time() + 15
            try:
                proc = psutil.Process(process.pid)
                children = proc.children(recursive=True)
                process_tree = [p for p in [proc, *children] if p.is_running()]
                memory = sum(p.memory_info().rss for p in process_tree)
                current_cpu_watch = {}
                cpu = 0.0
                measured_cpu = False
                for current in process_tree:
                    previous = cpu_watch.get(current.pid)
                    try:
                        if previous is None:
                            current.cpu_percent(None)
                            current_cpu_watch[current.pid] = current
                        else:
                            cpu += previous.cpu_percent(None)
                            current_cpu_watch[current.pid] = previous
                            measured_cpu = True
                    except psutil.Error:
                        continue
                cpu_watch = current_cpu_watch
                cpu = cpu if measured_cpu else None
                heartbeat = time.time()
                update(identifier, memory_bytes=memory, cpu_percent=cpu, children=len(children), heartbeat=heartbeat)
                if heartbeat >= next_resource_sample:
                    try:
                        live_metrics = job(identifier)["detail"]
                        append_telemetry(identifier, "resource", {
                            "source": "worker",
                            "cpu_percent": cpu, "memory_bytes": memory, "children": len(children),
                            "trainer_step": live_metrics.get("trainer_step"),
                            "steps_per_second": live_metrics.get("steps_per_second"),
                            "heartbeat": heartbeat,
                        }, created=heartbeat)
                    except Exception:
                        pass
                    next_resource_sample = heartbeat + 15
            except psutil.Error:
                pass
            time.sleep(1)
        # Clean only the session created above; trainer exit must not leave Unity behind.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        return process.returncode, stopping


def bridge(identifier, action, **payload):
    bridge_dir = DATA / "bridge"
    request_id = payload.pop("_request_id", identifier)
    bridge_dir.mkdir(parents=True, exist_ok=True)
    request = bridge_dir / "request.json"
    response = bridge_dir / "response.json"
    request.unlink(missing_ok=True)
    response.unlink(missing_ok=True)
    write(request, {"id": request_id, "action": action, **payload})
    editor = None
    for process in psutil.process_iter(["cmdline", "name"]):
        args = process.info["cmdline"] or []
        if process.info["name"] == "Unity" and str(ROOT) in args:
            editor = process
            break
    if editor:
        update(identifier, "running", message="Waiting for Unity Editor preparation")
        deadline = time.time() + 120
        while not response.exists() and time.time() < deadline:
            if (DATA / "jobs" / identifier / "stop").exists():
                request.unlink(missing_ok=True)
                raise Cancelled("Stopped while waiting for Editor")
            heartbeat = read(bridge_dir / "editor.json", {}) or {}
            if time.time() - heartbeat.get("time", 0) > 20 and time.time() - (deadline - 120) > 30:
                request.unlink(missing_ok=True)
                raise RuntimeError("Unity Editor Rally bridge is not responding; check its Console and license")
            time.sleep(1)
        if not response.exists():
            request.unlink(missing_ok=True)
            raise RuntimeError("Unity Editor preparation timed out")
    else:
        code, _ = run_process(identifier, [str(UNITY), "-batchmode", "-nographics", "-quit", "-projectPath", str(ROOT),
                                  "-executeMethod", "EditorScripts.LabBridge.Execute", "-logFile",
                                  str(DATA / "jobs" / identifier / "unity.log")], interruptible=False)
        if code:
            request.unlink(missing_ok=True)
            raise RuntimeError("Unity preparation failed; see unity.log")
    result = read(response)
    if not result or result.get("id") != request_id or not result.get("ok"):
        raise RuntimeError((result or {}).get("error", "Unity did not return a preparation result"))
    return result


def build(identifier):
    bridge(identifier, "build", output=str(DATA / "player/RallyTraining.app"))
    binary = next((DATA / "player/RallyTraining.app/Contents/MacOS").iterdir())
    manifest = {"schema": 1, "created": time.time(), "source": fingerprint(), "executable": str(binary),
                "executable_hash": sha(binary), "contract": read(DATA / "contract.json")}
    write(DATA / "build.json", manifest)
    update(identifier, "completed", build=manifest["source"]["hash"])


def course(identifier, spec):
    temporary = DATA / "jobs" / identifier / f"course-{spec['seed']}"
    temporary.mkdir(exist_ok=True)
    bridge_id = f"{identifier}-{spec['seed']}"
    bridge(identifier, "course", _request_id=bridge_id, output=str(temporary), seed=spec["seed"],
           family=spec["family"], rocks=spec["rocks"], hasScenery=True,
           scenery=spec.get("scenery", True), length=spec.get("length", 1000),
           firstSector=spec.get("firstSector",0), sectorCount=spec.get("sectorCount",16))
    definition = read(temporary / "definition.json")
    source_hash = fingerprint()["hash"]
    course_id = __import__("hashlib").sha256(
        (source_hash + "\n" + json.dumps(definition, sort_keys=True, separators=(",", ":"))).encode()
    ).hexdigest()
    target = DATA / "courses" / course_id
    if not target.exists():
        shutil.copytree(temporary, target)
    manifest = {"schema": 1, "id": course_id, "name": spec["name"], "family": spec["family"], "seed": spec["seed"],
                "rocks": spec["rocks"], "scenery": definition.get("scenery", spec.get("scenery", True)),
                "suite": spec.get("suite", "library"), "bundle": str(target / "course"),
                "bundle_hash": sha(target / "course"), "definition": definition,
                "source": source_hash, "created": time.time(), "review": "unreviewed"}
    write(target / "manifest.json", manifest)
    return manifest


def circuit(identifier, spec):
    from .circuit import convert, variant, digest, ROOT as circuit_root
    data=convert();choice=variant(data,spec["firstSector"],spec["sectorCount"])
    if data["revision"]!=spec["revision"]:
        raise ValueError("Circuit reconstruction changed after job creation")
    packaged=read(circuit_root/"Assets/Resources/Circuits/Nordschleife.json")
    if packaged != data:
        raise ValueError("Rebuild the pinned circuit conversion before importing")
    terrain_path=circuit_root/"Assets/Resources/Circuits/NordschleifeTerrain.json"
    terrain=read(terrain_path);terrain_report=read(circuit_root/"art-source/circuits/nordschleife/terrain-report.json")
    if terrain.get("revision")!=data["revision"] or digest(terrain_path)!=terrain_report.get("terrainHash"):
        raise ValueError("Prepare matching immutable circuit terrain before import")
    name="Nordschleife / full lap" if choice["fullLap"] else f"Nordschleife / sectors {spec['firstSector']+1:02}-{spec['firstSector']+spec['sectorCount']:02}"
    result=course(identifier,{"name":name,"seed":20261004,"family":"nordschleife","rocks":0,
                              "scenery":False,"length":0,"suite":"library",**spec})
    if result["definition"]["circuitRevision"]!=data["revision"]:
        raise ValueError("Unity circuit revision differs from the pinned source")
    update(identifier,"completed",course=result["id"])


def circuit_preview(identifier,spec):
    selected=next(c for c in courses() if c["id"]==spec["course"])
    build_manifest=build_status()
    if build_manifest["status"]!="current" or not course_asset_valid(selected):
        raise ValueError("Current player and intact circuit are required")
    folder=DATA/"jobs"/identifier
    landmark=next(l for l in selected["definition"]["landmarks"] if l["name"]=="Karussell")
    launch={"schema":1,"mode":"specialist","viewer":True,"controlProbe":True,"controlMode":"waypoint-follow",
            "controlTargetSpeed":8,"controlProbeSeconds":32,"seed":2026,"startingGear":"neutral",
            "courseBundle":selected["bundle"],"courseId":selected["id"],"courseFamily":"nordschleife",
            "courseName":selected["name"]+" / unranked reconstruction review","runId":identifier,
            "output":str(folder/"attempts"),"attempts":1,"timeScale":1}
    write(folder/"launch.json",launch);write(folder/"review-manifest.json",{"build":build_manifest,"course":selected,"spec":spec})
    env=environment(folder/"launch.json")
    env.update(RALLY_PRESENTATION_REVIEW=str(folder),RALLY_CIRCUIT_AUDIT=str(folder/"circuit-audit.json"),
               RALLY_CIRCUIT_REVIEW_STATION=str(max(0,landmark["station"]-150)),RALLY_REVIEW_CAMERA=spec["camera"],RALLY_FOREST_V3="1")
    code,stopped=run_process(identifier,[build_manifest["executable"],"-screen-width","1280","-screen-height","720",
                                      "-screen-fullscreen","0","-logFile",str(folder/"player.log")],env)
    if code and not stopped:raise RuntimeError("Circuit review player failed")
    update(identifier,"stopped" if stopped else "completed",message="Unranked circuit review; no policy or training")


def course_suite(identifier, spec):
    known = {c["seed"] for c in courses()}
    created = []
    for suite_index, suite in enumerate(("development", "held-out")):
        for index in range(30):
            seed = BENCHMARK_SEEDS[suite][index]
            if seed in known:
                continue
            if (DATA / "jobs" / identifier / "stop").exists():
                raise Cancelled("Benchmark generation stopped")
            family = ("gentle", "technical", "crests")[index // 10]
            item = {"name": f"{suite.title()} {index + 1:02}", "seed": seed, "family": family,
                    "rocks": 0 if index % 2 == 0 else 2, "suite": suite}
            created.append(course(identifier, item)["id"])
            update(identifier, "running", generated=len(created), requested=60, latest_course=item["name"])
    audit = benchmark_suite_audit(courses())
    if not audit["complete"]:
        raise ValueError("Benchmark suite audit failed: " + "; ".join(audit["errors"]))
    suite_courses = [c for c in courses() if c["suite"] in BENCHMARK_SEEDS]
    update(identifier, "completed", generated=audit["total"], requested=60,
           courses=[c["id"] for c in suite_courses], audit=audit)


def selected_course(identifier):
    found = next((x for x in courses() if x["id"] == identifier), None)
    if not found or not course_asset_valid(found):
        raise ValueError("Course missing or modified")
    if found.get("review") != "reviewed":
        raise ValueError(f"Course {found['name']} has not been reviewed")
    return found


def checkpoint_trainer_state(checkpoint):
    if checkpoint.get("trainer_state"):
        explicit = Path(checkpoint["trainer_state"])
        if explicit.is_file():
            return explicit
        fallback = (DATA / "jobs" / checkpoint["run"] / "trainer" / checkpoint["run"] /
                    "RallyDriver" / f"RallyDriver-{checkpoint.get('step')}.pt")
        if fallback.is_file():
            return fallback
        return explicit
    return (DATA / "jobs" / checkpoint["run"] / "trainer" / checkpoint["run"] /
            "RallyDriver" / f"RallyDriver-{checkpoint.get('step')}.pt")


def snapshot_checkpoints(identifier, build_manifest):
    folder = DATA / "jobs" / identifier
    found = []
    onnx_files = sorted((folder / "trainer").rglob("*.onnx"))
    trainer_states = {}
    for model in onnx_files:
        state_path = model.with_suffix(".pt")
        if state_path.is_file():
            trainer_states[sha(model)] = (str(state_path), sha(state_path))
    for onnx in onnx_files:
        digest = sha(onnx)
        destination = DATA / "checkpoints" / digest
        destination.mkdir(parents=True, exist_ok=True)
        existing = read(destination / "manifest.json")
        if not existing:
            temporary = destination / f"policy.{os.getpid()}.tmp.onnx"
            shutil.copy2(onnx, temporary)
            temporary.replace(destination / "policy.onnx")
        step = None
        try:
            from onnx import load
            model = load(str(onnx))
            inputs = [{"name": v.name, "shape": [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]} for v in model.graph.input]
            contract = build_manifest.get("contract") or {}
            expected = sorted(contract.get("observationWidths", []))
            actual = sorted(shape[-1] for item in inputs if item["shape"] and isinstance((shape := item["shape"])[-1], int))
            if expected and actual != expected:
                raise ValueError(f"Observation mismatch: policy expects {actual}, current build provides {expected}")
            action_outputs = [v for v in model.graph.output if "continuous_actions" in v.name]
            if action_outputs and int(action_outputs[0].type.tensor_type.shape.dim[-1].dim_value) != contract.get("continuous"):
                raise ValueError("Continuous action count does not match the current build")
        except Exception as exc:
            raise RuntimeError(f"Invalid checkpoint: {exc}") from exc
        try:
            step = int(onnx.stem.rsplit("-", 1)[-1])
        except ValueError:
            pass
        trainer_state = trainer_states.get(digest)
        manifest = {"schema": 1, "id": digest, "run": identifier, "step": step, "created": time.time(),
                    "policy": str(destination / "policy.onnx"), "build": build_manifest["source"]["hash"],
                    "contract": build_manifest["contract"], "inputs": inputs, "prepared": False,
                    "trainer_state": trainer_state[0] if trainer_state else None,
                    "trainer_state_sha256": trainer_state[1] if trainer_state else None}
        if existing:
            origins = existing.setdefault("origins", [])
            origin = {"run": identifier, "step": step}
            if origin not in origins: origins.append(origin)
            if not existing.get("trainer_state") and trainer_state:
                existing["trainer_state"] = trainer_state[0]
            if existing.get("trainer_state"):
                saved_state = Path(existing["trainer_state"])
                if saved_state.is_file():
                    existing.setdefault("trainer_state_sha256", sha(saved_state))
        else:
            manifest["origins"] = [{"run": identifier, "step": step}]
        write(destination / "manifest.json", existing or manifest)
        found.append(digest)
    return found


def environment(launch):
    result = {k: v for k, v in os.environ.items() if not k.startswith("RALLY_EVAL_")}
    result.update({"RALLY_LAB_LAUNCH": str(launch), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"})
    return result


def tune_short_run_updates(behavior, steps):
    hyper = behavior["hyperparameters"]
    original_buffer = hyper["buffer_size"]
    if steps >= original_buffer:
        return

    # Short probes otherwise finish before the default buffer can trigger PPO or BC updates.
    base_batch = hyper["batch_size"]
    target_batch = max(64, steps // 8)
    batch_size = min(base_batch, 2 ** (target_batch.bit_length() - 1))
    target_buffer = max(batch_size, steps // 4)
    buffer_size = max(batch_size, (target_buffer // batch_size) * batch_size)
    hyper["batch_size"] = batch_size
    hyper["buffer_size"] = min(original_buffer, buffer_size)


def tune_demonstration_updates(behavior, imitation_steps):
    hyper = behavior["hyperparameters"]
    batch_size = min(512, hyper["batch_size"])
    target_buffer = max(batch_size, imitation_steps // 20)
    buffer_size = max(batch_size, (target_buffer // batch_size) * batch_size)
    hyper["batch_size"] = batch_size
    hyper["buffer_size"] = min(hyper["buffer_size"], buffer_size)


# A full Nordschleife lap from a fixed start with its 1800 s budget would only ever teach
# the first kilometre. Training spreads starts over the route and truncates episodes;
# evaluation and viewing keep the fixed-start full-lap budget.
CIRCUIT_TRAINING_SECONDS = 180


def circuit_training_launch(selected):
    if not selected or selected.get("definition", {}).get("topology") != "circuit":
        return {}
    return {"spawnProfile": "distributed", "episodeSeconds": CIRCUIT_TRAINING_SECONDS}


def train(identifier, spec):
    manifest = build_status()
    if manifest["status"] != "current":
        raise ValueError("Prepare a current managed player before training")
    folder = DATA / "jobs" / identifier
    prior = read(folder / "run-manifest.json")
    if prior:
        compatible, changed_files = resume_build_compatibility(prior["build"], manifest)
        if not compatible:
            raise ValueError("Cannot resume against a changed environment: " + changed_files)
        if changed_files:
            record = {"from": prior["build"]["source"]["hash"], "to": manifest["source"]["hash"],
                      "changed_files": changed_files, "reason": "Known inference initialization-only change; training branch and driving contract are unchanged"}
            history = prior.setdefault("compatible_builds", [])
            if record not in history:
                history.append(record)
                write(folder / "run-manifest.json", prior)
    baseline_path = ROOT / "rallylab/baseline-v1.yaml"
    baseline_hash = sha(baseline_path)
    if prior and prior.get("baseline", {}).get("sha256") != baseline_hash:
        raise ValueError("Cannot resume after the recorded baseline configuration changed")
    selected = selected_course(spec["course"]) if spec["mode"] == "specialist" else None
    if selected and selected["suite"] != "library":
        raise ValueError("Benchmark courses cannot be used for training")
    demonstration_items = {item["id"]: item for item in demonstrations()}
    selected_demos = [demonstration_items[key] for key in spec.get("demonstrations", []) if key in demonstration_items]
    if len(selected_demos) != len(spec.get("demonstrations", [])):
        raise ValueError("A selected demonstration is no longer in the registry")
    demo_dir = folder / "demonstrations"
    for item in selected_demos:
        source = Path(item["path"])
        if not source.is_file() or sha(source) != item["sha256"]:
            raise ValueError(f"Demonstration failed its integrity check: {item['id']}")
        demo_dir.mkdir(parents=True, exist_ok=True)
        destination = demo_dir / f"{item['id']}.demo"
        if not destination.exists(): shutil.copy2(source, destination)
    launch = {"schema": 1, "mode": spec["mode"], "seed": spec["seed"], "refresh": 15,
              "courseBundle": selected["bundle"] if selected else "", "courseId": selected["id"] if selected else "",
              "runId": identifier, "output": str(folder / "episodes"), "reward": spec["reward"],
              "startingGear": spec.get("startingGear", "neutral"),
              "timeScale": 10, "excludedSeeds": procedural_excluded_seeds(courses())}
    circuit_training = circuit_training_launch(selected)
    if prior and prior.get("circuitTraining", {}) != circuit_training:
        raise ValueError("Cannot resume with a different circuit start profile or episode budget")
    launch.update(circuit_training)
    write(folder / "launch.json", launch)
    configuration = yaml.safe_load(baseline_path.read_text())
    behavior = configuration["behaviors"]["RallyDriver"]
    behavior.update(max_steps=spec["steps"], checkpoint_interval=min(500000, max(1000, spec["steps"] // 4)),
                    keep_checkpoints=100, summary_freq=min(20000, max(1000, spec["steps"] // 10)))
    tune_short_run_updates(behavior, spec["steps"])
    isolated_cloning = bool(selected_demos) and spec.get("imitationMode", "joint") == "isolated"
    if selected_demos and not isolated_cloning:
        tune_demonstration_updates(behavior, min(spec.get("imitationSteps", 10000), spec["steps"]))
    if selected:
        configuration.pop("environment_parameters", None)
    if selected_demos and not isolated_cloning:
        behavior["behavioral_cloning"] = {
            "demo_path": str(demo_dir),
            "steps": min(spec.get("imitationSteps", 10000), spec["steps"]),
            "strength": 1.0,
        }
    config_path = folder / "effective.yaml"
    if not prior:
        config_path.write_text(yaml.safe_dump(configuration))
        write(folder / "run-manifest.json", {"schema": 1, "spec": spec, "build": manifest,
                                              "baseline": {"id": "baseline-v1", "sha256": baseline_hash},
                                              "effective_configuration": configuration, "circuitTraining": circuit_training,
                                              "created": time.time()})
    trainer = folder / "trainer"
    pretraining_path = trainer / "pretraining"
    pretraining_record = read(folder / "pretraining.json", {}) or {}
    trainer_state = trainer / identifier / "RallyDriver" / "checkpoint.pt"
    parent = None
    parent_state = None
    if spec.get("parent"):
        parent = next((c for c in checkpoints() if c["id"] == spec["parent"]), None)
        if not parent or parent["contract"] != manifest["contract"]:
            raise ValueError("Parent checkpoint contract is incompatible")
        parent_state = checkpoint_trainer_state(parent)
        if not parent_state.is_file():
            raise ValueError("Exact trainer state for the selected parent checkpoint is unavailable")
        expected_state_hash = parent.get("trainer_state_sha256")
        if expected_state_hash and sha(parent_state) != expected_state_hash:
            raise ValueError("Trainer state for the selected parent checkpoint failed its integrity check")

    if isolated_cloning and not trainer_state.is_file() and not pretraining_record.get("complete"):
        pretraining_config = folder / "pretraining.json"
        write(pretraining_config, {
            "schema": 1,
            "run": identifier,
            "demonstrations": [item["id"] for item in selected_demos],
            "budget_steps": min(spec.get("imitationSteps", 10000), spec["steps"]),
            "strength": 1.0,
            "parent": spec.get("parent") or None,
        })
        clone_args = [str(PYTHON), "-m", "rallylab.clone", "--config", str(config_path),
                      "--demonstrations", str(demo_dir), "--steps",
                      str(min(spec.get("imitationSteps", 10000), spec["steps"])),
                      "--seed", str(spec["seed"]), "--output", str(pretraining_path),
                      "--state", str(pretraining_path / "clone-state.pt")]
        if parent_state:
            clone_args += ["--initial-checkpoint", str(parent_state)]
        def publish_cloning_progress():
            progress = read(pretraining_path / "clone-state.json")
            if progress:
                update(identifier, cloning_step=progress.get("completed_steps", 0),
                       cloning_budget=progress.get("target_steps", 0),
                       cloning_updates=progress.get("completed_updates", 0),
                       cloning_target_updates=progress.get("target_updates", 0))
        code, stopped = run_process(identifier, clone_args, environment(folder / "launch.json"),
                                    on_tick=publish_cloning_progress)
        if stopped:
            update(identifier, "stopped", resumable=True, cloning_interrupted=True)
            return
        if code != 0:
            raise RuntimeError("Demonstration-only pretraining failed; inspect process.log")
        pretraining_record.update({"schema": 1, "complete": True,
                                   "demonstrations": [item["id"] for item in selected_demos],
                                   "budget_steps": min(spec.get("imitationSteps", 10000), spec["steps"]),
                                   "parent": spec.get("parent") or None,
                                   "checkpoint": str(pretraining_path / "RallyDriver" / "checkpoint.pt"),
                                   "schedule": read(pretraining_path / "clone-state.json", {}),
                                   "created": time.time()})
        write(folder / "pretraining.json", pretraining_record)
        snapshot_checkpoints(identifier, manifest)

    args = [str(ROOT / ".venv/bin/mlagents-learn"), str(config_path), "--run-id", identifier,
            "--results-dir", str(trainer), "--env", str(DATA / "player/RallyTraining.app"),
            "--num-envs", str(spec["workers"]), "--seed", str(spec["seed"]), "--no-graphics",
            "--time-scale", "10", "--torch-device", "cpu"]
    if trainer_state.is_file():
        if not prior:
            raise ValueError("Unexpected trainer state without a saved run manifest")
        args.append("--resume")
    elif isolated_cloning and pretraining_record.get("complete"):
        args += ["--initialize-from", str(pretraining_path)]
    elif prior:
        if not list((trainer / identifier).rglob("*.pt")):
            raise ValueError("No trainer state exists for resume")
    elif spec.get("parent"):
        destination = trainer / "parent"
        behavior_folder = destination / "RallyDriver"
        behavior_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent_state, behavior_folder / "checkpoint.pt")
        args += ["--initialize-from", "parent"]
    lock = subprocess.Popen(["caffeinate", "-i", "-m", "-s", "-w", str(os.getpid())])
    size_seen = {}
    snapshotted_signatures = set()
    previous_detail = job(identifier)["detail"]
    previous_metrics = {"step": previous_detail.get("trainer_step", 0), "time": time.time()}
    checkpoint_events = {
        item["id"] for item in checkpoints()
        if item.get("run") == identifier or any(origin.get("run") == identifier for origin in item.get("origins", []))
    }
    scalar_steps = {}
    def publish_stable_models():
        snapshot_ready = False
        for model in trainer.rglob("*.onnx"):
            try: stat = model.stat()
            except FileNotFoundError: continue
            signature = (stat.st_size, stat.st_mtime_ns)
            signature_key = (str(model), signature)
            if size_seen.get(str(model)) == signature and signature_key not in snapshotted_signatures and time.time() - stat.st_mtime >= 4:
                snapshot_ready = True
            size_seen[str(model)] = signature
        if snapshot_ready:
            try:
                saved = snapshot_checkpoints(identifier, manifest)
                for model in trainer.rglob("*.onnx"):
                    try:
                        stat = model.stat()
                        snapshotted_signatures.add((str(model), (stat.st_size, stat.st_mtime_ns)))
                    except FileNotFoundError:
                        continue
                for checkpoint_id in saved:
                    if checkpoint_id in checkpoint_events:
                        continue
                    checkpoint = read(DATA / "checkpoints" / checkpoint_id / "manifest.json", {}) or {}
                    append_telemetry(identifier, "checkpoint", {
                        "id": checkpoint_id, "step": checkpoint.get("step"),
                        "created": checkpoint.get("created"),
                    })
                    checkpoint_events.add(checkpoint_id)
            except Exception:
                pass
        event_files = sorted(trainer.rglob("events.out.tfevents.*"), key=lambda path: path.stat().st_mtime, reverse=True)
        if event_files:
            try:
                from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
                accumulator = EventAccumulator(str(event_files[0]), size_guidance={"scalars": 100})
                accumulator.Reload()
                tags = accumulator.Tags().get("scalars", [])
                reward = accumulator.Scalars("Environment/Cumulative Reward") if "Environment/Cumulative Reward" in tags else []
                for tag in tags:
                    points = accumulator.Scalars(tag)
                    if not points:
                        continue
                    point = points[-1]
                    if scalar_steps.get(tag) == point.step:
                        continue
                    if not math.isfinite(float(point.value)):
                        continue
                    append_telemetry(identifier, "scalar", {
                        "tag": tag, "step": int(point.step), "value": float(point.value),
                        "wall_time": float(point.wall_time),
                    }, created=float(point.wall_time))
                    scalar_steps[tag] = point.step
                if reward:
                    latest = reward[-1]
                    now = time.time()
                    rate = trainer_step_rate(previous_metrics["step"], latest.step,
                                             now - previous_metrics["time"])
                    details = {"trainer_step": latest.step, "cumulative_reward": latest.value,
                               "last_summary_time": now}
                    if rate is not None:
                        details["steps_per_second"] = rate
                        previous_metrics.update(step=latest.step, time=now)
                    update(identifier, **details)
            except Exception:
                pass  # Training continues if an event file is in the middle of a write.
    try:
        code, stopped = run_process(identifier, args, environment(folder / "launch.json"), on_tick=publish_stable_models)
    finally:
        lock.terminate(); lock.wait()
    saved = snapshot_checkpoints(identifier, manifest)
    resumable = bool(list(trainer.rglob("*.pt")))
    state = "stopped" if stopped and saved and resumable else "completed" if code == 0 and not stopped and saved else "failed"
    update(identifier, state, exit_code=code, checkpoints=saved, resumable=resumable,
           error=None if state != "failed" else "Trainer failed or did not save a checkpoint; inspect process.log")


def prepare(identifier, spec):
    checkpoint = next((c for c in checkpoints() if c["id"] == spec["checkpoint"]), None)
    manifest = build_status()
    if not checkpoint or manifest["status"] != "current" or checkpoint["contract"] != manifest["contract"]:
        raise ValueError("Checkpoint or current compatible build is unavailable")
    if sha(checkpoint["policy"]) != checkpoint["id"]:
        raise ValueError("Checkpoint contents changed")
    destination = DATA / "checkpoints" / checkpoint["id"] / "bundle"
    destination.mkdir(exist_ok=True)
    bridge(identifier, "prepare", input=checkpoint["policy"], output=str(destination))
    checkpoint.update(prepared=True, bundle=str(destination / "policy"), bundle_hash=sha(destination / "policy"))
    write(DATA / "checkpoints" / checkpoint["id"] / "manifest.json", checkpoint)
    update(identifier, "completed", checkpoint=checkpoint["id"])


def inference(identifier, spec, viewer=False):
    checkpoint = next((c for c in checkpoints() if c["id"] == spec["checkpoint"]), None)
    manifest = build_status()
    if manifest["status"] != "current" or not checkpoint or not checkpoint.get("prepared"):
        raise ValueError("Prepare a compatible player and checkpoint first")
    if checkpoint["contract"] != manifest["contract"] or sha(checkpoint["bundle"]) != checkpoint["bundle_hash"]:
        raise ValueError("Checkpoint bundle is incompatible or modified")
    folder = DATA / "jobs" / identifier
    selected = [selected_course(c) for c in spec["courses"]]
    attempts = 1 if viewer else spec["attempts"]
    expected = len(selected) * attempts
    manifest_spec = {**spec, "attempts": attempts}
    write(folder / "evaluation-manifest.json", {"schema": 1, "checkpoint": checkpoint, "build": manifest,
                                               "spec": manifest_spec, "requested": expected})
    for index, course in enumerate(selected):
        output = folder / "attempts" / course["id"]
        launch = {"schema": 1, "mode": "specialist", "courseBundle": course["bundle"], "courseId": course["id"],
                  "courseName": course.get("name", ""), "courseFamily": course.get("family", ""),
                  "policyBundle": checkpoint["bundle"], "checkpointId": checkpoint["id"], "runId": identifier,
                  "output": str(output), "evaluation": not viewer, "viewer": viewer,
                  "attempts": attempts, "seed": spec["seed"], "deterministic": spec["deterministic"],
                  "timeScale": 1 if viewer else spec.get("timeScale", 10),
                  "startingGear": spec.get("startingGear", "neutral"),
                  "controlProbe": bool(spec.get("controlProbe", False)),
                  "recordDemonstration": bool(spec.get("recordDemonstration", False)),
                  "controlMode": spec.get("controlMode", "constant-throttle"),
                  "controlProbeSeconds": spec.get("controlProbeSeconds", 0),
                  "controlSteer": spec.get("controlSteer", 0),
                  "controlDrive": spec.get("controlDrive", 0.1),
                  "controlTargetSpeed": spec.get("controlTargetSpeed", 8)}
        write(folder / "launch.json", launch)
        args = [manifest["executable"], "-logFile", str(folder / f"player-{index}.log")]
        if viewer:
            args += ["-screen-width", "1280", "-screen-height", "720", "-screen-fullscreen", "0"]
        else:
            args += ["-batchmode", "-nographics"]
        code, stopped = run_process(identifier, args, environment(folder / "launch.json"))
        if stopped:
            update(identifier, "stopped", summary=summary(episodes(identifier), expected))
            return
        if not viewer:
            course_records = [r for r in episodes(identifier) if r.get("courseId") == course["id"]]
            count = len(course_records)
            if count != spec["attempts"]:
                raise RuntimeError(f"Evaluation produced {count}/{spec['attempts']} attempts on {course['name']} (exit {code})")
            if len({r.get("attempt") for r in course_records}) != spec["attempts"]:
                raise RuntimeError(f"Duplicate attempt identities on {course['name']}")
    records = episodes(identifier)
    result = summary(records, expected)
    if not viewer:
        by_course = {course["id"]: sum(r.get("courseId") == course["id"] for r in records) for course in selected}
        result["course_attempts"] = by_course
        result["complete"] = all(count == spec["attempts"] for count in by_course.values())
    if not viewer and spec.get("recordDemonstration"):
        register_demonstrations(identifier, spec, selected, checkpoint, manifest)
    update(identifier, "completed", summary=result)


def record_reference(identifier, spec):
    manifest = build_status()
    if manifest["status"] != "current":
        raise ValueError("Prepare a current managed player before recording a reference drive")
    course = selected_course(spec["course"])
    folder = DATA / "jobs" / identifier
    output = folder / "attempts" / course["id"]
    launch = {"schema": 1, "mode": "specialist", "courseBundle": course["bundle"],
              "courseId": course["id"], "runId": identifier, "output": str(output),
              "evaluation": True, "viewer": False, "attempts": 1, "seed": spec["seed"],
              "deterministic": True, "timeScale": spec["timeScale"],
              "startingGear": spec["startingGear"], "controlProbe": True,
              "recordDemonstration": True, "controlMode": "waypoint-follow",
              "controlProbeSeconds": 120}
    write(folder / "reference-manifest.json", {"schema": 1, "spec": spec, "course": course,
                                               "build": manifest, "controller": "waypoint-follow"})
    write(folder / "launch.json", launch)
    args = [manifest["executable"], "-batchmode", "-nographics", "-logFile", str(folder / "player-0.log")]
    code, stopped = run_process(identifier, args, environment(folder / "launch.json"))
    if stopped:
        update(identifier, "stopped")
        return
    if code != 0:
        raise RuntimeError(f"Reference drive exited with code {code}")
    records = episodes(identifier)
    result = summary(records, expected=1)
    register_demonstrations(identifier, {"recordDemonstration": True, "controlMode": "waypoint-follow"},
                            [course], None, manifest)
    update(identifier, "completed", summary=result)


def register_demonstrations(identifier, spec, selected, checkpoint, build):
    from mlagents.trainers.demo_loader import demo_to_buffer
    from mlagents.trainers.buffer import BufferKey

    for course in selected:
        source = DATA / "jobs" / identifier / "attempts" / course["id"] / "demonstrations" / "RallyTeacher.demo"
        if not source.is_file():
            raise RuntimeError(f"Reference demonstration was not written for {course['name']}")
        record = next((r for r in episodes(identifier)
                       if r.get("courseId") == course["id"] and r.get("attempt") == 1), None)
        if not record or record.get("outcome") != "Finished" or not record.get("valid"):
            raise RuntimeError(f"Reference drive did not validly finish {course['name']}; demonstration was not registered")
        behavior_spec, buffer = demo_to_buffer(str(source), sequence_length=128)
        widths = [list(item.shape) for item in behavior_spec.observation_specs]
        contract = build["contract"]
        if behavior_spec.action_spec.continuous_size != contract["continuous"] or widths != [[n] for n in contract["observationWidths"]]:
            raise RuntimeError("Recorded demonstration does not match the active policy contract")
        digest = sha(source)
        directory = DATA / "demonstrations" / digest
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "teacher.demo"
        if not target.exists(): shutil.copy2(source, target)
        manifest_path = directory / "manifest.json"
        prior = read(manifest_path, {}) or {}
        source_jobs = sorted(set(prior.get("sourceJobs", [prior["sourceJob"]] if prior.get("sourceJob") else []) + [identifier]))
        write(directory / "manifest.json", {
            "schema": 1, "id": digest, "sha256": digest, "path": str(target),
            "contract": contract, "sourceJob": prior.get("sourceJob", identifier),
            "sourceJobs": source_jobs, "courseId": course["id"],
            "sourceCheckpoint": checkpoint.get("id") if checkpoint else None, "controller": spec.get("controlMode"),
            "outcome": record.get("outcome"),
            "transitions": len(buffer[BufferKey.CONTINUOUS_ACTION]), "created": time.time(),
        })


def main(identifier):
    folder = DATA / "jobs" / identifier
    try:
        # One compute owner, including Editor preparation; viewers use a separate lock.
        item = job(identifier)
        lock_name = "viewer.lock" if item["kind"] == "viewer" else "compute.lock"
        with (DATA / lock_name).open("w") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if (folder / "stop").exists():
                        raise Cancelled("Cancelled while queued")
                    time.sleep(1)
            if (folder / "stop").exists():
                raise Cancelled("Cancelled before starting")
            update(identifier, "starting")
            kind, spec = item["kind"], item["spec"]
            if kind == "build": build(identifier)
            elif kind == "circuit": circuit(identifier,spec)
            elif kind == "circuit-preview": circuit_preview(identifier,spec)
            elif kind == "course":
                result = course(identifier, spec)
                update(identifier, "completed", course=result["id"])
            elif kind == "suite": course_suite(identifier, spec)
            elif kind == "demonstration": record_reference(identifier, spec)
            elif kind == "training": train(identifier, spec)
            elif kind == "prepare": prepare(identifier, spec)
            elif kind in {"evaluation", "viewer"}: inference(identifier, spec, kind == "viewer")
            else: raise ValueError("Unknown job type")
    except Cancelled as exc:
        update(identifier, "interrupted", error=str(exc))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        update(identifier, "failed", error=str(exc))


if __name__ == "__main__":
    main(sys.argv[1])
