"""Dashboard data, validated edits, and ownership-checked process controls.

This module does not import MuJoCo or Torch, so reading a dashboard never starts
simulation or touches checkpoint state. Missing evidence stays missing.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any

import psutil
import yaml

ROOT = Path(__file__).resolve().parent
STAGE_NAMES = {1: "Hopper warmup", 2: "Humanoid locomotion", 3: "Parkour foundations", 4: "Adaptive curriculum", 5: "Robustness"}
KINDS = ("flat", "boxes", "stairs_up", "stairs_down", "gap", "low_wall")


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return default


def read_yaml(path: Path):
    try:
        value = yaml.safe_load(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def atomic_write(path: Path, text: str):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def mean_present(rows, key):
    values = [r[key] for r in rows if finite(r.get(key))]
    return sum(values) / len(values) if values else None


def normalize_evaluation(raw: dict | None):
    """Legacy completion_rate is distance fraction, NEVER binary success."""
    if not isinstance(raw, dict):
        return {}
    result = dict(raw)
    episodes = [e for e in raw.get("episodes", []) if isinstance(e, dict)]
    result["episodes"] = episodes
    for key in ("distance", "speed", "upright", "facing", "clearance_rate"):
        if key not in result:
            result[key] = mean_present(episodes, key)
    if "course_progress" not in result:
        result["course_progress"] = result.get("completion_rate", mean_present(episodes, "completion_rate"))
    if "survival_fraction" not in result and episodes:
        measured = [e for e in episodes if "terminated" in e and "truncated" in e]
        result["survival_fraction"] = sum(bool(e["truncated"]) and not bool(e["terminated"]) for e in measured) / len(measured) if measured else None
    # Only explicit binary course_success is admissible; legacy benchmark
    # success_rate was an obstacle-clearance threshold, not course completion.
    if "course_success_rate" not in result:
        measured = [e for e in episodes if isinstance(e.get("course_success"), (bool, int))]
        result["course_success_rate"] = sum(bool(e["course_success"]) for e in measured) / len(measured) if measured else None
    result["mean_length"] = mean_present(episodes, "length")
    return result


def process_snapshot(process):
    return {"pid": process.pid, "create_time": process.create_time(), "command": process.cmdline(), "cwd": str(Path(process.cwd()).resolve())}


def is_trainer(command: list[str], cwd: str, root: Path = ROOT):
    # A shell command containing 'train.py' is not a verified Python trainer.
    if not command or "python" not in Path(command[0]).name.lower():
        return False
    for part in command[1:3]:
        candidate = Path(part)
        if not candidate.is_absolute():
            candidate = Path(cwd) / candidate
        if candidate.resolve() == (root / "train.py").resolve():
            return True
    return False


def arg_value(command, name):
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError):
        return None


def trainers(root: Path = ROOT):
    found = []
    for process in psutil.process_iter(["pid", "name"]):
        if "python" not in (process.info.get("name") or "").lower():
            continue
        try:
            snap = process_snapshot(process)
            if is_trainer(snap["command"], snap["cwd"], root):
                run = arg_value(snap["command"], "--run") or "runs/stage1"
                snap["run"] = str((Path(snap["cwd"]) / run).resolve())
                found.append(snap)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    return found


def verify_process(record: dict, root: Path = ROOT):
    """PID reuse, changed command, or changed working directory fails closed."""
    if not isinstance(record, dict) or not all(k in record for k in ("pid", "create_time", "command", "cwd")):
        return False
    try:
        actual = process_snapshot(psutil.Process(int(record["pid"])))
        return (abs(actual["create_time"] - float(record["create_time"])) < 0.01
                and actual["command"] == record["command"]
                and actual["cwd"] == str(Path(record["cwd"]).resolve())
                and is_trainer(actual["command"], actual["cwd"], root))
    except (ValueError, TypeError, psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return False


def run_state(path: Path, active: list[dict]):
    status = read_json(path / "status.json", {})
    state = status.get("state")
    matching = next((p for p in active if Path(p["run"]) == path.resolve()), None)
    if matching:
        return (state if state in ("starting", "training", "evaluating", "stopping") else "training"), status
    if state in ("completed", "stopped", "failed"):
        return state, status
    if state in ("starting", "training", "evaluating", "stopping"):
        return "interrupted", status
    performance = read_json(path / "performance.json", {})
    if finite(performance.get("steps")) and finite(performance.get("seconds")):
        return "completed", status
    return "incomplete", status


def discover_runs(root: Path = ROOT, active=None):
    active = trainers(root) if active is None else active
    found = []
    for path in (root / "runs").glob("*"):
        if not path.is_dir() or not (path / "config.yaml").exists():
            continue
        config = read_yaml(path / "config.yaml")
        raw = read_json(path / "evaluations.json", [])
        rows = [normalize_evaluation(r) for r in raw if isinstance(r, dict)] if isinstance(raw, list) else []
        state, status = run_state(path, active)
        latest = normalize_evaluation(status.get("latest_evaluation")) or (rows[-1] if rows else {})
        performance = read_json(path / "performance.json", {})
        lineage = read_json(path / "resume.json") or read_json(path / "warm_start.json") or {}
        evidence_paths = [p for p in (path / "evaluations.json", path / "status.json", path / "performance.json", path / "config.yaml") if p.exists()]
        found.append(dict(name=path.name, path=path, config=config, rows=rows, latest=latest,
                          state=state, status=status, performance=performance, lineage=lineage,
                          best=normalize_evaluation(read_json(path / "best" / "selection.json", {})),
                          updated=max(p.stat().st_mtime for p in evidence_paths),
                          utility=any(s in path.name for s in ("bench", "smoke", "test_"))))
    return sorted(found, key=lambda r: (r["state"] in ("training", "evaluating", "starting", "stopping"), r["updated"]), reverse=True)


def checkpoint_choices(run):
    path = run["path"]
    choices = []
    for label in ("best", "last", "stopped"):
        p = path / label
        if (p / "policy.zip").exists():
            choices.append(p)
    choices += sorted((path / "checkpoints").glob("update-*"), reverse=True)
    return [p for p in choices if (p / "policy.zip").exists() and (p / "vecnormalize.pkl").exists()]


def checkpoint_evidence(run, checkpoint: Path):
    metadata = read_json(checkpoint / "metadata.json", {})
    if checkpoint.name == "best":
        evaluation = normalize_evaluation(read_json(checkpoint / "selection.json", {}))
    else:
        evaluation = {}
    if not evaluation:
        evaluation = next((r for r in run["rows"] if r.get("timesteps") == metadata.get("timesteps")), {})
    update = evaluation.get("update")
    video = run["path"] / "videos" / f"update-{int(update):05d}.mp4" if update is not None else None
    return {"metadata": metadata, "evaluation": evaluation, "video": video if video and video.exists() else None}


def videos(run):
    entries = []
    for path in sorted(run["path"].glob("*.mp4")):
        entries.append({"path": path, "label": path.stem.replace("-", " "), "kind": "standalone", "evaluation": normalize_evaluation(read_json(path.with_suffix(".json"), {}))})
    for path in sorted((run["path"] / "videos").glob("*.mp4"), reverse=True):
        match = re.fullmatch(r"update-(\d+)", path.stem)
        evaluation = next((r for r in run["rows"] if match and r.get("update") == int(match[1])), {})
        entries.append({"path": path, "label": f"Training review · {path.stem}", "kind": "review", "evaluation": evaluation})
    for path in sorted((run["path"] / "benchmark").glob("*.mp4")):
        entries.append({"path": path, "label": f"Benchmark · {path.stem.replace('_', ' ')}", "kind": "benchmark", "evaluation": normalize_evaluation(read_json(path.with_suffix(".json"), {}))})
    return entries


def validate_config(content: str):
    errors, warnings = [], []
    try:
        config = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        return {}, [f"YAML syntax: {exc}"], []
    if not isinstance(config, dict):
        return {}, ["The configuration must be a YAML mapping."], []
    required = ("stage", "env_id", "seed", "total_timesteps", "n_envs", "n_steps", "batch_size", "n_epochs", "learning_rate", "gamma", "gae_lambda", "clip_range", "ent_coef", "max_grad_norm", "vf_coef", "eval_every_updates", "eval_episodes")
    errors += [f"Missing required setting: {key}" for key in required if key not in config]
    if config.get("stage") not in (1, 2, 3, 4):
        errors.append("Only stages 1–4 currently have an implemented trainer.")
    for key in ("total_timesteps", "n_envs", "n_steps", "batch_size", "n_epochs", "eval_every_updates", "eval_episodes"):
        value = config.get(key)
        if key in config and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
            errors.append(f"{key} must be a positive integer.")
    for key in ("learning_rate", "gamma", "gae_lambda", "clip_range", "ent_coef", "max_grad_norm", "vf_coef"):
        value = config.get(key)
        if key in config and (not finite(value) or value < 0):
            errors.append(f"{key} must be a finite non-negative number.")
    for key in ("gamma", "gae_lambda", "clip_range"):
        if finite(config.get(key)) and not 0 < config[key] <= 1:
            errors.append(f"{key} must be greater than zero and at most 1.")
    if all(isinstance(config.get(k), int) and config[k] > 0 for k in ("n_envs", "n_steps", "batch_size")):
        rollout = config["n_envs"] * config["n_steps"]
        if config["batch_size"] > rollout:
            errors.append("batch_size cannot exceed n_envs × n_steps.")
        elif rollout % config["batch_size"]:
            warnings.append("Rollout size is not divisible by batch size; the final minibatch will be smaller.")
    if config.get("stage") in (3, 4) and config.get("env_id") != "ParkourHumanoid":
        errors.append("Stages 3–4 require env_id: ParkourHumanoid.")
    if config.get("stage") in (1, 2) and config.get("env_id") not in ("Hopper-v5", "Walker2d-v5", "Humanoid-v5"):
        errors.append("Choose a supported Gymnasium MuJoCo environment for stages 1–2.")
    curriculum = config.get("curriculum", {})
    if not isinstance(curriculum, dict):
        errors.append("curriculum must be a mapping.")
    else:
        for key in ("replay_prob", "advance_threshold", "demote_threshold"):
            if key in curriculum and (not finite(curriculum[key]) or not 0 <= curriculum[key] <= 1):
                errors.append(f"curriculum.{key} must be between 0 and 1.")
        if finite(curriculum.get("initial_level")) and finite(curriculum.get("max_level")) and curriculum["initial_level"] > curriculum["max_level"]:
            errors.append("The initial curriculum level exceeds the maximum.")
    return config, errors, warnings


def resolve_run_name(name: str, root: Path = ROOT):
    name = name.removeprefix("runs/")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", name):
        raise ValueError("Use a run name with letters, numbers, dashes or underscores, up to 80 characters.")
    path = root / "runs" / name
    if path.exists():
        raise ValueError("This run already exists. Choose a new name to preserve its evidence.")
    return path


def validate_checkpoint(path: Path, config: dict, resume: bool, root: Path = ROOT):
    path = path.resolve()
    if not path.is_relative_to((root / "runs").resolve()):
        raise ValueError("Choose a checkpoint inside this project's runs directory.")
    for name in ("policy.zip", "vecnormalize.pkl"):
        if not (path / name).is_file():
            raise ValueError(f"Checkpoint is missing {name}.")
    meta = read_json(path / "metadata.json", {})
    if resume and meta.get("env_id") != config.get("env_id"):
        raise ValueError("Resume requires checkpoint metadata with the same environment. Use transfer for a compatible new environment.")
    for name, expected in meta.get("sha256", {}).items():
        if name not in ("policy.zip", "vecnormalize.pkl"):
            continue
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checkpoint integrity check failed for {name}.")
    return path


def launch_training(config_path: Path, run_name: str, steps: int, mode="fresh", checkpoint: str = "", apply_config=False, root: Path = ROOT):
    """Serialize launches and re-check live trainers inside the lock."""
    with (root / ".dashboard-launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if trainers(root):
            raise ValueError("A trainer is already active in this project. Finish or stop it before launching another.")
        config_path = config_path.resolve()
        if not config_path.is_relative_to((root / "configs").resolve()):
            raise ValueError("Select a config inside this project's configs directory.")
        config, errors, _ = validate_config(config_path.read_text())
        if errors:
            raise ValueError(" ".join(errors))
        run = resolve_run_name(run_name, root)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("The new training-step budget must be a positive integer.")
        if mode not in ("fresh", "resume", "transfer"):
            raise ValueError("Unknown checkpoint mode.")
        if mode == "fresh" and checkpoint:
            raise ValueError("Fresh training cannot also use a checkpoint.")
        if mode != "fresh" and not checkpoint:
            raise ValueError("Select a paired checkpoint before continuing.")
        command = [str(root / ".venv/bin/python"), str(root / "train.py"), "--config", str(config_path), "--run", str(run), "--steps", str(steps)]
        if mode != "fresh":
            checked = validate_checkpoint(Path(checkpoint), config, mode == "resume", root)
            command += ["--resume" if mode == "resume" else "--warm-start", str(checked)]
        if apply_config:
            if mode != "resume":
                raise ValueError("Applying config overrides is only valid for resume.")
            command.append("--apply-config-on-resume")
        logfile = root / "dashboard_train.log"
        with logfile.open("ab") as output:
            output.write(f"\n--- Dashboard launch {time.strftime('%Y-%m-%d %H:%M:%S')} {run.name} ---\n".encode())
            process = subprocess.Popen(command, cwd=root, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            snapshot = process_snapshot(psutil.Process(process.pid))
        except psutil.NoSuchProcess:
            raise RuntimeError("The trainer exited during startup. Check the launch log.")
        snapshot.update(run=str(run), log=str(logfile), launched_at=time.time())
        atomic_write(root / "dashboard_process.json", json.dumps(snapshot, indent=2))
        return snapshot


def stop_training(record: dict, root: Path = ROOT):
    if not verify_process(record, root):
        raise ValueError("Process ownership no longer matches. Nothing was stopped.")
    psutil.Process(record["pid"]).send_signal(signal.SIGINT)
    record = dict(record, stop_requested_at=time.time())
    atomic_write(root / "dashboard_process.json", json.dumps(record, indent=2))
    return record


def new_course():
    return append_segment({"length": 0.0, "segments": []}, "flat", 5.0)


def append_segment(course, kind, length, height=0.15, width=3.4, count=4):
    if kind not in KINDS or not finite(length) or length <= 0:
        raise ValueError("Select a valid segment type and positive length.")
    result = json.loads(json.dumps(course))
    start = float(result["length"])
    segments = result["segments"]
    # Segment.height historically describes either entry or exit. Geometry is
    # the authoritative landing elevation, including an ascending staircase.
    previous = next((s for s in reversed(segments) if s.get("geom_specs")), None)
    base = 0.0
    if previous:
        last = max(previous["geom_specs"], key=lambda g: g["pos"][0] + g["size"][0])
        base = float(last["pos"][2] + last["size"][2] - 0.05)
    index = len(segments)
    geoms = []
    def box(tag, x, z, hx, hy, hz):
        geoms.append(dict(name=f"custom_{index}_{tag}", type="box", pos=[x, 0.0, z], size=[hx, hy, hz]))
    if kind != "gap":
        if kind in ("stairs_up", "stairs_down"):
            rise = height * (1 if kind == "stairs_up" else -1)
            for i in range(count):
                box(f"step_{i}", start + (i + .5) * length / count, base + i * rise, length / count / 2, width / 2, .05)
            end_height = base + (count - 1) * rise
        else:
            box("platform", start + length / 2, base, length / 2, width / 2, .05)
            if kind == "boxes":
                box("box", start + length / 2, base + .05 + height / 2, min(.4, length / 4), min(.5, width / 2), height / 2)
            elif kind == "low_wall":
                box("wall", start + length / 2, base + .05 + height / 2, .075, width / 2, height / 2)
            end_height = base
    else:
        end_height = base
    segments.append(dict(kind=kind, x_start=start, x_end=start + length, height=end_height, geom_specs=geoms))
    result["length"] = start + length
    return result


def validate_course(course):
    errors = []
    if not isinstance(course, dict) or not isinstance(course.get("segments"), list):
        return ["A course requires a segments list and total length."]
    segments = course["segments"]
    if not segments:
        return ["Add a starting platform and at least one landing segment."]
    if segments[0].get("kind") != "flat" or segments[0].get("x_start") != 0 or segments[0].get("x_end", 0) < 5:
        errors.append("Start with a flat platform from x=0 to at least x=5 m for the humanoid spawn.")
    if segments[-1].get("kind") == "gap":
        errors.append("Add a landing platform after the final gap.")
    cursor, count, names = 0.0, 0, set()
    for i, seg in enumerate(segments):
        if seg.get("kind") not in KINDS:
            errors.append(f"Segment {i+1}: unknown terrain type.")
        a, b = seg.get("x_start"), seg.get("x_end")
        if not finite(a) or not finite(b) or a != cursor or b <= a:
            errors.append(f"Segment {i+1}: segments must be contiguous with positive lengths.")
        cursor = b
        geoms = seg.get("geom_specs", [])
        if not isinstance(geoms, list):
            errors.append(f"Segment {i+1}: geom_specs must be a list.")
            continue
        count += len(geoms)
        for geom in geoms:
            name = geom.get("name")
            if not name or name in names:
                errors.append(f"Segment {i+1}: geometry names must be unique.")
            names.add(name)
            for key in ("pos", "size"):
                values = geom.get(key)
                if not isinstance(values, list) or len(values) != 3 or not all(finite(v) for v in values):
                    errors.append(f"Segment {i+1}: {key} needs three finite numbers.")
                elif key == "size" and any(v <= 0 for v in values):
                    errors.append(f"Segment {i+1}: box dimensions must be positive.")
            if geom.get("type") != "box":
                errors.append(f"Segment {i+1}: only box geometry is supported.")
    if count > 200:
        errors.append("The course exceeds the 200-geometry simulation budget.")
    if not finite(course.get("length")) or course["length"] != cursor:
        errors.append("Course length must equal the end of the final segment.")
    return errors
