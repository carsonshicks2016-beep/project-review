from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import threading
import time
from typing import Any
from urllib.parse import quote
import uuid

from flask import Flask, Response, jsonify, request, send_from_directory

from crypt_heist.curriculum import load_curriculum_manifest, plan_curriculum_from_manifest, write_curriculum_plan
from crypt_heist.replay import load_replay, summarize_replay, validate_replay

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "command_center" / "static"
COMMANDS_PATH = ROOT / "configs" / "commands.json"
CONFIG_DIR = ROOT / "configs"
REPLAY_DIR = ROOT / "replays"
LOG_DIR = ROOT / "logs"
CHECKPOINT_DIR = ROOT / "checkpoints"
CURRICULUM_PLAN_PATH = LOG_DIR / "curriculum_plan.json"
CURRICULUM_HISTORY_PATH = LOG_DIR / "curriculum_history.json"
ACCEPTANCE_MANIFEST_PATH = LOG_DIR / "acceptance_scenarios.json"
STRICT_ACCEPTANCE_MANIFEST_PATH = LOG_DIR / "acceptance_full_targets.json"
EVADER_CHECKPOINT_ACCEPTANCE_PATH = LOG_DIR / "evader_checkpoint_acceptance.json"
PURSUER_TEAM_CHECKPOINT_ACCEPTANCE_PATH = LOG_DIR / "pursuer_team_checkpoint_acceptance.json"
ACTIVE_CONTROL_ACCEPTANCE_PATH = LOG_DIR / "control_acceptance_active.json"
CHECKPOINT_LEAGUE_MANIFEST_PATH = LOG_DIR / "checkpoint_league_manifest.json"
ACTIVE_INFORMATION_MANIFEST_PATH = ROOT / "models" / "active" / "manifest.json"
REPLAY_FIDELITY_MANIFEST_PATH = LOG_DIR / "replay_fidelity.json"
PHYSICS_VALIDATION_MANIFEST_PATH = LOG_DIR / "physics_validation.json"
ENV_VALIDATION_MANIFEST_PATH = LOG_DIR / "env_validation.json"
SPECTATOR_VALIDATION_MANIFEST_PATH = LOG_DIR / "spectator_validation.json"
OPERATIONAL_READINESS_MANIFEST_PATH = LOG_DIR / "operational_readiness.json"
EVIDENCE_BUNDLE_PATH = LOG_DIR / "evidence_bundle.json"
EVIDENCE_BUNDLE_VERIFICATION_PATH = LOG_DIR / "evidence_bundle_verification.json"
SELF_PLAY_MANIFEST_PATH = LOG_DIR / "selfplay" / "selfplay_scenario_selfplay_manifest.json"
ACCEPTANCE_COMMAND_IDS = {"eval_acceptance_scenarios", "eval_acceptance_full_targets"}
READINESS_CHECKPOINTS = (
    {
        "id": "evader_ppo",
        "label": "Evader PPO driver",
        "path": CHECKPOINT_DIR / "evader_ppo.pt",
        "command_id": "train_evader_ppo",
        "lane": "evader_waypoint_driver",
        "required_for": "learned control",
    },
    {
        "id": "pursuer_team_ppo",
        "label": "Five-pursuer team PPO",
        "path": CHECKPOINT_DIR / "pursuer_team_ppo.pt",
        "command_id": "train_pursuer_team_ppo",
        "lane": "pursuer_team_containment",
        "required_for": "learned containment",
    },
    {
        "id": "scanner_decoder",
        "label": "Scanner decoder",
        "path": CHECKPOINT_DIR / "scanner_decoder.pt",
        "command_id": "train_scanner_decoder",
        "lane": "scanner_decoder",
        "required_for": "evader decryptor confidence",
    },
    {
        "id": "radio_policy",
        "label": "Pursuer radio policy",
        "path": CHECKPOINT_DIR / "radio_policy.pt",
        "command_id": "train_radio_policy",
        "lane": "radio_authentication_probe",
        "required_for": "learned English-token comms",
    },
    {
        "id": "jammer_policy",
        "label": "Evader jammer policy",
        "path": CHECKPOINT_DIR / "jammer_policy.pt",
        "command_id": "train_jammer_policy",
        "lane": "evader_jammer",
        "required_for": "learned spoof bursts",
    },
)

app = Flask(__name__, static_folder=str(STATIC), static_url_path="")
jobs_lock = threading.Lock()
jobs: dict[str, "Job"] = {}


@dataclass
class Job:
    id: str
    command_id: str
    label: str
    argv: list[str]
    cwd: str
    status: str = "queued"
    returncode: int | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    log: list[str] = field(default_factory=list)
    process: subprocess.Popen | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_log: bool = False):
        data = {
            "id": self.id,
            "command_id": self.command_id,
            "label": self.label,
            "argv": self.argv,
            "cwd": self.cwd,
            "status": self.status,
            "returncode": self.returncode,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration": (self.finished_at or time.time()) - self.started_at,
        }
        if include_log:
            data["log"] = "".join(self.log[-2000:])
        return data


def load_command_registry() -> dict:
    with COMMANDS_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def find_command(command_id: str) -> dict:
    registry = load_command_registry()
    for command in registry.get("commands", []):
        if command.get("id") == command_id:
            return command
    raise KeyError(command_id)


def safe_split(args_line: str) -> list[str]:
    args_line = args_line.strip()
    return shlex.split(args_line) if args_line else []


def run_job(job: Job, env_extra: dict | None = None):
    env = os.environ.copy()
    env.update(env_extra or {})
    env["PYTHONUNBUFFERED"] = "1"
    with jobs_lock:
        job.status = "running"
        job.log.append(f"$ {' '.join(shlex.quote(x) for x in job.argv)}\n")
        job.log.append(f"[cwd] {job.cwd}\n")
    try:
        proc = subprocess.Popen(
            job.argv,
            cwd=job.cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        job.process = proc
        assert proc.stdout is not None
        for line in proc.stdout:
            with jobs_lock:
                job.log.append(line)
                if len(job.log) > 4000:
                    del job.log[:1000]
        rc = proc.wait()
        proc.stdout.close()
        finished_at = time.time()
        postprocess_log: list[str] = []
        if rc == 0:
            postprocess_log = _postprocess_completed_job(job)
        history_log = _record_completed_curriculum_job(job, rc, finished_at)
        with jobs_lock:
            job.returncode = rc
            job.status = "succeeded" if rc == 0 else "failed"
            job.finished_at = finished_at
            job.log.extend(postprocess_log)
            job.log.extend(history_log)
            job.log.append(f"\n[exit] {rc}\n")
    except Exception as exc:
        with jobs_lock:
            job.status = "failed"
            job.returncode = -1
            job.finished_at = time.time()
            job.log.append(f"\n[dashboard error] {type(exc).__name__}: {exc}\n")


@app.get("/")
def index():
    return Response((STATIC / "index.html").read_text(encoding="utf-8"), mimetype="text/html")


@app.get("/api/commands")
def api_commands():
    return jsonify(load_command_registry())


@app.post("/api/run")
def api_run():
    payload = request.get_json(force=True) or {}
    command_id = payload.get("id")
    args_line = payload.get("args", "")
    command = find_command(command_id)
    argv = list(command["base"]) + safe_split(args_line)
    job = Job(
        id=uuid.uuid4().hex[:12],
        command_id=command_id,
        label=command.get("label", command_id),
        argv=argv,
        cwd=str(ROOT),
    )
    with jobs_lock:
        jobs[job.id] = job
    thread = threading.Thread(target=run_job, args=(job, command.get("env")), daemon=True)
    thread.start()
    return jsonify(job.to_dict(include_log=True))


@app.post("/api/run_replay_action")
def api_run_replay_action():
    payload = request.get_json(force=True) or {}
    action = payload.get("action")
    replay = payload.get("replay")
    path = _replay_path(replay)
    if action == "play":
        command_id = "play_replay"
        args_line = shlex.quote(str(path.relative_to(ROOT)))
    elif action == "validate":
        command_id = "summarize_replay"
        args_line = shlex.quote(str(path.relative_to(ROOT)))
    elif action == "report":
        command_id = "replay_report"
        out = _replay_report_path(path)
        args_line = f"{shlex.quote(str(path.relative_to(ROOT)))} --json-out {shlex.quote(str(out.relative_to(ROOT)))}"
    else:
        return jsonify({"error": "unknown replay action"}), 400
    command = find_command(command_id)
    argv = list(command["base"]) + safe_split(args_line)
    job = Job(
        id=uuid.uuid4().hex[:12],
        command_id=command_id,
        label=f"{command.get('label', command_id)}: {path.name}",
        argv=argv,
        cwd=str(ROOT),
    )
    with jobs_lock:
        jobs[job.id] = job
    thread = threading.Thread(target=run_job, args=(job, command.get("env")), daemon=True)
    thread.start()
    return jsonify(job.to_dict(include_log=True))


@app.get("/api/curriculum_plan")
def api_curriculum_plan():
    if not CURRICULUM_PLAN_PATH.exists():
        return jsonify({
            "exists": False,
            "path": _display_path(CURRICULUM_PLAN_PATH),
            "plan": None,
            "history": _curriculum_history_summary(None),
        })
    plan = json.loads(CURRICULUM_PLAN_PATH.read_text(encoding="utf-8"))
    return jsonify({
        "exists": True,
        "path": _display_path(CURRICULUM_PLAN_PATH),
        "modified": datetime.fromtimestamp(CURRICULUM_PLAN_PATH.stat().st_mtime).isoformat(timespec="seconds"),
        "plan": plan,
        "history": _curriculum_history_summary(plan),
    })


@app.post("/api/run_curriculum_action")
def api_run_curriculum_action():
    if not CURRICULUM_PLAN_PATH.exists():
        return jsonify({"error": "missing curriculum plan"}), 404
    payload = request.get_json(force=True) or {}
    index = int(payload.get("index", 0))
    plan = json.loads(CURRICULUM_PLAN_PATH.read_text(encoding="utf-8"))
    actions = plan.get("actions", [])
    if index < 0 or index >= len(actions):
        return jsonify({"error": "invalid action index"}), 400
    action = actions[index]
    command_id = action.get("command_id")
    command = find_command(command_id)
    argv = list(command["base"]) + safe_split(str(action.get("default_args", "")))
    label = f"Curriculum: {action.get('lane', command.get('label', command_id))}"
    job = Job(
        id=uuid.uuid4().hex[:12],
        command_id=command_id,
        label=label,
        argv=argv,
        cwd=str(ROOT),
        metadata={
            "curriculum_kind": "repair",
            "action_index": index,
            "lane": action.get("lane"),
            "diagnostic_reasons": list(action.get("diagnostic_reasons", [])),
            "scenarios": list(action.get("scenarios", [])),
            "plan_command_id": command_id,
        },
    )
    with jobs_lock:
        jobs[job.id] = job
    thread = threading.Thread(target=run_job, args=(job, command.get("env")), daemon=True)
    thread.start()
    return jsonify(job.to_dict(include_log=True))


@app.post("/api/run_curriculum_followup")
def api_run_curriculum_followup():
    if not CURRICULUM_PLAN_PATH.exists():
        return jsonify({"error": "missing curriculum plan"}), 404
    payload = request.get_json(force=True) or {}
    index = int(payload.get("index", 0))
    followup_id = str(payload.get("command_id", ""))
    plan = json.loads(CURRICULUM_PLAN_PATH.read_text(encoding="utf-8"))
    actions = plan.get("actions", [])
    if index < 0 or index >= len(actions):
        return jsonify({"error": "invalid action index"}), 400
    action = actions[index]
    allowed = set(str(item) for item in action.get("followup_command_ids", []))
    if followup_id not in allowed:
        return jsonify({"error": "follow-up not listed for this action"}), 400
    command = find_command(followup_id)
    argv = list(command["base"]) + safe_split(str(command.get("default_args", "")))
    label = f"Follow-up: {command.get('label', followup_id)}"
    job = Job(
        id=uuid.uuid4().hex[:12],
        command_id=followup_id,
        label=label,
        argv=argv,
        cwd=str(ROOT),
        metadata={
            "curriculum_kind": "followup",
            "action_index": index,
            "lane": action.get("lane"),
            "diagnostic_reasons": list(action.get("diagnostic_reasons", [])),
            "scenarios": list(action.get("scenarios", [])),
            "plan_command_id": action.get("command_id"),
            "followup_command_id": followup_id,
        },
    )
    with jobs_lock:
        jobs[job.id] = job
    thread = threading.Thread(target=run_job, args=(job, command.get("env")), daemon=True)
    thread.start()
    return jsonify(job.to_dict(include_log=True))


def _postprocess_completed_job(job: Job) -> list[str]:
    if job.command_id not in ACCEPTANCE_COMMAND_IDS:
        return []
    lines: list[str] = []
    manifest_path = _manifest_path_from_argv(job.argv)
    if manifest_path is None:
        return ["[dashboard] curriculum plan skipped: could not resolve acceptance manifest path\n"]
    if not manifest_path.exists():
        return [f"[dashboard] curriculum plan skipped: manifest not found at {_display_path(manifest_path)}\n"]
    try:
        manifest = load_curriculum_manifest(manifest_path)
        plan = plan_curriculum_from_manifest(manifest, source=_display_path(manifest_path))
        out = write_curriculum_plan(plan, CURRICULUM_PLAN_PATH)
    except Exception as exc:
        return [f"[dashboard] curriculum plan failed: {type(exc).__name__}: {exc}\n"]
    actions = plan.get("actions", [])
    top = actions[0] if actions else {}
    lines.append(
        "[dashboard] curriculum plan updated: "
        f"{_display_path(out)} status={plan.get('status')} diagnostics={plan.get('diagnostic_count')}\n"
    )
    if top:
        lines.append(
            "[dashboard] next curriculum action: "
            f"{top.get('command_id')} lane={top.get('lane')} priority={top.get('priority')}\n"
        )
    history = _append_acceptance_history(job, manifest_path, plan)
    lines.append(
        "[dashboard] curriculum history updated: "
        f"acceptance diagnostics={history.get('diagnostic_count')} trend={history.get('diagnostic_trend')}\n"
    )
    return lines


def _append_acceptance_history(job: Job, manifest_path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    previous = _latest_history_event(kind="acceptance")
    previous_count = previous.get("diagnostic_count") if previous else None
    current_count = int(plan.get("diagnostic_count", 0))
    change = None if previous_count is None else current_count - int(previous_count)
    top = (plan.get("actions") or [{}])[0]
    record = {
        "kind": "acceptance",
        "timestamp": _timestamp(),
        "job_id": job.id,
        "command_id": job.command_id,
        "label": job.label,
        "status": "succeeded",
        "returncode": 0,
        "manifest": _display_path(manifest_path),
        "plan_status": plan.get("status"),
        "diagnostic_count": current_count,
        "previous_diagnostic_count": previous_count,
        "diagnostic_count_change": change,
        "diagnostic_trend": _trend_from_change(change),
        "top_lane": top.get("lane"),
        "top_command_id": top.get("command_id"),
        "top_priority": top.get("priority"),
        "top_reasons": list(top.get("diagnostic_reasons", [])),
    }
    _append_curriculum_history(record)
    return record


def _record_completed_curriculum_job(job: Job, rc: int, finished_at: float) -> list[str]:
    kind = job.metadata.get("curriculum_kind")
    if kind not in {"repair", "followup"}:
        return []
    record = {
        "kind": kind,
        "timestamp": _timestamp(finished_at),
        "job_id": job.id,
        "command_id": job.command_id,
        "label": job.label,
        "status": "succeeded" if rc == 0 else "failed",
        "returncode": int(rc),
        "duration": float(finished_at - job.started_at),
        "lane": job.metadata.get("lane"),
        "action_index": job.metadata.get("action_index"),
        "plan_command_id": job.metadata.get("plan_command_id"),
        "followup_command_id": job.metadata.get("followup_command_id"),
        "diagnostic_reasons": list(job.metadata.get("diagnostic_reasons", [])),
        "scenarios": list(job.metadata.get("scenarios", [])),
    }
    _append_curriculum_history(record)
    return [
        "[dashboard] curriculum history updated: "
        f"{record['kind']} {record['command_id']} status={record['status']}\n"
    ]


def _curriculum_history_summary(plan: dict[str, Any] | None) -> dict[str, Any]:
    history = _read_curriculum_history()
    events = list(history.get("events", []))
    recent = events[-12:]
    by_lane: dict[str, dict[str, Any]] = {}
    for action in (plan or {}).get("actions", []):
        lane = str(action.get("lane", ""))
        if not lane:
            continue
        by_lane[lane] = {
            "latest_repair": _latest_history_event(kind="repair", lane=lane, events=events),
            "latest_followup": _latest_history_event(kind="followup", lane=lane, events=events),
            "latest_acceptance": _latest_history_event(kind="acceptance", top_lane=lane, events=events)
            or _latest_history_event(kind="acceptance", events=events),
            "recent": _recent_lane_events(lane, events, limit=5),
        }
    return {
        "path": _display_path(CURRICULUM_HISTORY_PATH),
        "count": len(events),
        "latest_acceptance": _latest_history_event(kind="acceptance", events=events),
        "recent": recent,
        "by_lane": by_lane,
    }


def _recent_lane_events(lane: str, events: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in reversed(events):
        if event.get("lane") == lane or event.get("top_lane") == lane:
            out.append(event)
        if len(out) >= limit:
            break
    return list(reversed(out))


def _append_curriculum_history(record: dict[str, Any]) -> None:
    history = _read_curriculum_history()
    events = list(history.get("events", []))
    events.append(record)
    history = {"version": 1, "events": events[-100:]}
    CURRICULUM_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    CURRICULUM_HISTORY_PATH.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_curriculum_history() -> dict[str, Any]:
    if not CURRICULUM_HISTORY_PATH.exists():
        return {"version": 1, "events": []}
    try:
        history = json.loads(CURRICULUM_HISTORY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"version": 1, "events": []}
    if not isinstance(history.get("events"), list):
        history["events"] = []
    return history


def _latest_history_event(
    *,
    kind: str,
    lane: str | None = None,
    top_lane: str | None = None,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    source = events if events is not None else list(_read_curriculum_history().get("events", []))
    for event in reversed(source):
        if event.get("kind") != kind:
            continue
        if lane is not None and event.get("lane") != lane:
            continue
        if top_lane is not None and event.get("top_lane") != top_lane:
            continue
        return event
    return None


def _timestamp(value: float | None = None) -> str:
    return datetime.fromtimestamp(value or time.time()).isoformat(timespec="seconds")


def _trend_from_change(change: int | None) -> str:
    if change is None:
        return "unknown"
    if change < 0:
        return "improved"
    if change > 0:
        return "regressed"
    return "unchanged"


def _manifest_path_from_argv(argv: list[str]) -> Path | None:
    out = _flag_value(argv, "--out") or "logs/acceptance_scenarios.json"
    path = Path(out)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    try:
        path.relative_to(ROOT)
    except ValueError:
        return None
    return path


def _flag_value(argv: list[str], flag: str) -> str | None:
    for index, item in enumerate(argv):
        if item == flag and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith(f"{flag}="):
            return item.split("=", 1)[1]
    return None


@app.get("/api/jobs")
def api_jobs():
    with jobs_lock:
        ordered = sorted(jobs.values(), key=lambda job: job.started_at, reverse=True)
        return jsonify([job.to_dict(include_log=False) for job in ordered])


@app.get("/api/jobs/<job_id>")
def api_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return jsonify({"error": "unknown job"}), 404
        return jsonify(job.to_dict(include_log=True))


@app.post("/api/jobs/<job_id>/stop")
def api_stop_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return jsonify({"error": "unknown job"}), 404
        proc = job.process
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        return jsonify({"ok": True})
    return jsonify({"ok": False, "reason": "not running"})


@app.get("/api/replays")
def api_replays():
    return jsonify(_replay_items())


@app.get("/api/replay")
def api_replay():
    replay_name = str(request.args.get("path") or request.args.get("replay") or "").strip()
    if not replay_name:
        return jsonify({"error": "missing replay path"}), 400
    try:
        max_frames = int(request.args.get("max_frames", 2400))
        stride_arg = request.args.get("stride")
        requested_stride = int(stride_arg) if stride_arg is not None else None
    except ValueError:
        return jsonify({"error": "invalid frame sampling parameter"}), 400
    max_frames = min(max(1, max_frames), 12000)
    if requested_stride is not None and requested_stride < 1:
        return jsonify({"error": "stride must be >= 1"}), 400
    try:
        path = _replay_path(replay_name)
        lines = load_replay(path)
        validate_replay(lines)
        summary = summarize_replay(lines, include_fingerprint=False)
    except FileNotFoundError:
        return jsonify({"error": "replay not found"}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    frames = lines[1:]
    stride = requested_stride or max(1, (len(frames) + max_frames - 1) // max_frames)
    sampled = frames[::stride]
    return jsonify({
        "name": str(path.relative_to(REPLAY_DIR)),
        "path": _display_path(path),
        "meta": lines[0],
        "summary": summary,
        "frames": sampled,
        "frame_count": len(frames),
        "returned_frames": len(sampled),
        "stride": stride,
    })


@app.get("/api/metrics")
def api_metrics():
    replay_items = _replay_items()
    valid = [item for item in replay_items if item.get("valid")]
    running_jobs = 0
    failed_jobs = 0
    succeeded_jobs = 0
    with jobs_lock:
        for job in jobs.values():
            running_jobs += int(job.status == "running")
            failed_jobs += int(job.status == "failed")
            succeeded_jobs += int(job.status == "succeeded")
    total_duration = sum(float(item.get("duration", 0.0)) for item in valid)
    total_frames = sum(int(item.get("frames", 0)) for item in valid)
    total_radio = sum(int(item.get("radio_events", 0)) for item in valid)
    total_spoofed = sum(int(item.get("spoofed_radio_events", 0)) for item in valid)
    total_unique_radio = sum(int(item.get("unique_radio_events", 0)) for item in valid)
    total_unique_spoofed = sum(int(item.get("unique_spoofed_radio_events", 0)) for item in valid)
    total_captures = sum(int(item.get("captures", 0)) for item in valid)
    total_waypoints = sum(int(item.get("waypoints_hit", 0)) for item in valid)
    total_deception = sum(float(item.get("deception_score", 0.0)) for item in valid)
    total_evader_reward = sum(float(item.get("total_evader_reward", 0.0)) for item in valid)
    total_pursuer_reward = sum(float(item.get("total_pursuer_reward", 0.0)) for item in valid)
    total_jam_events = sum(int(item.get("jam_events", 0)) for item in valid)
    total_cipher_rotations = sum(int(item.get("cipher_rotations", 0)) for item in valid)
    avg_conf = (
        sum(float(item.get("avg_confidence", 0.0)) * int(item.get("frames", 0)) for item in valid)
        / max(1, total_frames)
    )
    avg_evader_reward = _weighted_replay_average(valid, "avg_evader_reward")
    avg_pursuer_reward = _weighted_replay_average(valid, "avg_pursuer_reward")
    avg_decoder_accuracy = _weighted_replay_average(valid, "avg_decoder_accuracy")
    avg_spoof_susceptibility = _weighted_replay_average(valid, "avg_spoof_susceptibility")
    avg_evader_information_reward = _weighted_replay_average(valid, "avg_evader_information_reward")
    avg_pursuer_auth_penalty = _weighted_replay_average(valid, "avg_pursuer_auth_penalty")
    avg_radio_word_entropy = _weighted_replay_average(valid, "radio_word_entropy")
    avg_pursuer_word_entropy = _weighted_replay_average(valid, "pursuer_word_entropy")
    avg_spoof_word_entropy = _weighted_replay_average(valid, "spoof_word_entropy")
    avg_radio_spoof_ratio = _weighted_replay_average(valid, "radio_spoof_ratio")
    avg_jam_confidence_drop = _weighted_replay_average(valid, "avg_jam_confidence_drop")
    avg_cipher_confidence_drop = _weighted_replay_average(valid, "avg_cipher_confidence_drop")
    avg_cipher_confidence_recovery = _weighted_replay_average(valid, "avg_cipher_confidence_recovery")
    return jsonify({
        "commands": len(load_command_registry().get("commands", [])),
        "replays": len(valid),
        "replay_duration": total_duration,
        "replay_frames": total_frames,
        "radio_events": total_radio,
        "spoofed_radio_events": total_spoofed,
        "unique_radio_events": total_unique_radio,
        "unique_spoofed_radio_events": total_unique_spoofed,
        "captures": total_captures,
        "waypoints": total_waypoints,
        "deception": total_deception,
        "avg_confidence": avg_conf,
        "total_evader_reward": total_evader_reward,
        "total_pursuer_reward": total_pursuer_reward,
        "jam_events": total_jam_events,
        "cipher_rotations": total_cipher_rotations,
        "avg_evader_reward": avg_evader_reward,
        "avg_pursuer_reward": avg_pursuer_reward,
        "avg_decoder_accuracy": avg_decoder_accuracy,
        "avg_spoof_susceptibility": avg_spoof_susceptibility,
        "avg_evader_information_reward": avg_evader_information_reward,
        "avg_pursuer_auth_penalty": avg_pursuer_auth_penalty,
        "avg_radio_word_entropy": avg_radio_word_entropy,
        "avg_pursuer_word_entropy": avg_pursuer_word_entropy,
        "avg_spoof_word_entropy": avg_spoof_word_entropy,
        "avg_radio_spoof_ratio": avg_radio_spoof_ratio,
        "avg_jam_confidence_drop": avg_jam_confidence_drop,
        "avg_cipher_confidence_drop": avg_cipher_confidence_drop,
        "avg_cipher_confidence_recovery": avg_cipher_confidence_recovery,
        "jobs_running": running_jobs,
        "jobs_failed": failed_jobs,
        "jobs_succeeded": succeeded_jobs,
    })


@app.get("/api/readiness")
def api_readiness():
    return jsonify(_readiness_summary())


def _readiness_summary() -> dict[str, Any]:
    acceptance = _acceptance_manifest_summary(ACCEPTANCE_MANIFEST_PATH)
    strict_acceptance = _acceptance_manifest_summary(STRICT_ACCEPTANCE_MANIFEST_PATH)
    evader_checkpoint_acceptance = _acceptance_manifest_summary(EVADER_CHECKPOINT_ACCEPTANCE_PATH)
    pursuer_team_checkpoint_acceptance = _acceptance_manifest_summary(PURSUER_TEAM_CHECKPOINT_ACCEPTANCE_PATH)
    active_control_acceptance = _acceptance_manifest_summary(ACTIVE_CONTROL_ACCEPTANCE_PATH)
    checkpoint_league = _information_manifest_summary(CHECKPOINT_LEAGUE_MANIFEST_PATH)
    active_information = _information_manifest_summary(ACTIVE_INFORMATION_MANIFEST_PATH)
    replay_fidelity = _replay_fidelity_manifest_summary(REPLAY_FIDELITY_MANIFEST_PATH)
    physics_validation = _physics_validation_manifest_summary(PHYSICS_VALIDATION_MANIFEST_PATH)
    env_validation = _env_validation_manifest_summary(ENV_VALIDATION_MANIFEST_PATH)
    spectator_validation = _spectator_validation_manifest_summary(SPECTATOR_VALIDATION_MANIFEST_PATH)
    operational_validation = _operational_validation_manifest_summary(OPERATIONAL_READINESS_MANIFEST_PATH)
    evidence_bundle = _evidence_bundle_summary(EVIDENCE_BUNDLE_PATH)
    evidence_bundle_verification = _evidence_bundle_verification_summary(EVIDENCE_BUNDLE_VERIFICATION_PATH)
    self_play = _self_play_manifest_summary(SELF_PLAY_MANIFEST_PATH)
    plan = _read_json(CURRICULUM_PLAN_PATH)
    checkpoints = _checkpoint_items()
    replay_items = _replay_items(validate=False)
    valid_replays = [item for item in replay_items if item.get("valid")]
    acceptance_replays = [
        item for item in valid_replays
        if str(item.get("path", "")).startswith("replays/acceptance/")
    ]
    next_action = _next_curriculum_action(plan)
    missing_checkpoints = [item for item in checkpoints if not item["exists"]]
    learned_gates = {
        "evader_checkpoint_acceptance": evader_checkpoint_acceptance,
        "pursuer_team_checkpoint_acceptance": pursuer_team_checkpoint_acceptance,
        "active_control_acceptance": active_control_acceptance,
    }
    information_gates = {
        "checkpoint_league": checkpoint_league,
        "active_information": active_information,
    }
    blockers = _readiness_blockers(
        acceptance,
        strict_acceptance,
        plan,
        missing_checkpoints,
        learned_gates,
        information_gates,
        replay_fidelity,
        physics_validation,
        env_validation,
        spectator_validation,
        operational_validation,
        evidence_bundle,
        evidence_bundle_verification,
        self_play,
    )
    status = _readiness_status(
        acceptance,
        strict_acceptance,
        plan,
        missing_checkpoints,
        learned_gates,
        information_gates,
        replay_fidelity,
        physics_validation,
        env_validation,
        spectator_validation,
        operational_validation,
        evidence_bundle,
        evidence_bundle_verification,
        self_play,
    )
    return {
        "version": 1,
        "status": status,
        "summary": _readiness_summary_text(
            status,
            acceptance,
            strict_acceptance,
            plan,
            missing_checkpoints,
            learned_gates,
            information_gates,
            replay_fidelity,
            physics_validation,
            env_validation,
            spectator_validation,
            operational_validation,
            evidence_bundle,
            evidence_bundle_verification,
            self_play,
        ),
        "acceptance": acceptance,
        "strict_acceptance": strict_acceptance,
        "learned_control": learned_gates,
        "learned_information": information_gates,
        "physics_validation": physics_validation,
        "env_validation": env_validation,
        "curriculum": {
            "exists": plan is not None,
            "path": _display_path(CURRICULUM_PLAN_PATH),
            "status": plan.get("status") if plan else "missing",
            "diagnostic_count": int(plan.get("diagnostic_count", 0)) if plan else None,
            "summary": plan.get("summary") if plan else "Run Acceptance Scenarios to generate a repair plan.",
            "next_action": next_action,
        },
        "checkpoints": {
            "total": len(checkpoints),
            "existing": len(checkpoints) - len(missing_checkpoints),
            "missing": len(missing_checkpoints),
            "items": checkpoints,
        },
        "replay_fidelity": replay_fidelity,
        "spectator_validation": spectator_validation,
        "operational_validation": operational_validation,
        "evidence_bundle": evidence_bundle,
        "evidence_bundle_verification": evidence_bundle_verification,
        "self_play": self_play,
        "replays": {
            "valid": len(valid_replays),
            "acceptance": len(acceptance_replays),
            "latest": valid_replays[0] if valid_replays else None,
        },
        "blockers": blockers,
        "recommended_command": _recommended_command(
            acceptance,
            plan,
            missing_checkpoints,
            learned_gates,
            information_gates,
            replay_fidelity,
            physics_validation,
            env_validation,
            spectator_validation,
            operational_validation,
            evidence_bundle,
            evidence_bundle_verification,
            self_play,
        ),
    }


def _acceptance_manifest_summary(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest is None:
        return {
            "exists": False,
            "path": _display_path(path),
            "status": "missing",
            "summary": "No acceptance manifest has been recorded.",
        }
    score = manifest.get("score") or {}
    diagnostics = manifest.get("diagnostics") or {}
    top = diagnostics.get("top") or []
    scenarios = []
    for row in manifest.get("scenarios", []):
        scenarios.append({
            "name": row.get("name"),
            "passed": bool(row.get("passed")),
            "required_checks_passed": int(row.get("required_checks_passed", 0)),
            "required_checks": int(row.get("required_checks", 0)),
            "milestone_targets_passed": int(row.get("milestone_targets_passed", 0)),
            "milestone_targets": int(row.get("milestone_targets", 0)),
            "replay": row.get("replay"),
            "primary_reason": (row.get("primary_diagnostic") or {}).get("reason"),
        })
    return {
        "exists": True,
        "path": _display_path(path),
        "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "status": "passed" if manifest.get("passed") else "failed",
        "passed": bool(manifest.get("passed")),
        "required_checks_passed": int(manifest.get("required_checks_passed", 0)),
        "required_checks": int(manifest.get("required_checks", 0)),
        "milestone_targets_passed": int(manifest.get("milestone_targets_passed", 0)),
        "milestone_targets": int(manifest.get("milestone_targets", 0)),
        "overall_score": float(score.get("overall_score", 0.0)),
        "required_score": float(score.get("required_score", 0.0)),
        "milestone_score": float(score.get("milestone_score", 0.0)),
        "diagnostic_count": int(diagnostics.get("count", 0)),
        "top_diagnostics": top[:5],
        "scenarios": scenarios,
    }


def _checkpoint_items() -> list[dict[str, Any]]:
    items = []
    for spec in READINESS_CHECKPOINTS:
        path = Path(spec["path"])
        item = {
            "id": spec["id"],
            "label": spec["label"],
            "path": _display_path(path),
            "command_id": spec["command_id"],
            "lane": spec["lane"],
            "required_for": spec["required_for"],
            "exists": path.exists(),
        }
        if path.exists():
            stat = path.stat()
            item["size"] = stat.st_size
            item["modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
        items.append(item)
    return items


def _information_manifest_summary(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest is None:
        return {
            "exists": False,
            "path": _display_path(path),
            "status": "missing",
            "summary": "No learned information manifest has been recorded.",
        }
    scores = manifest.get("scores") or {}
    aggregates = manifest.get("aggregates") or {}
    overall = float(scores.get("overall_score", 0.0))
    deception_lift = float(aggregates.get("mean_deception_lift", 0.0))
    spoof_lift = float(aggregates.get("mean_spoof_event_lift", 0.0))
    passed = overall >= 0.52 and (deception_lift > 0.0 or spoof_lift > 0.0)
    checkpoints = manifest.get("active_checkpoints") or manifest.get("checkpoint_set") or {}
    return {
        "exists": True,
        "path": _display_path(path),
        "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "status": "passed" if passed else "failed",
        "passed": bool(passed),
        "overall_score": overall,
        "pursuer_security_score": float(scores.get("pursuer_security_score", 0.0)),
        "evader_pressure_score": float(scores.get("evader_pressure_score", 0.0)),
        "adversarial_balance_score": float(scores.get("adversarial_balance_score", 0.0)),
        "mean_deception_lift": deception_lift,
        "mean_spoof_event_lift": spoof_lift,
        "mean_spoof_susceptibility": float(aggregates.get("mean_spoof_susceptibility", 0.0)),
        "checkpoints": checkpoints,
    }


def _replay_fidelity_manifest_summary(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest is None:
        return {
            "exists": False,
            "path": _display_path(path),
            "status": "missing",
            "summary": "No replay fidelity manifest has been recorded.",
        }
    summary = manifest.get("summary") or {}
    fingerprint = manifest.get("fingerprint") or {}
    scripted = manifest.get("scripted_determinism") or {}
    compare = manifest.get("compare") or {}
    state_hash = str(fingerprint.get("state_sha256", ""))
    radio_hash = str(fingerprint.get("radio_sha256", ""))
    has_hashes = bool(state_hash and radio_hash)
    scripted_ok = bool(scripted.get("matched", True))
    compare_ok = bool(compare.get("matched", True))
    passed = bool(manifest.get("valid", False)) and bool(manifest.get("passed", True)) and has_hashes and scripted_ok and compare_ok
    return {
        "exists": True,
        "path": _display_path(path),
        "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "replay": manifest.get("replay"),
        "frames": int(summary.get("frames", fingerprint.get("frames", 0))),
        "schema_version": int(fingerprint.get("schema_version", summary.get("schema_version", 0))),
        "state_sha256": state_hash,
        "radio_sha256": radio_hash,
        "state_sha256_short": state_hash[:16],
        "radio_sha256_short": radio_hash[:16],
        "scripted_determinism_matched": scripted.get("matched"),
        "scripted_seed": scripted.get("seed"),
        "scripted_steps": scripted.get("steps"),
        "compare_matched": compare.get("matched"),
    }


def _physics_validation_manifest_summary(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest is None:
        return {
            "exists": False,
            "path": _display_path(path),
            "status": "missing",
            "summary": "No physics validation manifest has been recorded.",
        }
    results = manifest.get("results") or []
    by_name = {str(item.get("name")): item for item in results}
    asymmetry = by_name.get("handbrake_asymmetry", {}).get("metrics", {})
    deterministic = by_name.get("deterministic_trace", {}).get("metrics", {})
    passed = bool(manifest.get("passed", False))
    return {
        "exists": True,
        "path": _display_path(path),
        "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "checks_passed": int(manifest.get("checks_passed", 0)),
        "checks": int(manifest.get("checks", len(results))),
        "score": float(manifest.get("score", 0.0)),
        "yaw_ratio": float(asymmetry.get("yaw_ratio", 0.0)),
        "radius_ratio": float(asymmetry.get("radius_ratio", 0.0)),
        "speed_ratio": float(asymmetry.get("speed_ratio", 0.0)),
        "max_state_delta": float(deterministic.get("max_state_delta", 0.0)),
        "failed_checks": [str(item.get("name")) for item in results if not item.get("passed")],
    }


def _env_validation_manifest_summary(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest is None:
        return {
            "exists": False,
            "path": _display_path(path),
            "status": "missing",
            "summary": "No MARL environment validation manifest has been recorded.",
        }
    results = manifest.get("results") or []
    by_name = {str(item.get("name")): item for item in results}
    pettingzoo = by_name.get("pettingzoo_parallel_api", {})
    determinism = by_name.get("seeded_env_determinism", {})
    passed = bool(manifest.get("passed", False))
    return {
        "exists": True,
        "path": _display_path(path),
        "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "checks_passed": int(manifest.get("checks_passed", 0)),
        "checks": int(manifest.get("checks", len(results))),
        "score": float(manifest.get("score", 0.0)),
        "pettingzoo_parallel_api": bool(pettingzoo.get("passed")),
        "seeded_env_determinism": bool(determinism.get("passed")),
        "failed_checks": [str(item.get("name")) for item in results if not item.get("passed")],
    }


def _spectator_validation_manifest_summary(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest is None:
        return {
            "exists": False,
            "path": _display_path(path),
            "status": "missing",
            "summary": "No spectator UI/audio validation manifest has been recorded.",
        }
    results = manifest.get("results") or []
    by_name = {str(item.get("name")): item for item in results}
    audio_plans = (
        by_name.get("audio_confidence_mapping", {})
        .get("metrics", {})
        .get("plans", [])
    )
    web_payload = by_name.get("web_replay_payload_contract", {}).get("metrics", {})
    replay_links = by_name.get("dashboard_replay_deep_links", {}).get("metrics", {})
    pygame = by_name.get("pygame_headless_smoke", {}).get("metrics", {})
    passed = bool(manifest.get("passed", False))
    return {
        "exists": True,
        "path": _display_path(path),
        "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "checks_passed": int(manifest.get("checks_passed", 0)),
        "checks": int(manifest.get("checks", len(results))),
        "score": float(manifest.get("score", 0.0)),
        "replay": manifest.get("replay"),
        "audio_buckets": [str(plan.get("bucket")) for plan in audio_plans],
        "web_payload_frames": int(web_payload.get("frame_count", 0) or 0),
        "web_payload_returned_frames": int(web_payload.get("returned_frames", 0) or 0),
        "valid_replays": int(replay_links.get("valid_replays", 0) or 0),
        "viewer_links": int(replay_links.get("with_viewer_links", 0) or 0),
        "pygame_duration": float(pygame.get("duration", 0.0) or 0.0),
        "failed_checks": [str(item.get("name")) for item in results if not item.get("passed")],
    }


def _operational_validation_manifest_summary(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest is None:
        return {
            "exists": False,
            "path": _display_path(path),
            "status": "missing",
            "summary": "No whole-project operational readiness manifest has been recorded.",
        }
    results = manifest.get("results") or []
    failed = [str(item.get("name")) for item in results if not item.get("passed")]
    passed = bool(manifest.get("passed", False))
    return {
        "exists": True,
        "path": _display_path(path),
        "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "summary": manifest.get("summary", "Operational readiness evidence recorded."),
        "checks_passed": int(manifest.get("checks_passed", 0)),
        "checks": int(manifest.get("checks", len(results))),
        "score": float(manifest.get("score", 0.0)),
        "failed_checks": failed,
    }


def _evidence_bundle_summary(path: Path) -> dict[str, Any]:
    exists = path.exists()
    manifest = _read_json(path)
    if manifest is None:
        return {
            "exists": exists,
            "path": _display_path(path),
            "status": "invalid" if exists else "missing",
            "passed": False,
            "summary": "Evidence bundle has not been built." if not exists else "Evidence bundle JSON is invalid.",
        }
    summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
    passed = bool(manifest.get("passed", False)) and manifest.get("kind") == "crypt_heist_evidence_bundle"
    return {
        "exists": True,
        "path": _display_path(path),
        "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "kind": manifest.get("kind"),
        "created_at": manifest.get("created_at"),
        "operational_passed": bool(summary.get("operational_passed")),
        "operational_checks_passed": int(summary.get("operational_checks_passed", 0) or 0),
        "operational_checks": int(summary.get("operational_checks", 0) or 0),
        "active_acceptance_passed": bool(summary.get("active_acceptance_passed")),
        "active_required_checks_passed": int(summary.get("active_required_checks_passed", 0) or 0),
        "active_required_checks": int(summary.get("active_required_checks", 0) or 0),
        "active_milestone_targets_passed": int(summary.get("active_milestone_targets_passed", 0) or 0),
        "active_milestone_targets": int(summary.get("active_milestone_targets", 0) or 0),
        "active_replay_reports": int(summary.get("active_replay_reports", 0) or 0),
        "active_replay_reports_nominal": int(summary.get("active_replay_reports_nominal", 0) or 0),
        "checkpoints_existing": int(summary.get("checkpoints_existing", 0) or 0),
        "checkpoints_expected": int(summary.get("checkpoints_expected", 0) or 0),
    }


def _evidence_bundle_verification_summary(path: Path) -> dict[str, Any]:
    exists = path.exists()
    manifest = _read_json(path)
    if manifest is None:
        return {
            "exists": exists,
            "path": _display_path(path),
            "status": "invalid" if exists else "missing",
            "passed": False,
            "summary": "Evidence bundle verification has not been recorded." if not exists else "Evidence verification JSON is invalid.",
        }
    failures = manifest.get("failures") if isinstance(manifest.get("failures"), list) else []
    passed = bool(manifest.get("passed", False)) and manifest.get("kind") == "crypt_heist_evidence_bundle_verification"
    return {
        "exists": True,
        "path": _display_path(path),
        "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "kind": manifest.get("kind"),
        "bundle": manifest.get("bundle"),
        "bundle_created_at": manifest.get("bundle_created_at"),
        "checks_passed": int(manifest.get("checks_passed", 0) or 0),
        "checks": int(manifest.get("checks", 0) or 0),
        "failure_count": len(failures),
        "failures": failures[:5],
    }


def _self_play_manifest_summary(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest is None:
        return {
            "exists": False,
            "path": _display_path(path),
            "status": "missing",
            "summary": "No scenario-gated self-play manifest has been recorded.",
        }
    promotion = manifest.get("control_promotion_decision") or {}
    control_league = manifest.get("control_league") or {}
    scores = control_league.get("scores") or {}
    scenario = control_league.get("scenario_acceptance") or {}
    diagnostics = scenario.get("diagnostics") or {}
    repair_plan = plan_curriculum_from_manifest(scenario, source=path) if isinstance(scenario, dict) and scenario else None
    repair_action = _next_curriculum_action(repair_plan)
    scenario_present = isinstance(scenario, dict) and bool(scenario)
    scenario_passed = bool(scenario.get("passed")) if scenario_present else None
    promoted = bool(promotion.get("promoted"))
    if promoted:
        status = "promoted"
    elif scenario_present and not scenario_passed:
        status = "failed"
    elif manifest.get("control_promotion_decision"):
        status = "not_promoted"
    else:
        status = "recorded"
    reason = str(promotion.get("reason", "unknown"))
    top = diagnostics.get("top") or []
    summary = "Latest scenario-gated self-play candidate promoted."
    if status == "failed":
        summary = f"Latest scenario-gated self-play candidate failed: {reason}."
    elif status == "not_promoted":
        summary = f"Latest scenario-gated self-play candidate did not promote: {reason}."
    return {
        "exists": True,
        "path": _display_path(path),
        "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "status": status,
        "passed": status != "failed",
        "summary": summary,
        "name": manifest.get("name"),
        "seed": manifest.get("seed"),
        "promoted": promoted,
        "evader_promoted": bool(promotion.get("evader_promoted")),
        "pursuer_team_promoted": bool(promotion.get("pursuer_team_promoted")),
        "reason": reason,
        "candidate_score": float(promotion.get("candidate_score", scores.get("overall_score", 0.0)) or 0.0),
        "incumbent_score": promotion.get("incumbent_score"),
        "scenario_present": scenario_present,
        "scenario_passed": scenario_passed,
        "scenario_acceptance_score": float(scores.get("scenario_acceptance_score", 0.0) or 0.0),
        "scenario_required_score": float((scenario.get("score") or {}).get("required_score", 0.0)) if scenario_present else 0.0,
        "scenario_milestone_score": float((scenario.get("score") or {}).get("milestone_score", 0.0)) if scenario_present else 0.0,
        "scenario_required_checks_passed": int(
            scenario.get("required_checks_passed", (scenario.get("score") or {}).get("required_checks_passed", 0))
        ) if scenario_present else 0,
        "scenario_required_checks": int(
            scenario.get("required_checks", (scenario.get("score") or {}).get("required_checks", 0))
        ) if scenario_present else 0,
        "diagnostic_count": int(diagnostics.get("count", 0)),
        "top_diagnostics": top[:5],
        "repair_action": repair_action,
    }


def _readiness_blockers(
    acceptance: dict[str, Any],
    strict_acceptance: dict[str, Any],
    plan: dict[str, Any] | None,
    missing_checkpoints: list[dict[str, Any]],
    learned_gates: dict[str, dict[str, Any]],
    information_gates: dict[str, dict[str, Any]],
    replay_fidelity: dict[str, Any],
    physics_validation: dict[str, Any],
    env_validation: dict[str, Any],
    spectator_validation: dict[str, Any],
    operational_validation: dict[str, Any],
    evidence_bundle: dict[str, Any],
    evidence_bundle_verification: dict[str, Any],
    self_play: dict[str, Any],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if not acceptance.get("exists"):
        blockers.append({
            "severity": "high",
            "area": "acceptance",
            "message": "Run the acceptance scenario gate to establish current replay-backed evidence.",
            "command_id": "eval_acceptance_scenarios",
        })
    elif not acceptance.get("passed"):
        blockers.append({
            "severity": "high",
            "area": "acceptance",
            "message": "Required acceptance checks are failing.",
            "command_id": "eval_acceptance_scenarios",
        })
    elif acceptance.get("milestone_targets") and acceptance.get("milestone_targets_passed") < acceptance.get("milestone_targets"):
        blockers.append({
            "severity": "medium",
            "area": "milestone",
            "message": "The operational smoke gate passes, but original full-plan milestones are not complete.",
            "command_id": "eval_acceptance_full_targets",
        })
    if plan and int(plan.get("diagnostic_count", 0)) > 0:
        top = (plan.get("actions") or [{}])[0]
        blockers.append({
            "severity": "medium",
            "area": "curriculum",
            "message": plan.get("summary", "Curriculum diagnostics need repair."),
            "command_id": top.get("command_id"),
            "lane": top.get("lane"),
        })
    if missing_checkpoints:
        blockers.append({
            "severity": "medium",
            "area": "checkpoints",
            "message": f"{len(missing_checkpoints)} expected learned-policy checkpoint(s) are missing.",
            "command_id": missing_checkpoints[0].get("command_id"),
        })
    if strict_acceptance.get("exists") and not strict_acceptance.get("passed"):
        blockers.append({
            "severity": "medium",
            "area": "strict_acceptance",
            "message": "The stricter full-target acceptance gate has been recorded and is failing.",
            "command_id": "eval_acceptance_full_targets",
        })
    for gate_id, gate in learned_gates.items():
        if not gate.get("exists") or not gate.get("passed"):
            blockers.append({
                "severity": "medium",
                "area": "learned_control",
                "message": f"{gate_id.replace('_', ' ')} is not green.",
                "command_id": "eval_evader_checkpoint_acceptance" if gate_id == "evader_checkpoint_acceptance" else "eval_acceptance_scenarios",
            })
            break
    for gate_id, gate in information_gates.items():
        if not gate.get("exists") or not gate.get("passed"):
            blockers.append({
                "severity": "medium",
                "area": "learned_information",
                "message": f"{gate_id.replace('_', ' ')} is not green.",
                "command_id": "eval_checkpoint_league",
            })
            break
    if not replay_fidelity.get("exists") or not replay_fidelity.get("passed"):
        blockers.append({
            "severity": "medium",
            "area": "replay_fidelity",
            "message": "Replay checksum/determinism evidence is not green.",
            "command_id": "verify_replay_fidelity",
        })
    if not physics_validation.get("exists") or not physics_validation.get("passed"):
        blockers.append({
            "severity": "medium",
            "area": "physics_validation",
            "message": "Vehicle physics acceptance evidence is not green.",
            "command_id": "verify_physics",
        })
    if not env_validation.get("exists") or not env_validation.get("passed"):
        blockers.append({
            "severity": "medium",
            "area": "env_validation",
            "message": "MARL environment contract evidence is not green.",
            "command_id": "verify_env",
        })
    if not spectator_validation.get("exists") or not spectator_validation.get("passed"):
        blockers.append({
            "severity": "medium",
            "area": "spectator_validation",
            "message": "Spectator UI/audio acceptance evidence is not green.",
            "command_id": "verify_spectator",
        })
    if not operational_validation.get("exists") or not operational_validation.get("passed"):
        blockers.append({
            "severity": "medium",
            "area": "operational_validation",
            "message": "Whole-project operational readiness evidence is not green.",
            "command_id": "verify_operational_readiness",
        })
    if not evidence_bundle.get("exists") or not evidence_bundle.get("passed"):
        blockers.append({
            "severity": "medium",
            "area": "evidence_bundle",
            "message": "Research evidence bundle is missing, stale, or failing.",
            "command_id": "build_evidence_bundle",
        })
    elif not evidence_bundle_verification.get("exists") or not evidence_bundle_verification.get("passed"):
        blockers.append({
            "severity": "medium",
            "area": "evidence_bundle",
            "message": "Research evidence bundle has not been verified against current files.",
            "command_id": "verify_evidence_bundle",
        })
    if self_play.get("exists") and self_play.get("status") == "failed":
        repair = self_play.get("repair_action") or {}
        blockers.append({
            "severity": "medium",
            "area": "self_play",
            "message": self_play.get("summary", "Latest scenario-gated self-play candidate failed."),
            "command_id": repair.get("command_id", "run_self_play_cycle_scenario_gated"),
            "lane": repair.get("lane", "scenario_gated_self_play"),
        })
    return blockers


def _readiness_status(
    acceptance: dict[str, Any],
    strict_acceptance: dict[str, Any],
    plan: dict[str, Any] | None,
    missing_checkpoints: list[dict[str, Any]],
    learned_gates: dict[str, dict[str, Any]],
    information_gates: dict[str, dict[str, Any]],
    replay_fidelity: dict[str, Any],
    physics_validation: dict[str, Any],
    env_validation: dict[str, Any],
    spectator_validation: dict[str, Any],
    operational_validation: dict[str, Any],
    evidence_bundle: dict[str, Any],
    evidence_bundle_verification: dict[str, Any],
    self_play: dict[str, Any],
) -> str:
    if not acceptance.get("exists"):
        return "needs_acceptance"
    if not acceptance.get("passed"):
        return "core_gate_failing"
    if plan and int(plan.get("diagnostic_count", 0)) > 0:
        return "needs_training"
    if strict_acceptance.get("exists") and not strict_acceptance.get("passed"):
        return "strict_gate_failing"
    if missing_checkpoints:
        return "needs_checkpoints"
    if any(not gate.get("exists") or not gate.get("passed") for gate in learned_gates.values()):
        return "learned_gate_failing"
    if any(not gate.get("exists") or not gate.get("passed") for gate in information_gates.values()):
        return "information_gate_failing"
    if not replay_fidelity.get("exists") or not replay_fidelity.get("passed"):
        return "fidelity_gate_failing"
    if not physics_validation.get("exists") or not physics_validation.get("passed"):
        return "physics_gate_failing"
    if not env_validation.get("exists") or not env_validation.get("passed"):
        return "env_gate_failing"
    if not spectator_validation.get("exists") or not spectator_validation.get("passed"):
        return "spectator_gate_failing"
    if not operational_validation.get("exists") or not operational_validation.get("passed"):
        return "operational_gate_failing"
    if self_play.get("exists") and self_play.get("status") == "failed":
        return "self_play_needs_repair"
    if not evidence_bundle.get("exists") or not evidence_bundle.get("passed"):
        return "evidence_bundle_failing"
    if not evidence_bundle_verification.get("exists") or not evidence_bundle_verification.get("passed"):
        return "evidence_verification_failing"
    if acceptance.get("milestone_targets") and acceptance.get("milestone_targets_passed") < acceptance.get("milestone_targets"):
        return "milestone_incomplete"
    return "operational"


def _readiness_summary_text(
    status: str,
    acceptance: dict[str, Any],
    strict_acceptance: dict[str, Any],
    plan: dict[str, Any] | None,
    missing_checkpoints: list[dict[str, Any]],
    learned_gates: dict[str, dict[str, Any]],
    information_gates: dict[str, dict[str, Any]],
    replay_fidelity: dict[str, Any],
    physics_validation: dict[str, Any],
    env_validation: dict[str, Any],
    spectator_validation: dict[str, Any],
    operational_validation: dict[str, Any],
    evidence_bundle: dict[str, Any],
    evidence_bundle_verification: dict[str, Any],
    self_play: dict[str, Any],
) -> str:
    if status == "needs_acceptance":
        return "No acceptance evidence yet. Run Acceptance Scenarios first."
    if status == "core_gate_failing":
        return "Required acceptance checks are failing; repair the top curriculum lane."
    if status == "needs_training" and plan:
        return str(plan.get("summary", "Curriculum diagnostics need training."))
    if status == "strict_gate_failing":
        return "The stricter full-target acceptance gate is failing."
    if status == "needs_checkpoints":
        return f"Smoke gate passes, but {len(missing_checkpoints)} learned checkpoint(s) are missing."
    if status == "learned_gate_failing":
        failed = [name for name, gate in learned_gates.items() if not gate.get("exists") or not gate.get("passed")]
        return f"Learned control gate is not green: {failed[0] if failed else 'unknown'}."
    if status == "information_gate_failing":
        failed = [name for name, gate in information_gates.items() if not gate.get("exists") or not gate.get("passed")]
        return f"Learned information gate is not green: {failed[0] if failed else 'unknown'}."
    if status == "fidelity_gate_failing":
        return "Replay checksum/determinism evidence is missing or failing."
    if status == "physics_gate_failing":
        return "Vehicle physics acceptance evidence is missing or failing."
    if status == "env_gate_failing":
        return "MARL environment contract evidence is missing or failing."
    if status == "spectator_gate_failing":
        return "Spectator UI/audio acceptance evidence is missing or failing."
    if status == "operational_gate_failing":
        return str(operational_validation.get("summary", "Whole-project operational readiness evidence is missing or failing."))
    if status == "self_play_needs_repair":
        return str(self_play.get("summary", "Latest scenario-gated self-play candidate needs repair."))
    if status == "evidence_bundle_failing":
        return str(evidence_bundle.get("summary", "Research evidence bundle is missing or failing."))
    if status == "evidence_verification_failing":
        return str(evidence_bundle_verification.get("summary", "Evidence bundle verification is missing or failing."))
    if status == "milestone_incomplete":
        return "Smoke gate passes, but original full-plan milestone targets remain incomplete."
    if strict_acceptance.get("exists"):
        return "Operational gates and full-target gates are passing."
    if acceptance.get("exists"):
        return "Operational smoke gate passes; run full-target acceptance when ready."
    return "Readiness unknown."


def _recommended_command(
    acceptance: dict[str, Any],
    plan: dict[str, Any] | None,
    missing_checkpoints: list[dict[str, Any]],
    learned_gates: dict[str, dict[str, Any]],
    information_gates: dict[str, dict[str, Any]],
    replay_fidelity: dict[str, Any],
    physics_validation: dict[str, Any],
    env_validation: dict[str, Any],
    spectator_validation: dict[str, Any],
    operational_validation: dict[str, Any],
    evidence_bundle: dict[str, Any],
    evidence_bundle_verification: dict[str, Any],
    self_play: dict[str, Any],
) -> dict[str, Any] | None:
    if not acceptance.get("exists") or not acceptance.get("passed"):
        return {"command_id": "eval_acceptance_scenarios", "source": "acceptance"}
    action = _next_curriculum_action(plan)
    if plan and int(plan.get("diagnostic_count", 0)) > 0 and action:
        return {
            "command_id": action.get("command_id"),
            "default_args": action.get("default_args"),
            "lane": action.get("lane"),
            "source": "curriculum",
        }
    if self_play.get("exists") and self_play.get("status") == "failed":
        repair = self_play.get("repair_action") or {}
        return {
            "command_id": repair.get("command_id", "run_self_play_cycle_scenario_gated"),
            "default_args": repair.get("default_args"),
            "lane": repair.get("lane", "scenario_gated_self_play"),
            "source": "self_play",
        }
    if missing_checkpoints:
        checkpoint = missing_checkpoints[0]
        return {
            "command_id": checkpoint.get("command_id"),
            "lane": checkpoint.get("lane"),
            "source": "checkpoint",
        }
    if any(not gate.get("exists") or not gate.get("passed") for gate in learned_gates.values()):
        return {"command_id": "eval_acceptance_scenarios", "source": "learned_control"}
    if any(not gate.get("exists") or not gate.get("passed") for gate in information_gates.values()):
        return {"command_id": "eval_checkpoint_league", "source": "learned_information"}
    if not replay_fidelity.get("exists") or not replay_fidelity.get("passed"):
        return {"command_id": "verify_replay_fidelity", "source": "replay_fidelity"}
    if not physics_validation.get("exists") or not physics_validation.get("passed"):
        return {"command_id": "verify_physics", "source": "physics_validation"}
    if not env_validation.get("exists") or not env_validation.get("passed"):
        return {"command_id": "verify_env", "source": "env_validation"}
    if not spectator_validation.get("exists") or not spectator_validation.get("passed"):
        return {"command_id": "verify_spectator", "source": "spectator_validation"}
    if not operational_validation.get("exists") or not operational_validation.get("passed"):
        return {"command_id": "verify_operational_readiness", "source": "operational_validation"}
    if not evidence_bundle.get("exists") or not evidence_bundle.get("passed"):
        return {"command_id": "build_evidence_bundle", "source": "evidence_bundle"}
    if not evidence_bundle_verification.get("exists") or not evidence_bundle_verification.get("passed"):
        return {"command_id": "verify_evidence_bundle", "source": "evidence_bundle"}
    if action:
        return {
            "command_id": action.get("command_id"),
            "default_args": action.get("default_args"),
            "lane": action.get("lane"),
            "source": "curriculum",
        }
    return {"command_id": "eval_acceptance_full_targets", "source": "milestone"}


def _next_curriculum_action(plan: dict[str, Any] | None) -> dict[str, Any] | None:
    actions = (plan or {}).get("actions") or []
    return actions[0] if actions else None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _replay_items(validate: bool = True):
    REPLAY_DIR.mkdir(exist_ok=True)
    out = []
    candidates = []
    for path in REPLAY_DIR.rglob("*.jsonl"):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        candidates.append((path, stat))
    for path, stat in sorted(candidates, key=lambda item: item[1].st_mtime, reverse=True):
        replay_name = str(path.relative_to(REPLAY_DIR))
        item = {
            "name": replay_name,
            "path": str(path.relative_to(ROOT)),
            "viewer_url": f"/viewer3d/index.html?replay={quote(str(path.relative_to(ROOT)), safe='')}",
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "valid": not validate,
        }
        if not validate:
            out.append(item)
            continue
        try:
            lines = load_replay(path)
            validate_replay(lines)
            item.update(summarize_replay(lines, include_fingerprint=False))
            item["valid"] = True
        except Exception as exc:
            item["error"] = str(exc)
        out.append(item)
    return out


def _weighted_replay_average(items: list[dict[str, Any]], key: str) -> float:
    weighted = 0.0
    frames = 0
    for item in items:
        frame_count = int(item.get("frames", 0) or 0)
        weighted += float(item.get(key, 0.0) or 0.0) * frame_count
        frames += frame_count
    return float(weighted / max(1, frames))


def _replay_report_path(path: Path) -> Path:
    try:
        rel = path.relative_to(REPLAY_DIR)
    except ValueError:
        rel = Path(path.name)
    safe = "_".join(rel.with_suffix("").parts)
    return LOG_DIR / "replay_reports" / f"{safe}_report.json"


@app.get("/api/configs")
def api_configs():
    configs = []
    for path in sorted(CONFIG_DIR.glob("*.json")):
        configs.append({"name": path.name, "path": str(path.relative_to(ROOT)), "size": path.stat().st_size})
    return jsonify(configs)


@app.get("/api/configs/<name>")
def api_config(name: str):
    path = _config_path(name)
    return jsonify({"name": path.name, "text": path.read_text(encoding="utf-8")})


@app.post("/api/configs/<name>")
def api_save_config(name: str):
    path = _config_path(name)
    payload = request.get_json(force=True) or {}
    text = payload.get("text", "")
    json.loads(text)
    backup = path.with_suffix(path.suffix + ".bak")
    if path.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return jsonify({"ok": True, "backup": str(backup.relative_to(ROOT))})


def _config_path(name: str) -> Path:
    path = (CONFIG_DIR / name).resolve()
    if CONFIG_DIR.resolve() not in path.parents or path.suffix != ".json":
        raise ValueError("invalid config path")
    if not path.exists():
        raise FileNotFoundError(name)
    return path


def _replay_path(name: str) -> Path:
    raw = Path(name)
    if raw.parts and raw.parts[0] == REPLAY_DIR.name:
        path = (ROOT / raw).resolve()
    else:
        path = (REPLAY_DIR / raw).resolve()
    try:
        path.relative_to(REPLAY_DIR.resolve())
    except ValueError as exc:
        raise ValueError("invalid replay path") from exc
    if path.suffix != ".jsonl":
        raise ValueError("invalid replay path")
    if not path.exists():
        raise FileNotFoundError(name)
    return path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


@app.get("/viewer3d/<path:filename>")
def viewer3d(filename: str):
    return send_from_directory(ROOT / "viewer3d" / "static", filename)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the Cryptographic Heist Command Center.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    REPLAY_DIR.mkdir(exist_ok=True)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
