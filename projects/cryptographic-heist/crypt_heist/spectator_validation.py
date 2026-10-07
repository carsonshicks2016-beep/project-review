"""Spectator UI and audio acceptance checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from command_center.server import app

from .replay import load_replay, summarize_replay, validate_replay
from .sound import soundtrack_plan
from .viewer_visual_validation import run_viewer_visual_validation


ROOT = Path(__file__).resolve().parents[1]


def run_spectator_validation(
    *,
    replay: str | Path = "replays/control_acceptance_active/downtown_chase_seed11.jsonl",
    pygame_frames: int = 5,
    out: str | Path | None = None,
    include_visual: bool = True,
) -> dict[str, Any]:
    checks = [
        _check_audio_mapping(),
        _check_web_static_contract(),
        _check_threejs_static_contract(),
        _check_web_replay_payload(replay),
        _check_dashboard_replay_links(),
        _check_pygame_headless(pygame_frames),
    ]
    if include_visual:
        checks.append(_check_viewer_visual(replay))
    passed = all(check["passed"] for check in checks)
    manifest = {
        "version": 1,
        "passed": passed,
        "checks_passed": sum(int(check["passed"]) for check in checks),
        "checks": len(checks),
        "score": sum(int(check["passed"]) for check in checks) / max(1, len(checks)),
        "replay": str(replay),
        "results": checks,
    }
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _check_audio_mapping() -> dict[str, Any]:
    probes = [
        (0.10, "low", "dissonant jazz"),
        (0.50, "mid", "hybrid tension"),
        (0.90, "high", "structured classical"),
    ]
    plans = [soundtrack_plan(confidence) for confidence, _, _ in probes]
    passed = all(plan["bucket"] == bucket and label in plan["label"] for plan, (_, bucket, label) in zip(plans, probes))
    return _check(
        "audio_confidence_mapping",
        passed,
        metrics={"plans": plans},
        thresholds={"low_lt": 0.3, "high_gt": 0.7, "required_buckets": ["low", "mid", "high"]},
    )


def _check_web_static_contract() -> dict[str, Any]:
    index = ROOT / "viewer3d" / "static" / "index.html"
    script = ROOT / "viewer3d" / "static" / "app.js"
    style = ROOT / "viewer3d" / "static" / "style.css"
    index_text = index.read_text(encoding="utf-8")
    script_text = script.read_text(encoding="utf-8")
    style_text = style.read_text(encoding="utf-8")
    required_index = [
        "scene",
        "replay-select",
        "radio",
        "predictions",
        "confidence-fill",
        "soundtrack-label",
        "audio-toggle",
    ]
    required_script = [
        "/api/replays",
        "/api/replay",
        "drawPredictions",
        "drawContainment",
        "soundtrackMode",
        "updateAudio",
        "radio-line",
        "spoof",
    ]
    required_style = [
        ".dashboard",
        ".radio-line.spoof",
        ".confidence-track",
        ".prediction",
    ]
    missing = (
        [item for item in required_index if item not in index_text]
        + [item for item in required_script if item not in script_text]
        + [item for item in required_style if item not in style_text]
    )
    return _check(
        "web_cockpit_static_contract",
        not missing,
        metrics={
            "missing": missing,
            "index_bytes": index.stat().st_size,
            "script_bytes": script.stat().st_size,
            "style_bytes": style.stat().st_size,
        },
        thresholds={"required_terms": len(required_index) + len(required_script) + len(required_style)},
    )


def _check_threejs_static_contract() -> dict[str, Any]:
    index = ROOT / "viewer3d" / "static" / "index.html"
    script = ROOT / "viewer3d" / "static" / "app.js"
    style = ROOT / "viewer3d" / "static" / "style.css"
    index_text = index.read_text(encoding="utf-8")
    script_text = script.read_text(encoding="utf-8")
    style_text = style.read_text(encoding="utf-8")
    required = {
        "index": [
            'data-fallback-renderer="2d"',
        ],
        "script": [
            "THREE_MODULE_URL",
            "initThreeRenderer",
            "WebGLRenderer",
            "PerspectiveCamera",
            "buildThreeCity",
            "renderThree",
            "updateThreeCars",
            "updateThreeTrails",
            "updateThreeVectors",
            "dataset.renderer = \"threejs\"",
        ],
        "style": [
            ".three-canvas",
            "#scene.fallback-hidden",
        ],
    }
    missing = (
        [item for item in required["index"] if item not in index_text]
        + [item for item in required["script"] if item not in script_text]
        + [item for item in required["style"] if item not in style_text]
    )
    return _check(
        "threejs_replay_renderer_contract",
        not missing,
        metrics={
            "missing": missing,
            "module_url": "cdn.jsdelivr.net/npm/three",
            "fallback_canvas": 'data-fallback-renderer="2d"' in index_text,
            "webgl_renderer": "WebGLRenderer" in script_text,
        },
        thresholds={
            "requires_threejs_webgl_renderer": True,
            "requires_fallback_canvas": True,
        },
    )


def _check_web_replay_payload(replay: str | Path) -> dict[str, Any]:
    replay_path = Path(replay)
    replay_arg = str(replay_path)
    lines = load_replay(replay_path)
    validate_replay(lines)
    summary = summarize_replay(lines, include_fingerprint=False)
    client = app.test_client()
    response = client.get("/api/replay", query_string={"path": replay_arg, "max_frames": 16})
    ok = response.status_code == 200
    payload = response.get_json() if ok else {}
    frames = payload.get("frames") or []
    sample = frames[0] if frames else {}
    passed = bool(
        ok
        and payload.get("returned_frames", 0) <= 16
        and payload.get("frame_count") == summary["frames"]
        and sample.get("agents")
        and sample.get("radio") is not None
        and sample.get("predictions")
        and sample.get("actions")
        and sample.get("rewards")
        and sample.get("reward_components")
        and sample.get("camera")
        and "confidence" in sample
        and "waypoint" in sample
        and "avg_evader_reward" in (payload.get("summary") or {})
        and "avg_pursuer_auth_penalty" in (payload.get("summary") or {})
        and "radio_word_entropy" in (payload.get("summary") or {})
        and "jam_events" in (payload.get("summary") or {})
        and "cipher_rotations" in (payload.get("summary") or {})
    )
    return _check(
        "web_replay_payload_contract",
        passed,
        metrics={
            "status_code": response.status_code,
            "frame_count": payload.get("frame_count"),
            "returned_frames": payload.get("returned_frames"),
            "summary_frames": summary["frames"],
            "has_predictions": bool(sample.get("predictions")),
            "has_radio": sample.get("radio") is not None,
            "has_actions": bool(sample.get("actions")),
            "has_rewards": bool(sample.get("rewards")),
            "has_reward_components": bool(sample.get("reward_components")),
            "has_camera": bool(sample.get("camera")),
            "has_reward_summary": "avg_evader_reward" in (payload.get("summary") or {}),
            "has_comms_summary": "radio_word_entropy" in (payload.get("summary") or {}),
        },
        thresholds={
            "max_returned_frames": 16,
            "requires_predictions": True,
            "requires_radio": True,
            "requires_actions_rewards_and_camera": True,
            "requires_comms_summary": True,
        },
    )


def _check_dashboard_replay_links() -> dict[str, Any]:
    client = app.test_client()
    response = client.get("/api/replays")
    payload = response.get_json() if response.status_code == 200 else []
    valid = [item for item in payload if item.get("valid")]
    with_links = [item for item in valid if str(item.get("viewer_url", "")).startswith("/viewer3d/index.html?replay=")]
    passed = bool(response.status_code == 200 and valid and len(with_links) == len(valid))
    return _check(
        "dashboard_replay_deep_links",
        passed,
        metrics={
            "status_code": response.status_code,
            "valid_replays": len(valid),
            "with_viewer_links": len(with_links),
        },
        thresholds={"requires_all_valid_replays_linked": True},
    )


def _check_viewer_visual(replay: str | Path) -> dict[str, Any]:
    visual = run_viewer_visual_validation(
        replay=replay,
        out=ROOT / "logs" / "viewer_visual_validation.json",
        screenshot_dir=ROOT / "logs" / "viewer_visual",
    )
    by_name = {str(row.get("name")): row for row in visual.get("results", []) if isinstance(row, dict)}
    desktop = by_name.get("viewer_screenshot_desktop", {})
    mobile = by_name.get("viewer_screenshot_mobile", {})
    passed = bool(visual.get("passed") and desktop.get("passed") and mobile.get("passed"))
    return _check(
        "viewer_browser_visual_acceptance",
        passed,
        metrics={
            "visual_passed": bool(visual.get("passed")),
            "checks_passed": visual.get("checks_passed"),
            "checks": visual.get("checks"),
            "screenshot_dir": visual.get("screenshot_dir"),
            "desktop": desktop.get("metrics", {}),
            "mobile": mobile.get("metrics", {}),
        },
        thresholds={
            "requires_desktop_and_mobile_screenshots": True,
            "requires_non_blank_canvas_pixels": True,
        },
    )


def _check_pygame_headless(frames: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["SDL_VIDEODRIVER"] = "dummy"
    env["PYTHONUNBUFFERED"] = "1"
    started = time.time()
    proc = subprocess.run(
        [sys.executable, "run.py", "--mute", "--max-frames", str(frames)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )
    duration = time.time() - started
    passed = proc.returncode == 0 and duration <= 30.0
    return _check(
        "pygame_headless_smoke",
        passed,
        metrics={
            "returncode": proc.returncode,
            "frames": frames,
            "duration": duration,
            "log_tail": proc.stdout[-800:],
        },
        thresholds={"max_duration_seconds": 30.0, "returncode": 0},
    )


def _check(name: str, passed: bool, *, metrics: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "metrics": metrics,
        "thresholds": thresholds,
    }
