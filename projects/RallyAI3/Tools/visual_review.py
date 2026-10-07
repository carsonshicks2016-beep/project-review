"""Run a bounded frozen-policy visual review without reusing an experiment's outputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import fcntl
from contextlib import ExitStack

ROOT = Path(__file__).resolve().parents[1]
COURSE = "f187eaf562a3681aba54681939a54345c84d484556cef1b97baa2048acb5f8f4"
CHECKPOINT = "02d659fb4993a27faeb14292e18eff37c39790038be25714d8d296eda6011b19"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--height", type=int, choices=(720, 1080), default=720)
    parser.add_argument("--legacy-forest", action="store_true")
    parser.add_argument("--course", default=COURSE)
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--no-screenshots", action="store_true")
    parser.add_argument("--audio", action="store_true")
    parser.add_argument("--audio-lifecycle", action="store_true")
    parser.add_argument("--camera", choices=("ThirdPerson", "Trackside", "Helicopter", "Hood", "Driver", "Cinematic"), default="ThirdPerson")
    args = parser.parse_args()
    if not args.label.replace("-", "").replace("_", "").isalnum():
        parser.error("label must contain letters, numbers, hyphens or underscores")
    from rallylab import core
    if any(j["state"] in core.ACTIVE for j in core.jobs()):
        parser.error("another managed job is active; review will not compete with it")
    if core.build_status()["status"] != "current":
        parser.error("build is not current")
    output = ROOT / ".rally/visual-review/v3" / f"{args.label}-{args.height}"
    output.mkdir(parents=True, exist_ok=False)
    course = json.loads((ROOT / ".rally/courses" / args.course / "manifest.json").read_text())
    spec = {"schema": 1, "mode": "specialist", "viewer": True, "evaluation": False,
            "courseBundle": str(ROOT / ".rally/courses" / args.course / "course"), "courseId": args.course,
            "courseName": course.get("name", args.course), "courseFamily": course.get("family", ""),
            "checkpointId": CHECKPOINT,
            "policyBundle": str(ROOT / ".rally/checkpoints" / CHECKPOINT / "bundle/policy"),
            "runId": f"visual-review-{args.label}", "output": str(output / "attempts"),
            "attempts": 1, "seed": 2026, "deterministic": True, "timeScale": 1,
            "startingGear": "neutral"}
    launch = output / "launch.json"
    launch.write_text(json.dumps(spec, indent=2))
    (output / "build.json").write_text(json.dumps(core.build_status(), indent=2))
    env = dict(os.environ, RALLY_LAB_LAUNCH=str(launch), RALLY_PRESENTATION_REVIEW=str(output),
               RALLY_FOREST_V3="0" if args.legacy_forest else "1", RALLY_REVIEW_CAMERA=args.camera,
               RALLY_REVIEW_SEQUENCE="1" if args.sequence else "0",
               RALLY_REVIEW_NO_SCREENSHOTS="1" if args.no_screenshots else "0",
               RALLY_CAMERA_TRACE=str(output / "camera-frames.jsonl"))
    if args.audio:
        env["RALLY_AUDIO_CAPTURE"] = str(output)
        env["RALLY_AUDIO_TRACE"] = str(output / "audio-frames.jsonl")
    if args.audio_lifecycle:
        env["RALLY_AUDIO_LIFECYCLE"] = str(output)
    width = 1280 if args.height == 720 else 1920
    with ExitStack() as stack:
        for name in ("compute.lock", "viewer.lock"):
            lock = stack.enter_context((ROOT / ".rally" / name).open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        subprocess.run([str(ROOT / ".rally/player/RallyTraining.app/Contents/MacOS/RallyAI3"),
                        "-screen-width", str(width), "-screen-height", str(args.height), "-screen-fullscreen", "0",
                        "-logFile", str(output / "player.log")], env=env, check=True, timeout=90)
    metrics = json.loads((output / "performance.json").read_text())
    if metrics["course"] != args.course or metrics["checkpoint"] != CHECKPOINT or metrics["frames"] < 100:
        raise RuntimeError("review identity or sample count failed validation")
    print(json.dumps({key: value for key, value in metrics.items()
                      if key not in ("samples", "frameTimesMs", "availableCounters")}, indent=2))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
