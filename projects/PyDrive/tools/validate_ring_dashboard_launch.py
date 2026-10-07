"""Validate repeated dashboard-launched Nordschleife PPO training.

This uses the Command Center Flask route and process manager directly, not just
`build_command()`, so it proves the dashboard backend can repeatedly launch Ring
PPO runs, stream parseable metrics, and save compatible checkpoints.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import time

import torch

ROOT = Path(__file__).resolve().parents[1]


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def load_server():
    spec = importlib.util.spec_from_file_location("cc_server", ROOT / "command-center" / "server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wait_proc(server, pid: int, timeout_s: float = 90.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        info = server.PROCS.get(pid)
        if info and not info.get("running"):
            return info
        time.sleep(0.2)
    raise TimeoutError(f"dashboard task {pid} did not finish in {timeout_s}s")


def run_one(server, client, stem: str):
    ckpt = ROOT / f"{stem}.pt"
    best = ROOT / f"{stem}_best.pt"
    for p in (ckpt, best):
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    payload = {
        "action": "ring_ppo_train",
        "params": {
            "iters": 5,
            "out": ckpt.name,
            "workers": 1,
            "pop": 1,
            "patience": 850,
            "max_restarts": 0,
            "anneal": True,
        },
    }
    resp = client.post("/api/launch", json=payload)
    if resp.status_code != 200:
        raise AssertionError(f"launch failed {resp.status_code}: {resp.get_data(as_text=True)}")
    data = resp.get_json()
    cmd = data["command"]
    if "--track nordschleife" not in cmd or "--flat" in cmd:
        raise AssertionError(f"bad Ring train command: {cmd}")
    if "--pop 1" not in cmd or "--workers" in cmd:
        # workers=1 is intentionally omitted by command builder; pop=1 must remain.
        raise AssertionError(f"test launch did not use lightweight env count: {cmd}")

    info = wait_proc(server, int(data["pid"]))
    proc = info["proc"]
    logs = "".join(info["logs"])
    if proc.returncode != 0:
        raise AssertionError(f"training exited {proc.returncode}\n{logs[-4000:]}")
    if "[track-profile] nordschleife" not in logs:
        raise AssertionError("Ring long-track profile did not print in training logs")
    if not info["metrics"] or info["metrics"][-1].get("x") != 5:
        raise AssertionError(f"dashboard did not parse PPO metric at iter 5: {info['metrics']}")
    if not ckpt.exists():
        raise AssertionError(f"checkpoint was not saved: {ckpt}")

    meta = torch.load(ckpt, map_location="cpu", weights_only=False)
    checks = {
        "track": meta.get("track") == "nordschleife",
        "track_profile": meta.get("track_profile") == "nordschleife-full-20.832km",
        "car": meta.get("car") == server.RING_CAR,
        "obs_dim": int(meta.get("obs_dim", 0)) == 60,
        "lookahead": tuple(meta.get("sensor_lookahead_distances", ())) == (
            15.0, 30.0, 55.0, 85.0, 125.0, 180.0),
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise AssertionError(f"bad checkpoint metadata {bad}: {meta}")
    return ckpt


def main() -> int:
    os.chdir(ROOT)
    server = load_server()
    client = server.app.test_client()
    stems = ["_tmp_ring_dashboard_smoke_a", "_tmp_ring_dashboard_smoke_b"]
    try:
        saved = [run_one(server, client, stem) for stem in stems]
    except Exception as e:
        return fail(str(e))
    finally:
        for stem in stems:
            for suffix in (".pt", "_best.pt"):
                try:
                    (ROOT / f"{stem}{suffix}").unlink()
                except FileNotFoundError:
                    pass
    print("Ring dashboard launch validation OK")
    print(f"  repeated runs: {', '.join(p.name for p in saved)}")
    print("  both emitted chartable PPO metrics and saved Ring-compatible checkpoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
