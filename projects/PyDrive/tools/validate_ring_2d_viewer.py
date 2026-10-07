"""Headless smoke test for watching a Ring PPO checkpoint in the 2D viewer."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Run standalone without PYTHONPATH: when invoked as `python3 tools/<this>.py`
# sys.path[0] is tools/, not the repo root, so `from run import ...` below would
# fail. Put the repo root on the path explicitly (os.chdir alone doesn't do it).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def make_checkpoint(path: Path):
    from run import _apply_ppo_args, _long_track_race_profile, _train_track
    from supra.config import PPOSpec
    from supra.ppo import PPO
    from supra.track import configure_hills
    from types import SimpleNamespace

    configure_hills(enabled=False, force_flat=False)
    args = SimpleNamespace(
        track="nordschleife", seed=7, pop=1, anneal=True, workers=1,
        lr=None, patience=850, max_restarts=0,
    )
    fixed, label = _train_track(args)
    cfg = _apply_ppo_args(PPOSpec(), args)
    cfg.rollout = 4
    reward = _long_track_race_profile(cfg, fixed, label)
    ppo = PPO(mode="race", car="mazda787b", ppo=cfg, reward=reward,
              fixed_track=fixed, track_name=label)
    ppo.save(str(path))


def main() -> int:
    os.chdir(ROOT)
    ckpt = Path("/tmp/supra_ring_2d_viewer_smoke.pt")
    try:
        ckpt.unlink()
    except FileNotFoundError:
        pass
    make_checkpoint(ckpt)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["SDL_VIDEODRIVER"] = "dummy"
    env["SDL_AUDIODRIVER"] = "dummy"
    cmd = [
        sys.executable, "run.py",
        "--watch-ppo",
        "--track", "nordschleife",
        "--checkpoint", str(ckpt),
        "--car", "mazda787b",
        "--no-audio",
        "--smoke-frames", "3",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True,
                          capture_output=True, timeout=90)
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return fail(f"2D viewer smoke exited {proc.returncode}\n{out[-4000:]}")
    if "Watching PPO race policy" not in out:
        return fail(f"watcher did not load PPO policy as expected\n{out[-2000:]}")
    if "nordschleife" not in out.lower():
        return fail(f"watcher output did not reference Nordschleife\n{out[-2000:]}")
    print("Ring 2D viewer validation OK")
    print("  loaded checkpoint and rendered/stepped 3 headless frames on nordschleife")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
