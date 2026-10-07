"""Validate the dedicated Nordschleife PPO dashboard/training pipeline.

This is intentionally small but end-to-end-ish: command-center action wiring,
long-track PPO profile, one tiny collect/update cycle, checkpoint metadata, and
watch/diagnose track assumptions.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def assert_has(cmd, *items):
    missing = [x for x in items if x not in cmd]
    if missing:
        raise AssertionError(f"command missing {missing}: {cmd}")


def assert_not_has(cmd, *items):
    bad = [x for x in items if x in cmd]
    if bad:
        raise AssertionError(f"command unexpectedly has {bad}: {cmd}")


def main() -> int:
    os.chdir(ROOT)

    server = load_server()
    train = server.build_command("ring_ppo_train", {"iters": 200, "out": "ring_smoke.pt"})
    assert_has(train, "run.py", "--ppo", "200", "--track", "nordschleife",
               "--out", "ring_smoke.pt", "--anneal", "--workers", "8",
               "--patience", "850", "--max-restarts", "6", "--car", server.RING_CAR)
    assert_not_has(train, "--flat")

    live = server.build_command("ring_ppo_live", {"iters": 200})
    assert_has(live, "--live", "--track", "nordschleife", "--anneal")
    assert_not_has(live, "--flat")

    cont = server.build_command("ring_ppo_continue", {"out": "ring_smoke.pt"})
    assert_has(cont, "--resume", "ring_smoke_best.pt", "--track", "nordschleife")
    assert_not_has(cont, "--flat")

    watch = server.build_command("ring_ppo_watch", {"out": "ring_smoke.pt"})
    assert_has(watch, "--watch-ppo", "--checkpoint", "ring_smoke_best.pt",
               "--track", "nordschleife")
    assert_not_has(watch, "--flat")

    diag = server.build_command("ring_ppo_diagnose", {"out": "ring_smoke.pt", "max_steps": 8})
    assert_has(diag, "tools/diagnose_checkpoint.py", "ring_smoke_best.pt",
               "--track", "nordschleife", "--scenario", "solo")
    assert_not_has(diag, "--flat")

    from run import _apply_ppo_args, _long_track_race_profile, _train_track
    from supra.config import PPOSpec, SensorSpec
    from supra.ppo import PPO
    from supra.sensors import SensorSuite
    from supra.track import configure_hills

    configure_hills(enabled=False, force_flat=False)
    args = SimpleNamespace(
        track="nordschleife", seed=7, pop=1, anneal=True, workers=1,
        lr=None, patience=850, max_restarts=6,
    )
    fixed, label = _train_track(args)
    if label != "nordschleife" or fixed.length < 20_831:
        return fail("failed to resolve Nordschleife training track")

    cfg = _apply_ppo_args(PPOSpec(), args)
    cfg.rollout = 4
    reward = _long_track_race_profile(cfg, fixed, label)
    if reward is None:
        return fail("long-track reward profile was not applied")
    if cfg.episode_seconds < 540 or not cfg.random_start:
        return fail("Ring PPO profile did not enable long episodes and exploring starts")
    if cfg.track_profile != "nordschleife-full-20.832km":
        return fail(f"unexpected track profile: {cfg.track_profile}")

    base_obs = SensorSuite(SensorSpec()).obs_size
    ring_sensor = SensorSpec()
    ring_sensor.lookahead_distances = cfg.sensor_lookahead_distances
    if SensorSuite(ring_sensor).obs_size != base_obs:
        return fail("Ring lookahead profile changed observation dimensionality")

    ppo = PPO(mode="race", car=server.RING_CAR, ppo=cfg, reward=reward,
              fixed_track=fixed, track_name=label)
    obs = ppo.vec.reset()
    obs, batch = ppo.collect(obs)
    adv, ret = ppo.gae(batch[3], batch[4], batch[5], obs)
    stats = ppo.update(batch, adv, ret)
    if not all(np.isfinite(v) for v in stats.values()):
        return fail(f"non-finite PPO update stats: {stats}")

    ckpt = Path("/tmp/supra_ring_pipeline_smoke.pt")
    best = Path("/tmp/supra_ring_pipeline_smoke_best.pt")
    for p in (ckpt, best):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    ppo.save(str(ckpt))
    meta = torch.load(ckpt, map_location="cpu", weights_only=False)
    if meta.get("track") != "nordschleife":
        return fail("saved checkpoint does not record Nordschleife track")
    if meta.get("track_profile") != cfg.track_profile:
        return fail("saved checkpoint does not record Ring track profile")
    if tuple(meta.get("sensor_lookahead_distances", ())) != tuple(cfg.sensor_lookahead_distances):
        return fail("saved checkpoint does not record Ring lookahead distances")

    net, norm, loaded = PPO.load_policy(str(ckpt))
    if loaded["obs_dim"] != ppo.obs_dim or loaded["sdim"] != ppo.sdim:
        return fail("watch loader metadata does not match trained policy dims")
    _ = net, norm

    print("Ring PPO pipeline validation OK")
    print(f"  command car={server.RING_CAR} track=nordschleife out=ring_smoke.pt")
    print(f"  obs_dim={ppo.obs_dim} sdim={ppo.sdim} profile={cfg.track_profile}")
    print(f"  reward_progress={reward.progress:.1f} straight_speed={reward.straight_speed:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
