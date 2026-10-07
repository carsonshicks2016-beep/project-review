"""Fable Five validation gates.

Run after ANY edit to supra/fable5.py, the pace block, or the envelope solver:

    PYTHONPATH=$PWD python3 tools/validate_fable5.py

Gates:
  1. envelope sanity — finite, bounded, real braking zones, believable lap time
  2. obs layout — fable-v1 = hills-v1 58 + pace 8 (+2 mode) and watch parity
  3. env smoke — reset/step/reward finite, clean-lap bookkeeping, terminations
  4. train smoke — 2 tiny PPO iterations end-to-end + checkpoint save/load
  5. transplant — a hills-v1 (60-dim) checkpoint imports into fable-v1 (68-dim)
"""
import os
import math
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import torch

from supra.config import PPOSpec, SensorSpec, get_car
from supra.fable5 import (DRIVETRAIN_VERSION, FABLE_CAR, FableEnv, FableEvaluator, PACE_DISTANCES,
                          RaceBox, SUPERHUMAN_LAP, attach_envelope,
                          compute_speed_envelope, load_or_transplant,
                          stage_defaults, vref_at)
from supra.ppo import PPO, state_dict_hash
from supra.sensors import SensorSuite
from supra.track import named_track


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        raise SystemExit(f"gate failed: {name} {detail}")


def main():
    t0 = time.time()
    print("== 1. speed envelope ==")
    trk = named_track("nordschleife")
    car = get_car(FABLE_CAR)
    env = compute_speed_envelope(trk, car)
    v, lap = env["v"], env["lap_time"]
    gate("finite + positive", bool(np.all(np.isfinite(v)) and np.all(v > 0)))
    gate("bounded by terminal speed", bool(np.max(v) <= env["v_max"] + 1e-6),
         f"vmax {np.max(v):.1f} of {env['v_max']:.1f} m/s")
    gate("slow in the tightest corner", bool(np.min(v) < 22.0),
         f"vmin {np.min(v):.1f} m/s")
    gate("believable theoretical lap", 240.0 < lap < 620.0,
         f"{lap:.1f}s ({lap/60:.0f}:{lap%60:05.2f}) vs superhuman bar {SUPERHUMAN_LAP:.1f}s")
    # braking zones: deceleration into the slowest points must be grip-bounded
    ds = np.diff(np.append(trk.arc, trk.length))
    dv2 = np.diff(np.append(v, v[0]) ** 2)
    decel = -dv2 / (2.0 * np.maximum(ds, 1e-6))
    gate("braking bounded by grip+aero", bool(np.max(decel) < 60.0),
         f"max decel {np.max(decel):.1f} m/s^2")
    # Forward reachability is the regression that matters: the old solver only
    # constrained positive acceleration and left 1,495 impossible samples, with
    # 3 m jumps as large as 60 m/s and a fake ~400 s theoretical lap.
    g, rho, m = 9.81, float(car.air_density), float(car.mass)
    q = 0.5 * rho * float(car.downforce_ClA) / m
    cd = 0.5 * rho * float(car.drag_area) / m
    crr = float(car.rolling_resistance) * g
    peak_power = max(float(tq) * float(rpm) * 2 * math.pi / 60
                     for rpm, tq in car.torque_curve)
    p_kg = peak_power * float(car.drivetrain_efficiency) / m
    k = np.maximum(np.abs(np.asarray(trk.curvature, float)), 1e-9)
    k = (np.roll(k, -1) + 2.0 * k + np.roll(k, 1)) / 4.0
    excess = []
    for i, vi in enumerate(v):
        j = (i + 1) % len(v)
        mu = float(car.mu) * max(
            0.55, 1.0 - float(car.load_sensitivity) * m * q * vi * vi / 4.0)
        cap = mu * (g + q * vi * vi)
        room = max(0.0, 1.0 - (vi * vi * k[i] / max(cap, 1e-9)) ** 2)
        a_tr = 0.90 * mu * (g + q * vi * vi) * math.sqrt(room) * 0.62
        accel = (min(a_tr, p_kg / max(vi, 4.0)) - cd * vi * vi
                 - crr - g * float(trk.grade[i]))
        reachable_v2 = max(6.0 ** 2, vi * vi + 2.0 * accel * ds[i])
        excess.append(v[j] ** 2 - reachable_v2)
    gate("every forward sample is physically reachable",
         max(excess) <= 1e-6,
         f"max v^2 excess {max(excess):.3e}; max speed jump "
         f"{np.max(np.roll(v, -1) - v):.2f} m/s")

    print("== 2. obs layout ==")
    attach_envelope(trk, FABLE_CAR)
    spec = SensorSpec()
    base = SensorSuite(spec).obs_size
    spec.pace_block = True
    spec.pace_distances = PACE_DISTANCES
    full = SensorSuite(spec).obs_size
    gate("pace block appends", full == base + len(PACE_DISTANCES) + 1,
         f"{base} -> {full}")
    gate("vref interp wraps", abs(vref_at(trk, trk.length + 5.0)
                                  - vref_at(trk, 5.0)) < 1e-9)

    print("== 2.5 RaceBox: downshifts through a braking zone ==")
    rb = RaceBox(car)

    class _Stub:
        gear, speed, rpm = len(rb.k), 85.0, 0.0
    stub = _Stub()
    max_rpm, trace = 0.0, []
    for step in range(240):                      # 85 -> 12 m/s over ~2.4 s
        stub.speed = max(12.0, 85.0 - step * 0.31)
        stub.rpm = stub.speed * rb.k[stub.gear - 1]
        max_rpm = max(max_rpm, stub.rpm)
        _, up, down = rb.update(stub, 0.0, 0.01)
        if up:
            stub.gear = min(len(rb.k), stub.gear + 1)
        if down:
            stub.gear = max(1, stub.gear - 1)
        trace.append(stub.gear)
    gate("downshifts under braking", stub.gear <= 2,
         f"gear {len(rb.k)} -> {stub.gear} at {stub.speed:.0f} m/s")
    gate("never money-shifts", max_rpm < float(car.cutoff_rpm),
         f"max rpm {max_rpm:.0f} < cutoff {car.cutoff_rpm:.0f}")
    gate("monotonic-ish walk down", all(b - a <= 1 for a, b in zip(trace, trace[1:])))

    print("== 3. env smoke ==")
    fspec = stage_defaults("foundation")
    cfg = PPOSpec()
    cfg.episode_seconds = 20.0
    cfg.random_start = True
    cfg.sensor_pace_block = True
    cfg.sensor_pace_distances = fspec.pace_distances
    cfg.sensor_lookahead_distances = fspec.lookahead_distances
    e = FableEnv(mode="race", car=FABLE_CAR, ppo=cfg, fixed_track=trk,
                 fable_spec=fspec, rng_seed=3, diagnostics=True)
    obs = e.reset()
    gate("obs dim = sensors + mode", len(obs) == full + 2, f"{len(obs)}")
    gate("3 actions (steer, long, gear offset)", e.action_dim == 3)
    rng = np.random.default_rng(0)
    total, done_reasons = 0.0, set()
    for _ in range(300):
        a = np.clip(rng.normal([0.0, 0.5, 0.0], 0.4), -1, 1)
        obs, r, term, trunc, info = e.step(a)
        total += r
        gate_ok = np.all(np.isfinite(obs)) and np.isfinite(r)
        if not gate_ok:
            gate("finite obs/reward", False)
        if term or trunc:
            done_reasons.add(info.get("termination_reason"))
            obs = e.reset()
    gate("finite obs/reward over 300 steps", True, f"ret {total:.1f}")
    gate("bookkeeping present", "fable_clean" in info and "fable_pace_ratio" in info)
    print(f"      termination reasons seen: {sorted(str(x) for x in done_reasons)}")

    print("== 4. train smoke ==")
    cfg2 = PPOSpec()
    cfg2.n_envs, cfg2.n_workers, cfg2.rollout = 2, 1, 24
    cfg2.episode_seconds = 15.0
    cfg2.sensor_pace_block = True
    cfg2.sensor_pace_distances = fspec.pace_distances
    cfg2.sensor_lookahead_distances = fspec.lookahead_distances
    cfg2.specialist_patience = 0
    fspec2 = stage_defaults("foundation")
    fspec2.eval_starts = 2
    fspec2.eval_sector_seconds = 3.0
    fspec2.eval_lap_budget = 5.0
    evaluator = FableEvaluator(fspec2, trk, "/tmp/fable5_smoke_manifest.json")
    ppo = PPO(mode="race", car=FABLE_CAR, ppo=cfg2, reward=fspec2.reward,
              fixed_track=trk, track_name="nordschleife",
              env_cls_override=FableEnv, env_kwargs={"fable_spec": fspec2},
              eval_callback=evaluator, extra_metadata=evaluator.metadata)
    ppo.train(iterations=2, checkpoint="/tmp/fable5_smoke.pt", log_every=1,
              save_every=2)
    gate("train loop runs", os.path.exists("/tmp/fable5_smoke.pt"))
    net, norm, meta = PPO.load_policy("/tmp/fable5_smoke.pt")
    gate("checkpoint round-trips", meta.get("obs_layout") == "fable-v1"
         and meta.get("sensor_pace_block") is True
         and int(meta["obs_dim"]) == full + 2
         and int(meta["act_dim"]) == 3,
         f"layout {meta.get('obs_layout')} obs {meta.get('obs_dim')} "
         f"act {meta.get('act_dim')}")
    gate("checkpoint carries autonomous-safety provenance",
         meta.get("fable_eval_protocol")
         and meta.get("fable_code_fingerprint")
         and meta.get("fable_drivetrain_version") == DRIVETRAIN_VERSION
         and meta.get("policy_sha256")
         and meta.get("norm_schema") in ("raw-v2", "legacy-frozen-v1")
         and isinstance(meta.get("ppo_config"), dict),
         f"protocol={meta.get('fable_eval_protocol')} norm={meta.get('norm_schema')}")

    print("== 4.5 six-speed checkpoint migration isolation ==")
    legacy = torch.load("/tmp/fable5_smoke.pt", map_location="cpu", weights_only=False)
    legacy["fable_drivetrain_version"] = "mazda787b-6spd-legacy"
    legacy["updates"] = 777
    legacy["state_dict"]["mean.weight"][2].fill_(1.0)
    legacy["state_dict"]["mean.bias"][2] = 0.8
    legacy["policy_sha256"] = state_dict_hash(legacy["state_dict"])
    gear_obs_i = SensorSpec().n_beams + 11
    legacy["norm_mean"][gear_obs_i] = 9.0
    legacy_path = "/tmp/fable5_6spd_migration_probe.pt"
    torch.save(legacy, legacy_path)
    msg = load_or_transplant(ppo, legacy_path, "foundation")
    gate("legacy drivetrain takes explicit migration path", msg.startswith("MIGRATED"), msg)
    gate("migration resets gear head and gear normalizer",
         float(ppo.net.mean.weight.data[2].abs().max()) == 0.0
         and abs(float(ppo.net.mean.bias.data[2])) < 1e-8
         and abs(float(ppo.norm.mean[gear_obs_i])) < 1e-8
         and int(ppo.updates) == 0)

    print("== 5. transplant (58s/2a hills-v1 -> 66s/3a fable-v1) ==")
    ring_best = "ring_787b_best.pt"
    if os.path.exists(ring_best):
        msg = load_or_transplant(ppo, ring_best, "foundation")
        gate("obs + action transplant", msg.startswith("TRANSPLANTED"), msg)
        gate("gear head starts neutral",
             abs(float(ppo.net.mean.bias.data[2])) < 1e-6
             and float(ppo.net.mean.weight.data[2].abs().max()) < 0.2,
             f"bias {float(ppo.net.mean.bias.data[2]):.3f}")
        o = ppo.vec.reset()
        nobs = ppo._norm(o)
        a = ppo.net.act_mean(ppo._t(nobs))
        gate("transplanted net acts (3 dims)",
             bool(torch.isfinite(a).all()) and a.shape[-1] == 3,
             f"act {a[0].tolist()}")
    else:
        print("  [SKIP] ring_787b_best.pt not present")

    print(f"ALL GATES PASSED in {time.time()-t0:.1f}s "
          f"(theoretical lap {lap:.1f}s = {lap/60:.0f}:{lap%60:05.2f})")


if __name__ == "__main__":
    main()
