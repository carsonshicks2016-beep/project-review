"""Score a trained policy against the scripted pilot, stage by stage."""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

import numpy as np

from drift.config import CURRICULUM, stage_by_name
from drift.env import DriftEnv
from drift.pilot import DriftPilot

TRICK_KEYS = ("n_switchback", "n_donut", "n_clip", "n_manji", "n_wall_kiss", "n_big_angle")


def rollout(env, act, seed):
    obs, _ = env.reset(seed=seed)
    ret = 0.0
    while True:
        obs, r, term, trunc, info = env.step(act(env, obs))
        ret += r
        if term or trunc:
            return ret, info


def run(env, name, act, episodes, seed0):
    rets, scores, combos, mults, crashes, wipes = [], [], [], [], 0, 0
    tricks = dict.fromkeys(TRICK_KEYS, 0)
    for ep in range(episodes):
        ret, info = rollout(env, act(env), seed0 + ep)
        rets.append(ret)
        scores.append(info.get("score", 0.0))
        combos.append(info.get("best_combo", 0.0))
        mults.append(info.get("best_multiplier", 1.0))
        crashes += bool(info.get("crashed"))
        wipes += info.get("wipeouts", 0)
        for k in tricks:
            tricks[k] += info.get(k, 0)
    return {
        "name": name, "score": st.mean(scores), "score_sd": st.pstdev(scores),
        "ret": st.mean(rets), "combo": st.mean(combos), "mult": st.mean(mults),
        "crash": crashes / episodes, "wipes": wipes / episodes, "tricks": tricks,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True, help="run directory or .zip")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--stages", nargs="*", default=[s.name for s in CURRICULUM])
    args = p.parse_args()

    from play import load_policy
    model, norm = load_policy(args.model)

    def policy_act(env):
        def _a(env, obs):
            o = norm.normalize_obs(obs[None]) if norm is not None else obs[None]
            a, _ = model.predict(o, deterministic=True)
            return a[0]
        return _a

    def pilot_act(env):
        pilot = DriftPilot(slip_deg=38.0, speed=16.0, flip_every=2.5, arena=env.arena)
        def _a(env, obs):
            stg, th, hb = pilot.act(env.state)
            return np.array([stg, th, hb * 2 - 1])
        return _a

    print(f"{'stage':10s} {'driver':7s} {'score':>16} {'return':>8} {'best combo':>11} "
          f"{'maxmult':>8} {'crash':>6} {'wipe':>5}  tricks/ep")
    print("-" * 108)
    for name in args.stages:
        env = DriftEnv(stage=stage_by_name(name))
        for label, act in (("policy", policy_act), ("pilot", pilot_act)):
            r = run(env, label, act, args.episodes, args.seed)
            tr = " ".join(f"{k[2:]}={v / args.episodes:.1f}" for k, v in r["tricks"].items() if v)
            print(f"{name:10s} {label:7s} {r['score']:9,.0f}+-{r['score_sd']:5,.0f} "
                  f"{r['ret']:8.1f} {r['combo']:11,.0f} {r['mult']:8.1f} "
                  f"{r['crash']:5.0%} {r['wipes']:5.1f}  {tr}")
        print()


if __name__ == "__main__":
    main()
