"""Named-set checkpoint evaluation — the Stage 7 / Gate 7 measuring stick.

Scores a PPO checkpoint the HONEST way (deterministic act_mean, full-episode
budget, fixed starts around the loop — exactly how you watch it) across the
named track set, and optionally repeats the whole pass on FLAT variants of
the same tracks (the secondary metric: hill competence must not cost flat
competence — Gate 7 tolerance is ~10%).

`ridge` is in the default set as the jump/commitment probe: a good hills
policy either clears its crests with intent (jumps > 0, laps intact) or
correctly declines by speed — sustained airtime with collapsing laps is the
jump-exploit signature.

Usage (from the repo root):
  PYTHONPATH="$PWD" python3 tools/eval_checkpoint.py CHECKPOINT.pt
  ... --tracks club,akina,ridge --starts 4 --flat
Exits non-zero if the checkpoint can't load (e.g. pre-hills) or no track
completes a single step.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_TRACKS = "club,national,akina,ridge"


def eval_on_track(net, norm, meta, track, n_starts, budget_s):
    """The trainer's _deterministic_eval loop, standalone: act_mean from
    n_starts fixed points at ~racing pace, full-episode budget each."""
    import torch
    from supra.config import PPOSpec
    from supra.ppo_env import SupraEnv

    cfg = PPOSpec()
    cfg.random_start = False
    env = SupraEnv(mode=meta.get("mode", "race"), car=meta.get("car", "supra"),
                   ppo=cfg, fixed_track=track, rng_seed=999)
    sdim = int(meta["sdim"])
    steps_per = int(round((budget_s or cfg.episode_seconds) * cfg.control_hz))
    M = len(track.center)
    drifts, laps = [], []
    air, jumps, landing_g = 0.0, 0, 0.0
    for s in range(n_starts):
        idx = int(M * s / n_starts)
        k = abs(float(track.curvature[idx]))
        nat = min(30.0, (11.0 / k) ** 0.5) if k > 1e-4 else 30.0
        obs = env.reset_at(idx, speed=nat * 0.85)
        info, best_lap = {}, 0.0
        for _ in range(steps_per):
            nobs = np.concatenate([norm.normalize(obs[:sdim]), obs[sdim:]])
            with torch.no_grad():
                a = net.act_mean(torch.as_tensor(
                    nobs.astype(np.float32)).unsqueeze(0)).squeeze(0).numpy()
            obs, _, term, trunc, info = env.step(a)
            best_lap = max(best_lap, info.get("laps", 0.0))
            if term or trunc:
                break
        drifts.append(info.get("drift_frac", 0.0))
        laps.append(min(best_lap, 1.0))
        air += info.get("airtime", 0.0)
        jumps += info.get("jumps", 0)
        landing_g = max(landing_g, info.get("max_landing_g", 0.0))
    return {"lap": float(np.mean(laps)), "drift": float(np.mean(drifts)),
            "jumps": jumps, "air": air, "max_landing_g": landing_g}


def composite(mode, r):
    if mode == "drift":
        return r["drift"] * (0.3 + 0.7 * r["lap"])
    if mode == "hybrid":
        return r["lap"] * (0.6 + 0.4 * r["drift"])
    return r["lap"]


def run_pass(net, norm, meta, names, n_starts, budget_s, label):
    from supra.track import named_track
    mode = meta.get("mode", "race")
    print(f"\n{label} ({mode}, {n_starts} starts/track, act_mean):")
    out = {}
    for name in names:
        r = eval_on_track(net, norm, meta, named_track(name), n_starts, budget_s)
        out[name] = r
        extra = (f"  jumps={r['jumps']} air={r['air']:.1f}s "
                 f"land {r['max_landing_g']:.1f}g" if r["jumps"] else "")
        print(f"  {name:<10} lap {r['lap']:.2f}  drift {r['drift']:.2f}  "
              f"score {composite(mode, r):.3f}{extra}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help=".pt checkpoint to score")
    ap.add_argument("--tracks", default=DEFAULT_TRACKS,
                    help=f"comma list of named tracks (default {DEFAULT_TRACKS})")
    ap.add_argument("--starts", type=int, default=4, help="starts per track")
    ap.add_argument("--budget", type=float, default=None,
                    help="seconds per start (default: full episode)")
    ap.add_argument("--flat", action="store_true",
                    help="ALSO score flat variants (the Gate 7 secondary metric)")
    args = ap.parse_args()

    from supra.ppo import PPO
    from supra.track import configure_hills
    try:
        net, norm, meta = PPO.load_policy(args.checkpoint)
    except Exception as e:
        print(f"FAIL: cannot load {args.checkpoint}:\n  {e}")
        return 2

    names = [n.strip() for n in args.tracks.split(",") if n.strip()]
    mode = meta.get("mode", "race")
    print(f"checkpoint {os.path.basename(args.checkpoint)} — {mode} · "
          f"{meta.get('updates', 0)}u · car {meta.get('car', 'supra')} · "
          f"obs {meta.get('obs_dim')}")

    hills = run_pass(net, norm, meta, names, args.starts, args.budget,
                     "HILLS (the world as trained)")
    if args.flat:
        try:
            configure_hills(enabled=False)
            flat = run_pass(net, norm, meta, names, args.starts, args.budget,
                            "FLAT (secondary metric — competence must survive)")
        finally:
            configure_hills(enabled=True)
        print("\nhills vs flat (lap):")
        for n in names:
            h, f = hills[n]["lap"], flat[n]["lap"]
            d = h - f
            print(f"  {n:<10} hills {h:.2f}  flat {f:.2f}  Δ{d:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
