"""Record Foldspace folds into JSON clips for the web viewer.

Runs one fold per shape (the policy folding a flat strip into each silhouette),
capturing every frame's strip vertices, and writes a single replay file the
viewer cycles through.

    python3 -m foldspace.record --ckpt foldspace/checkpoints/fold.pt
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .env import VecFold, T_MAX, K, SHAPE_NAMES, N_SHAPE
from .networks import ActorCritic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="foldspace/checkpoints/fold.pt")
    ap.add_argument("--out", default="foldview/replays")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    agent = ActorCritic(ck["obs_dim"], ck["act_dim"])
    agent.load_state_dict(ck["state_dict"])
    agent.eval()

    env = VecFold(N_SHAPE, seed=args.seed)
    obs = env.set_targets(np.arange(N_SHAPE))     # one env per shape

    def centred_verts():
        v = env._vertices(env.c)                  # (N,K+1,2)
        c = v.mean(1, keepdims=True)
        return (v - c)

    frames = [[centred_verts()[e].round(3).tolist()] for e in range(N_SHAPE)]
    for _ in range(T_MAX):
        with torch.no_grad():
            act, _, _ = agent.act(torch.as_tensor(obs), deterministic=not args.stochastic)
        obs, rew, done, info = env.step(act.numpy(), auto_reset=False)
        v = centred_verts()
        for e in range(N_SHAPE):
            frames[e].append(v[e].round(3).tolist())

    meta = VecFold.metadata()
    clips = []
    for e in range(N_SHAPE):
        clips.append({
            "shape": SHAPE_NAMES[e],
            "score": round(float(info["score"][e]), 3),
            "frames": frames[e],
        })
    clips.sort(key=lambda c: -c["score"])         # best folds first

    os.makedirs(args.out, exist_ok=True)
    data = {"meta": {"K": meta["K"], "shapes": meta["shapes"],
                     "targets": meta["targets"]}, "clips": clips}
    with open(os.path.join(args.out, "folds.json"), "w") as f:
        json.dump(data, f, separators=(",", ":"))

    print("wrote", len(clips), "clips ->", os.path.join(args.out, "folds.json"))
    for c in clips:
        print(f"   {c['shape']:9s} score {c['score']*100:5.1f}%")


if __name__ == "__main__":
    main()
