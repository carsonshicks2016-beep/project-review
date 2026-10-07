"""Record self-play matches into compact JSON replays for the web viewer.

Runs a batch of duels (one per arena), captures every frame, writes one file per
match plus a manifest the viewer uses as a playlist / highlight reel.

    python3 -m duel.record --ckpt duel/checkpoints/duel.pt --matches 12
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .env import VecDuel, DT, MAX_STEPS
from .networks import ActorCritic, ACT_DIM


def _frame(snap, e):
    agents = []
    for p in range(2):
        flags = (int(snap["invuln"][e, p]) |
                 (int(snap["dash_t"][e, p]) << 1) |
                 (int(snap["fired"][e, p] > 0) << 2))
        agents.append([
            round(float(snap["pos"][e, p, 0]), 2),
            round(float(snap["pos"][e, p, 1]), 2),
            round(float(snap["face"][e, p]), 3),
            round(float(snap["health"][e, p]), 1),
            flags,
        ])
    bullets = []
    for i in range(snap["b_life"].shape[1]):
        if snap["b_life"][e, i] > 0.0:
            bullets.append([
                round(float(snap["b_pos"][e, i, 0]), 2),
                round(float(snap["b_pos"][e, i, 1]), 2),
                int(snap["b_owner"][e, i]),
            ])
    return {"a": agents, "b": bullets}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="duel/checkpoints/duel.pt")
    ap.add_argument("--matches", type=int, default=12)
    ap.add_argument("--out", default="viewer/replays")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    agent = ActorCritic(ckpt["obs_dim"])
    agent.load_state_dict(ckpt["state_dict"])
    agent.eval()

    N = args.matches
    env = VecDuel(N, seed=args.seed)
    obs = env.reset()
    frames = [[] for _ in range(N)]
    finished = np.zeros(N, bool)

    snap = env.snapshot()
    for e in range(N):
        frames[e].append(_frame(snap, e))

    for _ in range(MAX_STEPS):
        with torch.no_grad():
            act, _, _ = agent.act(torch.as_tensor(obs.reshape(2 * N, -1)),
                                  deterministic=args.deterministic)
        obs, rew, done, info = env.step(act.numpy().reshape(N, 2, ACT_DIM), auto_reset=False)
        snap = env.snapshot()
        for e in range(N):
            if finished[e]:
                continue
            frames[e].append(_frame(snap, e))
            if done[e]:
                finished[e] = True
        if finished.all():
            break

    os.makedirs(args.out, exist_ok=True)
    meta = VecDuel.metadata()
    manifest = []
    for e in range(N):
        fr = frames[e]
        hp = [fr[-1]["a"][0][3], fr[-1]["a"][1][3]]
        winner = 0 if hp[1] <= 0 < hp[0] else (1 if hp[0] <= 0 < hp[1] else -1)
        # total hits = number of 34-ish health drops across the match
        hits = 0
        for k in range(1, len(fr)):
            for p in range(2):
                if fr[k]["a"][p][3] < fr[k - 1]["a"][p][3]:
                    hits += 1
        dur = round(len(fr) * DT, 1)
        name = f"match_{e:02d}.json"
        data = {"meta": meta, "winner": winner, "duration": dur,
                "hits": hits, "frames": fr}
        with open(os.path.join(args.out, name), "w") as f:
            json.dump(data, f, separators=(",", ":"))
        manifest.append({"file": name, "winner": winner, "duration": dur, "hits": hits})

    # highlight ordering: decisive first, then more action, then snappier
    manifest.sort(key=lambda m: (m["winner"] < 0, -m["hits"], m["duration"]))
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"meta": meta, "matches": manifest}, f, indent=2)

    decisive = sum(1 for m in manifest if m["winner"] >= 0)
    print(f"wrote {N} replays to {args.out}  ({decisive} decisive, "
          f"avg {np.mean([m['duration'] for m in manifest]):.1f}s, "
          f"avg hits {np.mean([m['hits'] for m in manifest]):.1f})")


if __name__ == "__main__":
    main()
