#!/usr/bin/env python3
"""Render a locomotion replay video of a creature (ROADMAP Stage 10.2).

Records a MuJoCo rollout (open-loop gait by default), exports the per-frame body
transforms + scene plan to JSON, then drives Blender headless to keyframe and render
a video.

    python3 scripts/replay_creature.py --seed-creature quadruped --steps 120
    python3 scripts/replay_creature.py --genome runs/.../most_divergent.genome.json --niche terrain

Needs Blender (defaults to the macOS app path).
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.seeds import SEEDS
from personal_cambrian.encoding.genome import Genome
from personal_cambrian.viz import record_trajectory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-creature", choices=list(SEEDS), default="quadruped")
    ap.add_argument("--genome", help="genome JSON path (overrides --seed-creature)")
    ap.add_argument("--niche", default="locomotion")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--blender", default=_DEFAULT_BLENDER)
    args = ap.parse_args()

    if args.genome:
        with open(args.genome) as f:
            genome = Genome.from_dict(json.load(f))
        name = os.path.splitext(os.path.basename(args.genome))[0]
    else:
        name = args.seed_creature
        genome = SEEDS[name]()

    traj = record_trajectory(genome, niche=args.niche, n_steps=args.steps,
                             fps=args.fps, seed=args.seed, name=name)
    print(f"{name}: {traj.n_frames} frames, root travelled {traj.root_displacement():.2f}m")

    os.makedirs(os.path.join(ROOT, "renders"), exist_ok=True)
    traj_path = os.path.join(ROOT, "renders", f"{name}_traj.json")
    frames_dir = os.path.join(ROOT, "renders", f"{name}_frames")
    out_stem = (args.out or os.path.join(ROOT, "renders", f"{name}_replay")).rsplit(".", 1)[0]
    with open(traj_path, "w") as f:
        json.dump(traj.to_dict(), f)

    if not os.path.exists(args.blender):
        print(f"Blender not found at {args.blender}; trajectory written to {traj_path}.")
        return
    builder = os.path.join(ROOT, "personal_cambrian", "viz", "blender_build.py")
    subprocess.run([args.blender, "--background", "--python", builder, "--",
                    traj_path, frames_dir], check=True)
    from personal_cambrian.viz.video import stitch
    video = stitch(frames_dir, out_stem, fps=args.fps)
    print(f"replay -> {video}")


if __name__ == "__main__":
    main()
