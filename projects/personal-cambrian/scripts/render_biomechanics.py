#!/usr/bin/env python3
"""Render an annotated biomechanics clip (ROADMAP Stage 10.4).

Records a rollout with per-muscle forces + moment arms, then drives Blender headless
to animate the creature with force-scaled muscle bars (red, radius ~ tension) and
moment-arm levers (green, joint -> muscle line), and stitches a video.

    python3 scripts/render_biomechanics.py --seed-creature quadruped --steps 80
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.seeds import SEEDS
from personal_cambrian.encoding.genome import Genome
from personal_cambrian.viz import record_biomechanics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-creature", choices=list(SEEDS), default="quadruped")
    ap.add_argument("--genome", help="genome JSON path (overrides --seed-creature)")
    ap.add_argument("--niche", default="locomotion")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blender", default=_DEFAULT_BLENDER)
    args = ap.parse_args()

    if args.genome:
        with open(args.genome) as f:
            genome = Genome.from_dict(json.load(f))
        name = os.path.splitext(os.path.basename(args.genome))[0]
    else:
        name = args.seed_creature
        genome = SEEDS[name]()

    bio = record_biomechanics(genome, niche=args.niche, n_steps=args.steps,
                              fps=args.fps, seed=args.seed, name=name)
    print(f"{name}: {bio.n_frames} frames, peak muscle force {bio.force_max:.2f}N, "
          f"peak moment arm {bio.moment_max:.3f}m")

    os.makedirs(os.path.join(ROOT, "renders"), exist_ok=True)
    bio_path = os.path.join(ROOT, "renders", f"{name}_bio.json")
    frames_dir = os.path.join(ROOT, "renders", f"{name}_bio_frames")
    out_stem = os.path.join(ROOT, "renders", f"{name}_biomechanics")
    with open(bio_path, "w") as f:
        json.dump(bio.to_dict(), f)

    if not os.path.exists(args.blender):
        print(f"Blender not found at {args.blender}; biomechanics JSON at {bio_path}.")
        return
    builder = os.path.join(ROOT, "personal_cambrian", "viz", "blender_build.py")
    subprocess.run([args.blender, "--background", "--python", builder, "--",
                    bio_path, frames_dir], check=True)
    from personal_cambrian.viz.video import stitch
    video = stitch(frames_dir, out_stem, fps=args.fps)
    print(f"biomechanics clip -> {video}")


if __name__ == "__main__":
    main()
