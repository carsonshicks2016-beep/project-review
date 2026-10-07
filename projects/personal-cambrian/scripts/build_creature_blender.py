#!/usr/bin/env python3
"""Build a creature in Blender from its genome/morphology (ROADMAP Stage 10.1).

Exports the creature's bpy-free ScenePlan to JSON, then drives Blender headless to
build it (capsule/box/sphere/ellipsoid bodies + muscle curves) and render a PNG.

    python3 scripts/build_creature_blender.py --seed-creature quadruped
    python3 scripts/build_creature_blender.py --genome runs/.../most_divergent.genome.json

Needs Blender installed (defaults to the macOS app path). The bpy builder also works
inside a live Blender via the MCP `execute_blender_code`; this script is the headless
one-command path.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.seeds import SEEDS
from personal_cambrian.encoding.genome import Genome
from personal_cambrian.viz import plan_from_genome

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-creature", choices=list(SEEDS))
    ap.add_argument("--genome", help="path to a genome JSON (overrides --seed-creature)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--blender", default=_DEFAULT_BLENDER)
    args = ap.parse_args()

    if args.genome:
        with open(args.genome) as f:
            genome = Genome.from_dict(json.load(f))
        name = os.path.splitext(os.path.basename(args.genome))[0]
    else:
        name = args.seed_creature or "quadruped"
        genome = SEEDS[name]()

    plan = plan_from_genome(genome, name=name)
    os.makedirs(os.path.join(ROOT, "renders"), exist_ok=True)
    plan_path = os.path.join(ROOT, "renders", f"{name}_plan.json")
    out_png = args.out or os.path.join(ROOT, "renders", f"{name}_blender.png")
    with open(plan_path, "w") as f:
        json.dump(plan.to_dict(), f)
    print(f"{name}: {len(plan.geoms)} bodies, {len(plan.muscles)} muscles -> {plan_path}")

    if not os.path.exists(args.blender):
        print(f"Blender not found at {args.blender}; plan JSON written -- run blender_build.py "
              f"-- {plan_path} {out_png} inside Blender.")
        return
    builder = os.path.join(ROOT, "personal_cambrian", "viz", "blender_build.py")
    subprocess.run([args.blender, "--background", "--python", builder, "--",
                    plan_path, out_png], check=True)
    print(f"rendered -> {out_png}")


if __name__ == "__main__":
    main()
