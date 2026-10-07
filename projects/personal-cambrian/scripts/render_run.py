#!/usr/bin/env python3
"""One-command full visual report for a run (ROADMAP Stage 10.6).

Runs a short open-ended evolution, then renders the whole Stage-10 report headless:
the innovation-annotated phylogeny tree, the best creature (static + locomotion replay
+ biomechanics clip), and a most-divergent lineage comparison.

    python3 scripts/render_run.py --seed-creature quadruped --iterations 30
    python3 scripts/render_run.py --out-dir renders/myreport --steps 80

Each render is best-effort (a failure doesn't abort the rest). Needs Blender for the
3D pieces; the phylogeny tree renders even without it.
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.seeds import SEEDS
from personal_cambrian.evo import run_qd, QDConfig
from personal_cambrian.viz.report import prepare_report
from personal_cambrian.viz.video import stitch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUILDER = os.path.join(ROOT, "personal_cambrian", "viz", "blender_build.py")
_DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


def _blender(blender, json_in, out):
    subprocess.run([blender, "--background", "--python", _BUILDER, "--", json_in, out],
                   check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-creature", choices=list(SEEDS), default="quadruped")
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--steps", type=int, default=60, help="replay/biomechanics length")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--blender", default=_DEFAULT_BLENDER)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"evolving {args.seed_creature} for {args.iterations} iters ...")
    cfg = QDConfig(axes=["aspect", "limb_count"], bins=10, iterations=args.iterations,
                   seed_steps=2000, fine_tune_steps=500, n_envs=2, n_steps=128,
                   ep_steps=80, hidden=32, macro_rate=0.6, novelty=True, seed=args.seed)
    archive, phylo = run_qd([SEEDS[args.seed_creature]()], cfg)

    out_dir = args.out_dir or os.path.join(ROOT, "renders",
                                           time.strftime("%Y%m%d-%H%M%S") + "-report")
    print(f"preparing report -> {out_dir}")
    rep = prepare_report(archive, phylo, out_dir, replay_steps=args.steps,
                         name=args.seed_creature)
    print(f"  phylogeny + plans ready ({rep['summary']['n_creatures']} creatures, "
          f"most-divergent {rep['summary']['most_divergent_distance']:.2f})")

    if not os.path.exists(args.blender):
        print(f"Blender not found at {args.blender}; phylogeny_tree.png + plan JSONs written.")
        return

    for json_in, out_name, kind in rep["manifest"]:
        jpath = os.path.join(out_dir, json_in)
        try:
            if kind == "static":
                _blender(args.blender, jpath, os.path.join(out_dir, out_name))
                print(f"  rendered {out_name}")
            else:                                            # animation -> frames -> video
                frames_dir = os.path.join(out_dir, out_name + "_frames")
                _blender(args.blender, jpath, frames_dir)
                video = stitch(frames_dir, os.path.join(out_dir, out_name), fps=args.fps)
                print(f"  rendered {os.path.basename(video)}")
        except Exception as e:                               # noqa: BLE001
            print(f"  WARN {out_name} failed: {e}")

    print(f"\nfull visual report -> {out_dir}/")
    for a in rep["summary"]["artifacts"]:
        print(f"   {a}")


if __name__ == "__main__":
    main()
