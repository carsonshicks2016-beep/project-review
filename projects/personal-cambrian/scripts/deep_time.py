#!/usr/bin/env python3
"""THE DEEP-TIME EXPERIMENT -- the headline run (ROADMAP Stage 8.5).

Long open-ended evolution composing the whole stack: a generative genome grown into
MuJoCo, gated for viability/budget, controlled by warm-started modular PPO, and bred
in a MAP-Elites archive with the Stage-8.2 NOVELTY emitter and Stage-7.3 SPECIATION,
under a chosen Stage-7 NICHE. Periodically snapshots the archive; at the end it
analyzes the phylogeny for the Stage-8.5 done-when -- a DEEP, BRANCHING tree with
clearly NON-HUMAN lineages and an INNOVATION TIMELINE -- exports the NHX tree, plots
the open-endedness curves, and renders the most-divergent ("most alien") creature.

    python3 scripts/deep_time.py --iterations 300 --niche terrain --render
    python3 scripts/deep_time.py --iterations 120 --fine-tune 3000

This is the slow headline run; the analysis machinery is proven fast in
tests/test_deeptime.py.
"""
import argparse
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.seeds import SEEDS
from personal_cambrian.sim.tasks import NICHES
from personal_cambrian.evo import (
    run_qd, QDConfig, deep_time_report, innovation_series, cumulative,
    coverage_growth, plot_openendedness, plot_archive, has_matplotlib,
    save_metrics,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-creature", default="quadruped", choices=list(SEEDS))
    ap.add_argument("--niche", default="locomotion", choices=list(NICHES))
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--seed-steps", type=int, default=30_000)
    ap.add_argument("--fine-tune", type=int, default=3_000)
    ap.add_argument("--ep-steps", type=int, default=200)
    ap.add_argument("--bins", type=int, default=14)
    ap.add_argument("--macro-rate", type=float, default=0.5)
    ap.add_argument("--divergence", type=float, default=0.3, help="non-human distance cutoff")
    ap.add_argument("--no-speciation", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    cfg = QDConfig(
        axes=["aspect", "limb_count"], bins=args.bins, niche=args.niche,
        iterations=args.iterations, seed_steps=args.seed_steps,
        fine_tune_steps=args.fine_tune, ep_steps=args.ep_steps, macro_rate=args.macro_rate,
        novelty=True, speciation=not args.no_speciation,
        snapshot_every=max(1, args.iterations // 10), seed=args.seed)
    print(f"DEEP TIME: {args.seed_creature} under '{args.niche}', {args.iterations} iters, "
          f"novelty+{'speciation' if not args.no_speciation else 'no-speciation'}")
    logs, t0 = [], time.time()
    archive, phylo = run_qd([SEEDS[args.seed_creature]()], cfg, log_fn=logs.append,
                            verbose=True)
    print(f"\nwall_time={time.time()-t0:.0f}s")

    # --- the Stage-8.5 assessment ------------------------------------------
    rep = deep_time_report(phylo, divergence_threshold=args.divergence, min_depth=5)
    tm = rep["tree"]
    print(f"\nPHYLOGENY: {tm['n_creatures']} creatures, depth {tm['max_depth']}, "
          f"{tm['n_branch_points']} branch points, {tm['n_leaves']} leaves")
    print(f"NON-HUMAN LINEAGES (dist > {args.divergence}): {rep['n_non_human']}  "
          f"| most divergent: {rep['most_divergent']['distance']:.2f}")
    print(f"innovation timeline ({rep['n_innovation_types']} types):")
    for e in rep["innovation_timeline"]:
        print(f"   gen {e['generation']:3d}  {e['flag']}")
    innov_status = ("SUSTAINED" if rep["innovation_sustained"]
                    else f"plateaued at gen {rep['innovation_plateau_at']}")
    print(f"innovation: {innov_status}")
    verdict = "MET" if rep["met"] else "not yet (longer run)"
    print(f"\nDONE-WHEN: {rep['done_when']}  ->  {verdict}")

    # --- persist + visualize -----------------------------------------------
    run_dir = os.path.join(ROOT, "runs", time.strftime("%Y%m%d-%H%M%S") + "-deeptime")
    os.makedirs(run_dir, exist_ok=True)
    phylo.export(os.path.join(run_dir, "phylogeny"))       # NHX .nwk + .json
    save_metrics(logs, os.path.join(run_dir, "metrics.jsonl"))
    saved = "phylogeny.nwk (NHX), phylogeny.json, metrics.jsonl"

    top = rep["most_divergent"]["node"]
    if top is not None:
        g = phylo.genome(top)
        if g is not None:
            with open(os.path.join(run_dir, "most_divergent.genome.json"), "w") as f:
                f.write(g.to_json())
            saved += ", most_divergent.genome.json"

    if has_matplotlib():
        innov = cumulative(innovation_series(phylo))
        cov = cumulative(coverage_growth(logs, key="cells"))
        plot_openendedness({"innovations": innov, "coverage": cov},
                           os.path.join(run_dir, "openendedness.png"),
                           plateaus={"innovations": rep["innovation_plateau_at"]})
        plot_archive(archive, os.path.join(ROOT, "renders", "deeptime_archive.png"))
        saved += ", openendedness.png; renders/deeptime_archive.png"
    print(f"\nsaved -> {run_dir}/ ({saved})")

    if args.render and top is not None and phylo.genome(top) is not None:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from view_creature import render
        out = os.path.join(ROOT, "renders", "deeptime_most_divergent.png")
        render(phylo.genome(top), out)
        print(f"rendered most-divergent creature -> {out}")


if __name__ == "__main__":
    main()
