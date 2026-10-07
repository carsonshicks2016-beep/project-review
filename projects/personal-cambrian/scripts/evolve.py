#!/usr/bin/env python3
"""Run the MAP-Elites quality-diversity loop (ROADMAP Stage 5.3 demonstration).

Evolves a diverse archive of locomoting creatures from a seed, then reports the
archive (coverage / QD-score / 2-D heatmap), saves the phylogeny, and renders the
best evolved creature.

    python3 scripts/evolve.py --iterations 60
    python3 scripts/evolve.py --iterations 120 --render
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.seeds import quadruped
from personal_cambrian.evo import (
    run_qd, QDConfig, save_metrics, plot_metrics, plot_archive, has_matplotlib,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--seed-steps", type=int, default=50_000)
    ap.add_argument("--fine-tune-steps", type=int, default=4_096)
    ap.add_argument("--bins", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    cfg = QDConfig(axes=["aspect", "limb_count"], bins=args.bins,
                   iterations=args.iterations, seed_steps=args.seed_steps,
                   fine_tune_steps=args.fine_tune_steps,
                   snapshot_every=max(1, args.iterations // 10), seed=args.seed)
    print(f"evolving from quadruped: {cfg.iterations} iterations, "
          f"axes={cfg.axes}, bins={cfg.bins}")
    logs = []
    t0 = time.time()
    archive, phylo = run_qd([quadruped()], cfg, log_fn=logs.append, verbose=True)

    print(f"\nwall_time={time.time()-t0:.0f}s")
    print(f"ARCHIVE: {len(archive.grid)}/{archive.n_cells} cells  "
          f"coverage={archive.coverage:.1%}  qd_score={archive.qd_score:.2f}  "
          f"evals={archive.n_evals}")
    best = archive.best()
    bdesc = {k: round(v, 2) for k, v in best.descriptors.items()}
    print(f"best fitness (forward distance) = {best.fitness:+.2f}m  descriptors={bdesc}")

    print("\nARCHIVE MAP  (x -> aspect ratio,  y -> limb count;  fitness ' .:-=+*#%@')")
    for line in archive.ascii_map().splitlines():
        print("  " + line)

    # phylogeny / innovation summary
    flags = Counter(f for n in phylo.nodes.values() for f in n.innovations)
    maxgen = max((n.generation for n in phylo.nodes.values()), default=0)
    print(f"\nPHYLOGENY: {len(phylo.nodes)} creatures, deepest lineage = generation {maxgen}")
    print(f"innovations seen: {dict(flags)}")

    run_dir = os.path.join(ROOT, "runs", time.strftime("%Y%m%d-%H%M%S") + "-qd")
    os.makedirs(run_dir, exist_ok=True)
    phylo.export(os.path.join(run_dir, "phylogeny"))   # -> phylogeny.nwk (NHX) + .json
    save_metrics(logs, os.path.join(run_dir, "metrics.jsonl"))
    with open(os.path.join(run_dir, "best.genome.json"), "w") as f:
        f.write(best.genome.to_json())
    saved = "phylogeny.nwk (NHX), phylogeny.json, metrics.jsonl, best.genome.json"
    if has_matplotlib():
        plot_metrics(logs, os.path.join(run_dir, "qd_progress.png"))
        plot_archive(archive, os.path.join(ROOT, "renders", "qd_archive.png"))
        saved += ", qd_progress.png; renders/qd_archive.png"
    print(f"\nsaved -> {run_dir}/ ({saved})")

    if args.render:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from view_creature import render
        out = os.path.join(ROOT, "renders", "qd_best.png")
        render(best.genome, out)
        print(f"rendered best -> {out}")


if __name__ == "__main__":
    main()
