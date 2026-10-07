"""ONE-COMMAND VISUAL REPORT (ROADMAP Stage 10.6, bpy-free orchestration).

`prepare_report` turns a finished run (archive + phylogeny) into a full visual
report's WORTH of inputs, doing everything that does NOT need Blender:

  * renders the innovation-annotated phylogeny tree (matplotlib, 10.3);
  * picks the BEST elite and the MOST-DIVERGENT creature and exports their scene
    plan / replay trajectory / biomechanics JSON (10.1 / 10.2 / 10.4);
  * exports the most-divergent lineage as a side-by-side comparison plan (10.5);
  * saves the genomes + a summary, and returns a render MANIFEST.

`scripts/render_run.py` then drives Blender headless over the manifest. Splitting it
this way keeps the orchestration unit-testable (no bpy) while a single command still
produces the whole report.
"""
from __future__ import annotations

import json
import os

from .phylo_render import render_phylogeny
from .morphology_plan import plan_from_genome
from .trajectory import record_trajectory
from .biomechanics import record_biomechanics
from .compare import lineage_plan, lineage_genomes
from ..evo.deeptime import most_divergent, tree_metrics


def _dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)
    return path


def prepare_report(archive, phylo, out_dir, *, replay_steps: int = 60,
                   name: str = "run") -> dict:
    """Build every Blender-free artifact + the render manifest. Returns
    {out_dir, manifest:[(json_in, out_name, kind)], summary}."""
    os.makedirs(out_dir, exist_ok=True)
    best = archive.best()
    best_g = best.genome
    md_id, md_dist = most_divergent(phylo)
    md_g = phylo.genome(md_id) if md_id is not None else best_g

    manifest = []
    # phylogeny tree (matplotlib -- rendered here, no Blender)
    render_phylogeny(phylo, os.path.join(out_dir, "phylogeny_tree.png"),
                     title=f"{name} phylogeny")

    _dump(plan_from_genome(best_g, name="best").to_dict(),
          os.path.join(out_dir, "best_plan.json"))
    manifest.append(("best_plan.json", "best_creature.png", "static"))

    traj = record_trajectory(best_g, n_steps=replay_steps, name="best")
    _dump(traj.to_dict(), os.path.join(out_dir, "best_traj.json"))
    manifest.append(("best_traj.json", "best_replay", "anim"))

    bio = record_biomechanics(best_g, n_steps=replay_steps, name="best")
    _dump(bio.to_dict(), os.path.join(out_dir, "best_bio.json"))
    manifest.append(("best_bio.json", "best_biomechanics", "anim"))

    genomes = lineage_genomes(phylo, md_id) or [best_g]
    _dump(lineage_plan(genomes, name="lineage").to_dict(),
          os.path.join(out_dir, "lineage_plan.json"))
    manifest.append(("lineage_plan.json", "lineage_comparison.png", "static"))

    with open(os.path.join(out_dir, "best.genome.json"), "w") as f:
        f.write(best_g.to_json())
    with open(os.path.join(out_dir, "most_divergent.genome.json"), "w") as f:
        f.write(md_g.to_json())

    tm = tree_metrics(phylo)
    summary = {
        "name": name,
        "coverage": archive.coverage, "qd_score": archive.qd_score,
        "best_fitness": best.fitness, "n_creatures": tm["n_creatures"],
        "max_depth": tm["max_depth"], "branch_points": tm["n_branch_points"],
        "most_divergent_distance": md_dist, "lineage_length": len(genomes),
        "artifacts": ["phylogeny_tree.png", "best_creature.png", "best_replay.(mp4|gif)",
                      "best_biomechanics.(mp4|gif)", "lineage_comparison.png"],
    }
    _dump(summary, os.path.join(out_dir, "summary.json"))
    return {"out_dir": out_dir, "manifest": manifest, "summary": summary}
