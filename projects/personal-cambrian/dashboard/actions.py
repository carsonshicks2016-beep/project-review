"""Action definitions -- the full control surface over the engine.

Every runner imports the engine LAZILY (inside the function) so the server starts
instantly and only pays the mujoco/torch import cost when an action actually runs.
Long jobs stream metrics via `ctx.metric(...)` and honour `ctx.check_stop()`.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

from .registry import Field, action
from .jobs import JobCancelled
from .artifacts import artifacts_modified_since

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEEDS = ["quadruped", "agent_zero"]
NICHES = ["locomotion", "track", "iron_zone", "ballistics", "endurance",
          "chaos_grid", "terrain"]
MACRO_OPS = ["micro", "add_part", "duplicate_subtree", "delete_subtree", "add_muscle",
             "remove_muscle", "reroute_tendon", "split_muscle", "toggle_symmetry",
             "change_recursion_count", "add_joint_dof", "remove_dof"]


def _rel(path: str) -> str:
    return os.path.relpath(path, ROOT)


def _seed(name: str):
    from personal_cambrian.seeds import SEEDS as S
    return S[name]()


def _run_script(ctx, args, *, dirs=("renders", "runs")):
    """Run an engine script as a subprocess, streaming stdout, cancellable, and
    collecting any artifacts it writes."""
    start = time.time()
    ctx.status(f"running {args[0]}")
    proc = subprocess.Popen([sys.executable] + args, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if not line or "warp" in line.lower() or "blenderkit" in line.lower():
                continue
            ctx.log(line)
            if ctx.should_stop:
                proc.terminate()
                raise JobCancelled()
    finally:
        proc.wait()
    arts = artifacts_modified_since(start, dirs)
    for a in arts:
        ctx.artifact(a)
    return {"returncode": proc.returncode, "artifacts": arts}


# ============================ GENOME ======================================
@action(id="genome.develop", label="Develop & inspect", category="Genome", kind="value",
        description="Grow a seed genome into a morphology; report stats + the pre-sim gate.",
        fields=[Field("seed_creature", "enum", "quadruped", options=SEEDS, label="creature")])
def _develop(p, ctx):
    from personal_cambrian.morphogenesis import develop, compute_budgets
    from personal_cambrian.evo.descriptors import morphology_descriptors
    from personal_cambrian.evo.qd import pre_sim_gate
    g = _seed(p["seed_creature"])
    m = develop(g)
    d = morphology_descriptors(g)
    ok, reasons = pre_sim_gate(g)
    budgets = compute_budgets(m, g)
    return {"creature": p["seed_creature"], "bodies": m.body_count,
            "muscles": len(g.muscles), "mass_kg": round(m.total_mass(), 2),
            "descriptors": {k: round(v, 3) for k, v in d.items()},
            "viable": ok, "reject_reasons": reasons,
            "budget_penalty": round(budgets.penalty(), 4)}


@action(id="genome.mutate", label="Mutate", category="Genome", kind="value",
        description="Apply a micro/macro mutation; report the structural innovations it introduced.",
        fields=[Field("seed_creature", "enum", "quadruped", options=SEEDS, label="creature"),
                Field("op", "enum", "duplicate_subtree", options=MACRO_OPS, label="operator"),
                Field("sigma", "float", 0.15, min=0.0, max=1.0, step=0.05, label="micro sigma"),
                Field("seed", "int", 0, min=0, max=9999)])
def _mutate(p, ctx):
    import numpy as np
    import personal_cambrian.encoding.mutate as mm
    from personal_cambrian.encoding.innovation import diff
    from personal_cambrian.morphogenesis import develop
    g = _seed(p["seed_creature"])
    rng = np.random.default_rng(p["seed"])
    op = p["op"]
    if op == "micro":
        child, rec_ok = mm.micro(g, rng, p["sigma"]), True
    else:
        child, record = getattr(mm, op)(g, rng)
        rec_ok = record.ok
    rep = diff(g, child)
    m = develop(child)
    return {"operator": op, "applied": rec_ok,
            "innovations": sorted(rep.flags), "graph_distance": rep.distance,
            "bodies": m.body_count, "muscles": len(child.muscles),
            "mass_kg": round(m.total_mass(), 2)}


# ============================ SIMULATION ==================================
@action(id="sim.rollout", label="Record rollout", category="Simulation", kind="value",
        description="Run a MuJoCo rollout under a niche (open-loop gait) and report motion.",
        fields=[Field("seed_creature", "enum", "quadruped", options=SEEDS, label="creature"),
                Field("niche", "enum", "locomotion", options=NICHES),
                Field("steps", "int", 80, min=10, max=400, step=10),
                Field("seed", "int", 0, min=0, max=9999)])
def _rollout(p, ctx):
    from personal_cambrian.viz import record_trajectory
    import json
    ctx.status("simulating")
    t = record_trajectory(_seed(p["seed_creature"]), niche=p["niche"],
                          n_steps=p["steps"], seed=p["seed"], name=p["seed_creature"])
    out = os.path.join(ROOT, "renders", f"{p['seed_creature']}_dash_traj.json")
    with open(out, "w") as f:
        json.dump(t.to_dict(), f)
    ctx.artifact(_rel(out))
    return {"frames": t.n_frames, "root_displacement_m": round(t.root_displacement(), 3),
            "artifacts": [_rel(out)]}


# ============================ CONTROL =====================================
@action(id="control.train_ppo", label="Train PPO", category="Control", kind="stream",
        streams=["mean_return"],
        description="Train a policy with PPO; live return curve; stoppable mid-run.",
        fields=[Field("seed_creature", "enum", "quadruped", options=SEEDS, label="creature"),
                Field("niche", "enum", "locomotion", options=NICHES),
                Field("steps", "int", 20000, min=2000, max=300000, step=2000, label="timesteps"),
                Field("n_envs", "int", 4, min=1, max=16),
                Field("hidden", "int", 64, min=16, max=256, step=16),
                Field("lr", "float", 0.0003, min=0.00001, max=0.01, step=0.0001),
                Field("seed", "int", 0, min=0, max=9999)])
def _train(p, ctx):
    from personal_cambrian.sim import CreatureEnv
    from personal_cambrian.sim.tasks import NICHES as N
    from personal_cambrian.control import PPOConfig
    from personal_cambrian.control.ppo import train
    g = _seed(p["seed_creature"])
    ep = 200

    def cme(s):
        return CreatureEnv(g, task=N[p["niche"]](max_steps=ep), obs_mode="structured")

    def log(rec):
        ctx.metric(iter=rec["iter"], mean_return=round(rec["mean_return"], 3),
                   step=rec["global_step"])
        ctx.check_stop()

    ctx.status("training")
    _policy, hist = train(cme, PPOConfig(total_timesteps=p["steps"], n_envs=p["n_envs"],
                                         n_steps=256, hidden=p["hidden"], lr=p["lr"],
                                         seed=p["seed"]), log_fn=log)
    return {"iters": len(hist), "final_return": round(hist[-1]["mean_return"], 3)}


# ============================ EVOLUTION (QD) ==============================
def _qd_artifacts(ctx, archive, phylo, tag):
    from personal_cambrian.evo import plot_archive, has_matplotlib
    from personal_cambrian.viz import render_phylogeny
    run_dir = os.path.join(ROOT, "runs", time.strftime("%Y%m%d-%H%M%S") + f"-dash-{tag}")
    os.makedirs(run_dir, exist_ok=True)
    phylo.export(os.path.join(run_dir, "phylogeny"))
    arts = []
    if has_matplotlib():
        try:
            p1 = os.path.join(run_dir, "archive.png")
            plot_archive(archive, p1)
            ctx.artifact(_rel(p1)); arts.append(_rel(p1))
        except Exception:        # noqa: BLE001
            pass
        p2 = os.path.join(run_dir, "phylogeny_tree.png")
        render_phylogeny(phylo, p2, title=f"{tag} phylogeny")
        ctx.artifact(_rel(p2)); arts.append(_rel(p2))
    return arts


@action(id="evo.run_qd", label="Run Quality-Diversity", category="Evolution", kind="stream",
        streams=["qd_score", "coverage", "novelty"],
        description="The MAP-Elites loop: novelty + speciation + viability gate. Live archive metrics.",
        fields=[Field("seed_creature", "enum", "quadruped", options=SEEDS, label="creature"),
                Field("niche", "enum", "locomotion", options=NICHES),
                Field("iterations", "int", 30, min=5, max=500, step=5),
                Field("bins", "int", 10, min=6, max=20),
                Field("novelty", "bool", True, label="novelty (NSLC)"),
                Field("speciation", "bool", False),
                Field("gate", "bool", True, label="pre-sim gate"),
                Field("macro_rate", "float", 0.5, min=0.0, max=1.0, step=0.1),
                Field("fine_tune", "int", 500, min=128, max=8000, step=128, label="fine-tune steps"),
                Field("seed", "int", 0, min=0, max=9999)])
def _run_qd(p, ctx):
    from personal_cambrian.evo import run_qd, QDConfig
    cfg = QDConfig(axes=["aspect", "limb_count"], bins=p["bins"], niche=p["niche"],
                   iterations=p["iterations"], seed_steps=2000, fine_tune_steps=p["fine_tune"],
                   n_envs=2, n_steps=128, ep_steps=60, hidden=32, macro_rate=p["macro_rate"],
                   novelty=p["novelty"], speciation=p["speciation"], gate=p["gate"],
                   seed=p["seed"])

    def log(rec):
        m = {"iter": rec["iter"], "coverage": round(rec["coverage"], 4),
             "qd_score": round(rec["qd_score"], 3), "cells": rec["cells"],
             "rejected": rec["rejected"]}
        if rec.get("novelty") is not None:
            m["novelty"] = round(rec["novelty"], 4)
        if "species" in rec:
            m["species"] = rec["species"]
        if "bc_spread" in rec:
            m["bc_spread"] = round(rec["bc_spread"], 4)
        ctx.metric(**m)
        ctx.check_stop()

    ctx.status("evolving")
    archive, phylo = run_qd([_seed(p["seed_creature"])], cfg, log_fn=log)
    arts = _qd_artifacts(ctx, archive, phylo, "qd")
    return {"coverage": round(archive.coverage, 4), "qd_score": round(archive.qd_score, 3),
            "cells": len(archive.grid), "creatures": len(phylo.nodes), "artifacts": arts}


@action(id="evo.deep_time", label="Deep-time run", category="Evolution", kind="stream",
        streams=["qd_score", "coverage"], danger=True,
        description="The headline run: novelty + speciation; reports non-human lineages + innovation timeline.",
        fields=[Field("seed_creature", "enum", "quadruped", options=SEEDS, label="creature"),
                Field("niche", "enum", "locomotion", options=NICHES),
                Field("iterations", "int", 60, min=10, max=800, step=10),
                Field("divergence", "float", 0.15, min=0.05, max=0.6, step=0.05, label="non-human cutoff"),
                Field("seed", "int", 0, min=0, max=9999)])
def _deep_time(p, ctx):
    from personal_cambrian.evo import run_qd, QDConfig, deep_time_report
    cfg = QDConfig(axes=["aspect", "limb_count"], bins=10, niche=p["niche"],
                   iterations=p["iterations"], seed_steps=2000, fine_tune_steps=500,
                   n_envs=2, n_steps=128, ep_steps=60, hidden=32, macro_rate=0.6,
                   novelty=True, speciation=True, seed=p["seed"])

    def log(rec):
        ctx.metric(iter=rec["iter"], coverage=round(rec["coverage"], 4),
                   qd_score=round(rec["qd_score"], 3))
        ctx.check_stop()

    ctx.status("deep time")
    archive, phylo = run_qd([_seed(p["seed_creature"])], cfg, log_fn=log)
    rep = deep_time_report(phylo, divergence_threshold=p["divergence"], min_depth=5)
    arts = _qd_artifacts(ctx, archive, phylo, "deeptime")
    return {"tree": rep["tree"], "non_human_lineages": rep["n_non_human"],
            "most_divergent": rep["most_divergent"], "n_innovation_types": rep["n_innovation_types"],
            "innovation_timeline": rep["innovation_timeline"], "done_when": rep["done_when"],
            "met": rep["met"], "artifacts": arts}


@action(id="evo.poet", label="POET curriculum", category="Evolution", kind="stream",
        streams=["max_difficulty", "cum_transfers"],
        description="Co-evolve niches + agents; watch the env set turn over with rising difficulty.",
        fields=[Field("seed_creature", "enum", "quadruped", options=SEEDS, label="creature"),
                Field("niche", "enum", "terrain", options=NICHES),
                Field("iterations", "int", 20, min=5, max=200, step=5),
                Field("train", "int", 400, min=128, max=4000, step=128, label="train steps/opt")])
def _poet(p, ctx):
    import numpy as np
    from personal_cambrian.evo import make_creature_poet, POETConfig
    from personal_cambrian.sim.tasks import NICHES as N
    cfg = POETConfig(max_active=3, reproduce_every=2, transfer_every=2, n_children=3,
                     optimize_steps=1, mc_low=-5.0, mc_high=30.0, repro_threshold=0.5)
    poet = make_creature_poet(_seed(p["seed_creature"]), cfg, train_steps=p["train"],
                              ep_steps=80, hidden=32, n_envs=2, n_steps=64,
                              rng=np.random.default_rng(0))
    poet.add_env(N[p["niche"]](difficulty=0.0, max_steps=80), None)

    def log(rec):
        ctx.metric(t=rec["t"], n_active=rec["n_active"],
                   max_difficulty=round(rec["max_difficulty"], 3),
                   cum_transfers=rec["cum_transfers"], cum_added=rec["cum_added"])
        ctx.check_stop()

    ctx.status("POET")
    poet.run(p["iterations"], log_fn=log)
    return {"frontier_difficulty": round(poet.history[-1]["max_difficulty"], 2),
            "created": poet.n_added, "retired": poet.n_removed,
            "transfers": len(poet.transfers)}


@action(id="evo.surrogate", label="Surrogate-assisted QD", category="Evolution", kind="value",
        description="Ablation: surrogate vs no-surrogate sims-to-QD-score (hardware-independent).",
        fields=[Field("sim_budget", "int", 300, min=100, max=1000, step=50),
                Field("seed", "int", 0, min=0, max=9999)])
def _surrogate(p, ctx):
    from personal_cambrian.evo import surrogate_sims_benchmark
    ctx.status("running ablation")
    return surrogate_sims_benchmark(sim_budget=p["sim_budget"], seed=p["seed"])


# ============================ VISUALIZATION ===============================
@action(id="viz.render_creature", label="Render creature (Blender)", category="Visualize",
        kind="render",
        description="Build the creature in Blender headless and render a PNG.",
        fields=[Field("seed_creature", "enum", "quadruped", options=SEEDS, label="creature")])
def _viz_creature(p, ctx):
    return _run_script(ctx, ["scripts/build_creature_blender.py",
                             "--seed-creature", p["seed_creature"]])


@action(id="viz.replay", label="Locomotion replay (video)", category="Visualize", kind="render",
        description="Record a rollout and render a Blender replay video.",
        fields=[Field("seed_creature", "enum", "quadruped", options=SEEDS, label="creature"),
                Field("niche", "enum", "locomotion", options=NICHES),
                Field("steps", "int", 60, min=20, max=200, step=10)])
def _viz_replay(p, ctx):
    return _run_script(ctx, ["scripts/replay_creature.py", "--seed-creature", p["seed_creature"],
                             "--niche", p["niche"], "--steps", str(p["steps"])])


@action(id="viz.biomechanics", label="Biomechanics clip (video)", category="Visualize",
        kind="render",
        description="Force-vector + moment-arm overlay clip.",
        fields=[Field("seed_creature", "enum", "agent_zero", options=SEEDS, label="creature"),
                Field("steps", "int", 50, min=20, max=200, step=10)])
def _viz_bio(p, ctx):
    return _run_script(ctx, ["scripts/render_biomechanics.py", "--seed-creature",
                             p["seed_creature"], "--steps", str(p["steps"])])


@action(id="viz.full_report", label="Full visual report", category="Visualize", kind="render",
        danger=True,
        description="One command: evolve, then render phylogeny + creature + replay + biomechanics + lineage.",
        fields=[Field("seed_creature", "enum", "quadruped", options=SEEDS, label="creature"),
                Field("iterations", "int", 18, min=8, max=80, step=2),
                Field("steps", "int", 40, min=20, max=120, step=10)])
def _viz_report(p, ctx):
    return _run_script(ctx, ["scripts/render_run.py", "--seed-creature", p["seed_creature"],
                             "--iterations", str(p["iterations"]), "--steps", str(p["steps"])])


# ============================ GPU / SCALE =================================
@action(id="gpu.mjx_benchmark", label="MJX parity + throughput", category="GPU / Scale",
        kind="render",
        description="MJX vs CPU dynamics parity and physics-steps/sec (GPU-bound; honest CPU numbers).",
        fields=[Field("substeps", "int", 5, min=1, max=10)])
def _mjx(p, ctx):
    return _run_script(ctx, ["scripts/mjx_benchmark.py", "--substeps", str(p["substeps"])])


@action(id="gpu.stage9_report", label="Stage-9 benchmark report", category="GPU / Scale",
        kind="render",
        description="Collate surrogate sims-saved + JAX-archive parity + MJX throughput into runs/.",
        fields=[Field("sim_budget", "int", 400, min=100, max=1000, step=50)])
def _stage9(p, ctx):
    return _run_script(ctx, ["scripts/stage9_benchmarks.py", "--sim-budget", str(p["sim_budget"])])


# ============================ REALITY ANCHOR =============================
def _anchor_metrics():
    from personal_cambrian.anchor.whoop import load_default as lw
    from personal_cambrian.anchor.lifting import load_default as ll
    ms = lw()
    for _n, m in ll().known().items():
        ms.add(m)
    return ms


@action(id="anchor.ingest", label="Ingest Whoop + lifting", category="Reality Anchor", kind="value",
        danger=True,
        description="Parse local health/strength exports into provenance-tagged metrics (local only).",
        fields=[])
def _anchor_ingest(p, ctx):
    ms = _anchor_metrics()
    rows = []
    for name in sorted(ms.known()):
        m = ms.get(name)
        rows.append({"name": name, "value": round(m.value, 2), "unit": m.unit,
                     "status": m.status, "n": m.n, "source": m.source.split(":")[0],
                     "ci": [round(c, 1) for c in m.ci] if m.ci else None})
    return {"n_measured": len(rows), "metrics": rows,
            "note": "local-only; unmeasured quantities stay 'unknown' (never invented)"}


@action(id="anchor.anchored_seed", label="Anchor the seed", category="Reality Anchor", kind="value",
        danger=True,
        description="Scale the Agent-Zero seed to measured height/mass; build priors (measured vs prior).",
        fields=[])
def _anchor_seed(p, ctx):
    from personal_cambrian.anchor import anchored_seed, standing_height
    from personal_cambrian.morphogenesis import develop
    ms = _anchor_metrics()
    seed = _seed("agent_zero")
    anchored, priors = anchored_seed(ms, seed)
    return {"height_before_m": round(standing_height(seed), 2),
            "height_after_m": round(standing_height(anchored), 2),
            "mass_before_kg": round(develop(seed).total_mass(), 1),
            "mass_after_kg": round(develop(anchored).total_mass(), 1),
            "priors": {k: {"value": round(v.value, 1), "status": v.status,
                           "unit": v.unit} for k, v in priors.items()}}


@action(id="anchor.calibrate", label="Near-human calibration", category="Reality Anchor", kind="value",
        danger=True,
        description="Kalman forecast of a lift's 1RM vs held-out, clamped to human-achievable.",
        fields=[Field("lift", "enum", "bench_press_1rm",
                      options=["bench_press_1rm", "squat_1rm", "overhead_press_1rm"]),
                Field("holdout", "int", 8, min=3, max=20)])
def _anchor_calibrate(p, ctx):
    from personal_cambrian.anchor.calibrate import lift_progression, calibrate_and_validate
    from personal_cambrian.anchor.lifting import DEFAULT_LIFTING_PATH
    if not os.path.exists(DEFAULT_LIFTING_PATH):
        return {"error": "no local lifting export found"}
    prog = lift_progression(DEFAULT_LIFTING_PATH, p["lift"])
    if len(prog) <= p["holdout"] + 2:
        return {"error": "not enough sessions logged for this lift"}
    c = calibrate_and_validate(prog, holdout=p["holdout"], param=p["lift"], obs_var=120.0)
    return {"lift": p["lift"], "sessions": len(prog),
            "within_ci_fraction": round(c.within_ci_fraction, 2),
            "predictions": [round(x, 1) for x in c.predictions],
            "observed": [round(x, 1) for x in c.observed], "human_bounds": c.bounds}


def register_all():
    """Import side-effect: all actions above are registered at module import."""
    return len(__import__("dashboard.registry", fromlist=["ACTIONS"]).ACTIONS)
