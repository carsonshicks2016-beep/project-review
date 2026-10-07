"""The MAP-Elites quality-diversity loop (ROADMAP Stage 5.3).

This is where every prior stage composes into open-ended evolution:

    seed the archive (Stage 5.2) with the founder creature(s)
    repeat:
        ask    -> sample an elite, mutate it                 (Stage 4)
        gate   -> reject non-viable bodies BEFORE sim/RL       (Stage 6.2-6.4)
        adapt  -> warm-start the controller from the parent,
                  short fine-tune                             (Stages 3.3/3.4/3.5)
        score  -> evaluate locomotion in the niche            (Stage 2)
        tell   -> insert by morphology descriptors            (Stages 5.1/5.2)
        record -> log the parent->child edge + innovations    (Stage 4.5)

Per-candidate cost is kept small by warm-starting from the parent and fine-tuning
for only a few thousand steps (the Stage-3.5 knee). Controllers are held in a side
map keyed by phylogeny node so the archive stays JSON-serializable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import mujoco

from ..encoding.genome import Genome
from ..encoding.mutate import mutate, MutationSchedule
from ..morphogenesis import develop, compute_budgets
from ..sim import CreatureEnv, LocomotionTask, NICHES
from ..control import (
    PPOConfig, evaluate, warm_started_make_agent, scratch_make_agent,
)
from ..control.ppo import train
from .archive import MAPElites
from .phylogeny import Phylogeny
from .descriptors import morphology_descriptors, distance_from
from .metrics import archive_novelty, fitness_grid, grid_to_json
from .supply import check_supply
from .viability import check_viability
from .structure import check_structure
from .speciation import sample_species_aware, count_species
from .behavior import characterize, bc_distance
from .novelty import NoveltyArchive, novelty_emitter, behavior_spread


def pre_sim_gate(genome: Genome, *, structural: bool = True):
    """Stage 6.2-6.4 viability gate, cheapest check first.

    Returns (ok, reasons). A non-viable genome must never reach develop/sim/RL.
    Order: supply (pure graph) -> geometric viability (compile) -> structural load.
    """
    s = check_supply(genome)
    if not s:
        return False, list(s.reasons)
    v = check_viability(genome)
    if not v:
        return False, list(v.reasons)
    if structural:
        st = check_structure(genome)
        if not st:
            return False, list(st.failures)
    return True, []


def budget_penalty(genome: Genome, weight: float) -> float:
    """Stage 6.1 biological-budget overflow as a non-negative fitness penalty."""
    if weight <= 0.0:
        return 0.0
    return weight * compute_budgets(develop(genome), genome).penalty()


@dataclass
class QDConfig:
    axes: list = field(default_factory=lambda: ["aspect", "limb_count"])
    bins: int = 12
    niche: str = "locomotion"         # selective regime from the Stage-7 NICHES gauntlet
    iterations: int = 100
    seed_steps: int = 50_000          # train the founder(s) well
    fine_tune_steps: int = 4_096      # cheap per-mutant adaptation (warm-started)
    n_envs: int = 4
    n_steps: int = 512
    ep_steps: int = 300
    hidden: int = 64
    macro_rate: float = 0.3
    micro_sigma: float = 0.1
    snapshot_every: int = 0           # 0 = off; else store a 2-D fitness grid every N iters
    gate: bool = True                 # Stage 6.2-6.4 reject-before-sim viability gate
    budget_penalty_weight: float = 1.0  # Stage 6.1 overflow -> fitness penalty
    speciation: bool = False          # Stage 7.3 species-aware emission + tracking
    species_threshold: float = 0.15   # morphological-distance compatibility cutoff
    novelty: bool = False             # Stage 8.2 NSLC novelty emitter + behavior archive
    novelty_k: int = 5                # k-NN for the novelty (sparseness) score
    novelty_weight: float = 1.0       # weight on behavioral novelty in NSLC selection
    local_competition_weight: float = 1.0  # weight on local competition in NSLC
    track_behavior: bool = False      # characterize + log behavior even with the emitter OFF
                                      # (so a novelty-off ablation has comparable BC coverage)
    behavior_radius: float = 0.05     # packing radius for the behavior-coverage metric
    seed: int = 0


def run_qd(seed_genomes, cfg: QDConfig = QDConfig(),
           log_fn: Optional[Callable[[dict], None]] = None, verbose: bool = False):
    """Run the QD loop. Returns (archive, phylogeny)."""
    rng = np.random.default_rng(cfg.seed)
    archive = MAPElites(cfg.axes, cfg.bins)
    phylo = Phylogeny()
    policies: dict = {}                              # phylo node id -> controller
    sched = MutationSchedule(macro_rate=cfg.macro_rate, micro_sigma=cfg.micro_sigma)
    seed0 = seed_genomes[0]
    stats = {"evals": 0, "rejected": 0, "unstable": 0}
    reject_reasons: dict = {}
    # Stage 8.2: a permanent record of explored behaviors drives the novelty emitter.
    nov_archive = NoveltyArchive(cfg.novelty_k, seed=cfg.seed) if cfg.novelty else None
    track_bc = cfg.novelty or cfg.track_behavior   # characterize behavior this run?
    # behavioral niches discovered across EVERY evaluated creature (incremental
    # greedy packing) -- a fair, elite-count-independent measure of BC-space reach.
    seen_reps: list = []

    # mutants often have degenerate geometry that destabilizes the integrator; the
    # env already terminates on NaN (-> low fitness), so just count the warnings
    # instead of spamming stderr. (A proper viability gate arrives in Stage 6.)
    mujoco.set_mju_user_warning(
        lambda msg: stats.__setitem__("unstable", stats["unstable"] + 1))

    niche_cls = NICHES[cfg.niche]

    def make_env_for(g):
        return lambda s: CreatureEnv(g, task=niche_cls(max_steps=cfg.ep_steps),
                                     obs_mode="structured")

    def evaluate_and_insert(genome, parent_policy, parent_node, mutations, generation, steps):
        # Stage 6.5: reject non-viable bodies BEFORE develop/sim/RL, log the reason.
        if cfg.gate:
            ok, reasons = pre_sim_gate(genome)
            if not ok:
                stats["rejected"] += 1
                for r in reasons:
                    reject_reasons[r] = reject_reasons.get(r, 0) + 1
                return None

        cme = make_env_for(genome)
        try:                                         # last-resort probe (raises if nu==0)
            env = cme(0)
        except Exception:                            # noqa: BLE001
            stats["rejected"] += 1
            reject_reasons["env_error"] = reject_reasons.get("env_error", 0) + 1
            return None
        control_dt = env.control_dt

        ma = (warm_started_make_agent(cme, parent_policy) if parent_policy is not None
              else scratch_make_agent(cme, cfg.hidden))
        policy, _ = train(cme, PPOConfig(total_timesteps=steps, n_envs=cfg.n_envs,
                                         n_steps=cfg.n_steps, hidden=cfg.hidden,
                                         seed=cfg.seed), make_agent=ma)
        perf = evaluate(policy, env, max_steps=cfg.ep_steps)
        # locomotion = forward progress, minus the Stage-6.1 biological-budget overflow
        fitness = perf["distance"] - budget_penalty(genome, cfg.budget_penalty_weight)

        # Stage 8.2: characterize WHAT it does (behavior) before releasing the env,
        # then record it so the novelty emitter can reward exploring new behaviors.
        meta = {}
        if track_bc:
            bc = characterize(env, policy, max_steps=cfg.ep_steps)
            meta["bc"] = bc.vector.tolist()
            if all(bc_distance(bc.vector, r) > cfg.behavior_radius for r in seen_reps):
                seen_reps.append(bc.vector.copy())       # a newly reached behavior
            if nov_archive is not None:
                nov_archive.add(bc, fitness)
        env.close()

        desc = morphology_descriptors(genome)
        if "morph_distance" in cfg.axes:
            desc["morph_distance"] = distance_from(genome, seed0)
        if "speed" in cfg.axes:
            desc["speed"] = perf["distance"] / max(perf["steps"] * control_dt, 1e-6)

        if parent_node is None:
            node = phylo.add_root(genome, fitness=fitness, generation=generation)
        else:
            node = phylo.add_offspring(parent_node, genome, mutations,
                                       fitness=fitness, generation=generation)
        policies[node] = policy
        archive.add(genome, fitness, desc, node=node, **meta)
        stats["evals"] += 1
        return node

    # --- seed -------------------------------------------------------------
    for sg in seed_genomes:
        evaluate_and_insert(sg, None, None, None, 0, cfg.seed_steps)

    # --- ask / tell loop --------------------------------------------------
    for it in range(cfg.iterations):
        # Stage 7.3: species-aware emission gives every body-plan species an equal
        # reproduction share, so a divergent lineage is not crowded out by a plan
        # that merely fills more cells. Off by default -> plain MAP-Elites sampling.
        if cfg.novelty:
            # Stage 8.2: bias parent choice toward novel + locally-competitive
            # behavior so the search keeps reaching into unexplored behavior space.
            elite = novelty_emitter(list(archive.grid.values()), nov_archive, rng,
                                    w_novelty=cfg.novelty_weight,
                                    w_compete=cfg.local_competition_weight)
        elif cfg.speciation:
            elite = sample_species_aware(list(archive.grid.values()), rng,
                                         cfg.species_threshold)
        else:
            elite = archive.sample(rng)
        if elite is None:
            break
        parent_node = elite.meta["node"]
        child, recs = mutate(elite.genome, rng, generation=it + 1, schedule=sched)
        evaluate_and_insert(child, policies.get(parent_node), parent_node, recs,
                            it + 1, cfg.fine_tune_steps)
        best = archive.best()
        rec = {"iter": it, "coverage": archive.coverage, "qd_score": archive.qd_score,
               "max_fitness": (best.fitness if best else float("nan")),
               "novelty": archive_novelty(archive),
               "cells": len(archive.grid), "rejected": stats["rejected"],
               "unstable": stats["unstable"], "reject_reasons": dict(reject_reasons)}
        if cfg.speciation:
            rec["species"] = count_species(
                [e.genome for e in archive.grid.values()], cfg.species_threshold)
        if track_bc:
            elite_bcs = [e.meta["bc"] for e in archive.grid.values() if "bc" in e.meta]
            rec["bc_cells"] = len(seen_reps)             # behavioral niches over all evals
            rec["bc_spread"] = behavior_spread(elite_bcs)  # diversity among survivors
            if nov_archive is not None:
                rec["bc_archive"] = len(nov_archive)
        if cfg.snapshot_every and (it + 1) % cfg.snapshot_every == 0:
            g = fitness_grid(archive)
            if g is not None:
                rec["snapshot"] = grid_to_json(g)
        if log_fn:
            log_fn(rec)
        if verbose and (it + 1) % max(1, cfg.iterations // 10) == 0:
            b = archive.best()
            print(f"  iter {it+1:4d}  cov={archive.coverage:5.1%}  cells={len(archive.grid):4d}"
                  f"  qd={archive.qd_score:7.2f}  best={b.fitness:+.2f}  rej={stats['rejected']}")

    return archive, phylo
