#!/usr/bin/env python3
"""End-to-end demo of the PERSONAL CAMBRIAN cheap prototype.

Runs MAP-Elites over the 14-gene morphology + training-program genome on the
4 analytic biomes, then prints: Agent Zero's profile, archive coverage, the best
descendant in each realism band, a morphology-vs-control attribution, and a text
heatmap of the athletic-descendant archive.

    python3 scripts/run_prototype.py            # default 4000 iterations, HA
    python3 scripts/run_prototype.py --open     # allow open-evolution genes
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.evolve import EvolveConfig, run, agent_zero
from personal_cambrian.fitness import Baseline, evaluate_agent
from personal_cambrian.agent import Agent
from personal_cambrian.metrics import AgentZeroProfile


def describe(elite, baseline) -> str:
    geno, prog = elite.genome
    ev = elite.eval
    f = prog.focus()
    top_focus = max(f, key=f.get)
    lines = [
        f"    fitness={elite.fitness:.2f}  realism={['HUMAN-ACHIEVABLE','HUMAN-POSSIBLE','OPEN-EVOLUTION'][elite.realism]}",
        f"    performance(vs you)={ev.performance:.2f}x  injury={ev.injury_risk:.2f}  "
        f"recovery_debt={ev.recovery_debt:.2f}  budget_overflow={ev.budget_overflow:.2f}",
        f"    program: focus={top_focus}  vol={prog.volume:.2f} int={prog.intensity:.2f}",
        "    biomes: " + "  ".join(
            f"{n}={r.score:.2f}{r.unit.split('(')[0]}" for n, r in ev.biomes.items()),
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=4000)
    ap.add_argument("--resolution", type=int, default=24)
    ap.add_argument("--open", action="store_true", help="allow OPEN-EVOLUTION genes")
    ap.add_argument("--possible", action="store_true", help="allow HUMAN-POSSIBLE genes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    level = "op" if args.open else ("hp" if args.possible else "ha")

    print("=" * 72)
    print("PERSONAL CAMBRIAN — cheap prototype")
    print("=" * 72)

    # --- Agent Zero profile (priors only; nothing measured yet) ------------
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prof_path = os.path.join(here, "data", "agent_zero.example.json")
    if os.path.exists(prof_path):
        prof = AgentZeroProfile.load(prof_path)
        print("\nAGENT ZERO (probabilistic twin — supply real metrics to narrow CIs):")
        print(prof.summary())
        print(f"\n  ** {prof.known_fraction():.0%} of metrics are MEASURED. "
              f"All baselines below are PRIORS, not facts. **")

    # --- run evolution ------------------------------------------------------
    cfg = EvolveConfig(iterations=args.iterations, resolution=args.resolution,
                       realism=level, seed=args.seed)
    print(f"\nRunning MAP-Elites: realism={level} iters={cfg.iterations} "
          f"grid={cfg.resolution}x{cfg.resolution} seed={cfg.seed}")
    archive, baseline = run(cfg)

    print(f"\nARCHIVE: coverage={archive.coverage:.1%}  cells={len(archive.grid)}  "
          f"qd_score={archive.qd_score:.1f}  evals={archive.n_evals}")

    # --- best descendant in each realism band ------------------------------
    for r, name in enumerate(["HUMAN-ACHIEVABLE", "HUMAN-POSSIBLE", "OPEN-EVOLUTION"]):
        e = archive.best_in_realism(r)
        if e:
            print(f"\nBEST {name}:")
            print(describe(e, baseline))

    # --- morphology vs control attribution ---------------------------------
    best = archive.best()
    if best:
        geno, prog = best.genome
        full = Agent(geno, prog)
        naive = full.with_naive_controller()
        full_ev = evaluate_agent(full, baseline)
        naive_ev = evaluate_agent(naive, baseline)
        ctrl_gain = full_ev.performance - naive_ev.performance
        print("\nATTRIBUTION (best overall): "
              f"structure+state={naive_ev.performance:.2f}x  "
              f"+control={ctrl_gain:+.2f}x  total={full_ev.performance:.2f}x")

    # --- archive heatmap ----------------------------------------------------
    print("\nATHLETIC-DESCENDANT MAP  (x -> strength/mass,  y -> max velocity)")
    print("  fitness:  low '" + " .:-=+*#%@"[0] + "' ... high '@'")
    for line in archive.ascii_map().splitlines():
        print("  " + line)
    print("\nDone. This is a SIMULATION over uncalibrated priors — not medical advice.")


if __name__ == "__main__":
    main()
