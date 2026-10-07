#!/usr/bin/env python3
"""Reality-anchor report (ROADMAP Stage 11 demonstration).

Ingests the local Whoop + lifting exports, builds the Agent-Zero seed's priors,
anchors the seed geometry to the measured height/weight, and runs a near-human Kalman
calibration on a lift's training block. Everything stays LOCAL and PROVENANCE-TAGGED;
unmeasured quantities remain explicit priors -- nothing is invented.

    python3 scripts/reality_anchor.py

The open-evolution / deep-time branch is untouched (Stage 11.5).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.seeds import agent_zero
from personal_cambrian.morphogenesis import develop
from personal_cambrian.anchor import (
    MetricSet, anchored_seed, standing_height, calibrate_and_validate,
)
from personal_cambrian.anchor.whoop import load_default as load_whoop
from personal_cambrian.anchor.lifting import load_default as load_lifting, DEFAULT_LIFTING_PATH
from personal_cambrian.anchor.calibrate import lift_progression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ms = load_whoop()
    for name, m in load_lifting().known().items():
        ms.add(m)
    if len(ms) == 0:
        print("no local Whoop/lifting exports found; nothing to anchor.")
        return

    print(f"ingested {len(ms.known())} measured metrics (provenance-tagged):")
    for name in sorted(ms.known()):
        m = ms.get(name)
        ci = f" CI[{m.ci[0]:.1f},{m.ci[1]:.1f}]" if m.ci else ""
        print(f"  {name:18} {m.value:8.2f} {m.unit:4} n={m.n:<4}{ci}  <- {m.source.split(':')[0]}")

    seed = agent_zero()
    anchored, priors = anchored_seed(ms, seed)
    print(f"\nseed anchored: height {standing_height(seed):.2f}->{standing_height(anchored):.2f} m, "
          f"mass {develop(seed).total_mass():.1f}->{develop(anchored).total_mass():.1f} kg")
    n_meas = sum(1 for p in priors.values() if p.status == "measured")
    print(f"priors: {n_meas} measured (narrow CI), {len(priors) - n_meas} population priors "
          f"(wide CI, unmeasured -- not invented)")

    if os.path.exists(DEFAULT_LIFTING_PATH):
        prog = lift_progression(DEFAULT_LIFTING_PATH, "bench_press_1rm")
        if len(prog) > 12:
            c = calibrate_and_validate(prog, holdout=8, param="bench_press_1rm", obs_var=120.0)
            print(f"\nnear-human calibration (bench 1RM, {len(prog)} sessions): "
                  f"held-out within-CI {c.within_ci_fraction:.0%}, "
                  f"predictions human-bounded {c.bounds}")

    out_dir = os.path.join(ROOT, "runs", "reality_anchor")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "anchored_seed.genome.json"), "w") as f:
        f.write(anchored.to_json())
    with open(os.path.join(out_dir, "priors.json"), "w") as f:
        json.dump({k: v.to_dict() for k, v in priors.items()}, f, indent=2)
    print(f"\nsaved anchored seed + priors -> {out_dir}/  (open-evolution branch untouched)")


if __name__ == "__main__":
    main()
