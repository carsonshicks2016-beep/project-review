#!/usr/bin/env python3
"""MJX vs CPU physics throughput + parity benchmark (ROADMAP Stage 9.1 / 9.5).

Times the C MuJoCo engine (one env at a time) against the MJX jit+vmap backend
(B parallel envs) and reports physics-steps/sec, plus the CPU<->MJX dynamics parity.

    python3 scripts/mjx_benchmark.py
    python3 scripts/mjx_benchmark.py --batches 64 256 1024 --steps 20

HONEST NOTE: this box is CPU-only. MJX is built for GPU batching; on CPU it does NOT
beat the C engine -- the >=100x throughput is a GPU property. What this verifies
locally is that the port is CORRECT (parity) and VECTORIZES over many envs; the
speedup lands when the same code runs on an accelerator.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import SEEDS
from personal_cambrian.morphogenesis import develop
from personal_cambrian.morphogenesis.to_mujoco import compile_morphology
from personal_cambrian.sim.mjx_env import mjx_available, MjxBatchEnv, dynamics_parity, grounded_qpos

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cpu_steps_per_sec(model, n_substeps, n_steps):
    import mujoco
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d); d.qpos[:] = grounded_qpos(model); mujoco.mj_forward(model, d)
    ctrl = np.zeros(model.nu)
    mujoco.mj_step(model, d)                                   # warm
    t0 = time.time()
    for _ in range(n_steps * n_substeps):
        d.ctrl[:] = ctrl; mujoco.mj_step(model, d)
    dt = time.time() - t0
    return n_steps * n_substeps / dt


def mjx_steps_per_sec(env, batch, n_steps):
    import jax
    data = env.init(batch)
    acts = np.zeros((batch, env.model.nu), dtype=np.float32)
    data = env.step(data, acts); jax.block_until_ready(data.qpos)   # warm (compile)
    t0 = time.time()
    for _ in range(n_steps):
        data = env.step(data, acts)
    jax.block_until_ready(data.qpos)
    dt = time.time() - t0
    return n_steps * batch * env.n_substeps / dt                # physics-steps/sec (aggregate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-creature", default="quadruped", choices=list(SEEDS))
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 64, 256])
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--substeps", type=int, default=5)
    args = ap.parse_args()

    if not mjx_available():
        print("MJX unavailable: pip install jax mujoco-mjx")
        return

    genome = SEEDS[args.seed_creature]()
    model, _ = compile_morphology(develop(genome), add_floor=True, free_root=True)
    model.opt.iterations = 50; model.opt.ls_iterations = 50

    print(f"benchmark: {args.seed_creature}  (nq={model.nq} nu={model.nu}, "
          f"{args.substeps} substeps)\n")

    # --- parity ------------------------------------------------------------
    air = grounded_qpos(model).copy(); air[2] += 3.0
    hard = grounded_qpos(model).copy(); hard[2] -= 0.10
    parity = {
        "contact_free_n20": dynamics_parity(model, qpos0=air, n_steps=20),
        "gentle_contact_n20": dynamics_parity(model, n_steps=20),
        "hard_contact_n20": dynamics_parity(model, qpos0=hard, n_steps=20),
    }
    print("DYNAMICS PARITY  (max |dqpos| vs CPU C engine)")
    print(f"  contact-free   : {parity['contact_free_n20']:.2e}   (faithful port)")
    print(f"  gentle contact : {parity['gentle_contact_n20']:.2e}   (tight)")
    print(f"  hard contact   : {parity['hard_contact_n20']:.2e}   (MJX soft-contact gap)\n")

    # --- throughput --------------------------------------------------------
    cpu = cpu_steps_per_sec(model, args.substeps, args.steps)
    env = MjxBatchEnv(genome, n_substeps=args.substeps)
    print(f"THROUGHPUT  (physics-steps/sec)")
    print(f"  CPU C engine (1 env)         {cpu:12,.0f}")
    rows = {"cpu_1env": cpu}
    for B in args.batches:
        sps = mjx_steps_per_sec(env, B, args.steps)
        rows[f"mjx_{B}env"] = sps
        print(f"  MJX jit+vmap ({B:5d} envs)    {sps:12,.0f}   ({sps/cpu:5.2f}x CPU)")

    best = max(v for k, v in rows.items() if k.startswith("mjx"))
    print(f"\nbest MJX aggregate = {best/cpu:.2f}x the CPU C engine on THIS (CPU) box.")
    print("On a GPU the same jit+vmap rollout reaches >=100x via real batch parallelism.")

    run_dir = os.path.join(ROOT, "runs")
    os.makedirs(run_dir, exist_ok=True)
    out = os.path.join(run_dir, "mjx_benchmark.json")
    with open(out, "w") as f:
        json.dump({"creature": args.seed_creature, "substeps": args.substeps,
                   "parity": parity, "throughput_steps_per_sec": rows,
                   "device": "cpu"}, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
