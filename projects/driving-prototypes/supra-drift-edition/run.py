#!/usr/bin/env python3
"""
Supra-AI entry point.

  python run.py                 # watch a population evolve (live, with sound)
  python run.py --train 300     # fast headless evolution for 300 generations
  python run.py --watch         # load best_supra.npz and watch it drive
  python run.py --ppo 400       # train a PPO reinforcement-learning policy
  python run.py --watch-ppo     # watch the trained PPO policy drive
  python run.py --seed 42       # fixed track + RNG for reproducibility
"""

import argparse
import math
from supra.config import Config
from supra.app import App, train_headless


def main():
    ap = argparse.ArgumentParser(description="Supra-AI driving sim")
    ap.add_argument("--train", type=int, metavar="GENERATIONS",
                    help="run headless evolution for N generations, then save")
    ap.add_argument("--watch", action="store_true",
                    help="load the saved best (GA) brain and watch it drive")
    ap.add_argument("--ppo", type=int, metavar="ITERATIONS",
                    help="train a PPO reinforcement-learning policy (headless), then save")
    ap.add_argument("--ppo-live", action="store_true", dest="ppo_live",
                    help="train PPO and watch it learn live in ONE window")
    ap.add_argument("--watch-ppo", action="store_true", dest="watch_ppo",
                    help="watch the trained PPO policy drive")
    ap.add_argument("--drift", type=int, metavar="ITERATIONS",
                    help="train a DRIFT-objective policy (separate drift lineage)")
    ap.add_argument("--watch-drift", action="store_true", dest="watch_drift",
                    help="watch the trained drift policy drive")
    ap.add_argument("--reload", action="store_true",
                    help="with --watch-ppo: hot-reload the policy as it trains (live training)")
    ap.add_argument("--fresh", action="store_true",
                    help="start PPO from scratch instead of resuming the saved policy")
    ap.add_argument("--3d", action="store_true", dest="use_3d",
                    help="launch the 3D OpenGL viewer instead of top-down 2D")
    ap.add_argument("--touge", action="store_true",
                    help="use a touge (mountain pass) track")
    ap.add_argument("--seed", type=int, default=None, help="fix the track + RNG seed")
    ap.add_argument("--save", default="best_supra.npz", help="GA brain save/load path")
    ap.add_argument("--ppo-save", default="ppo_supra.pt", help="PPO policy save/load path")
    ap.add_argument("--drift-save", default="drift_supra.pt", help="drift policy save/load path")
    ap.add_argument("--pop", type=int, help="override population size")
    ap.add_argument("--workers", type=int, default=None,
                    help="PPO rollout worker processes (default: auto = cpu-1)")
    ap.add_argument("--n-envs", type=int, default=None, dest="n_envs",
                    help="override PPO parallel env count (default from config)")
    ap.add_argument("--recurrent", action="store_true",
                    help="use a recurrent (LSTM) policy for --ppo/--drift "
                         "(use a separate --ppo-save/--drift-save file)")
    ap.add_argument("--car", choices=["supra", "rx7", "skyline"], default="supra",
                    help="which chassis to drive/train (pass the same one when watching)")
    args = ap.parse_args()

    cfg = Config()
    from supra.config import car_spec
    cfg.car = car_spec(args.car)
    if args.car != "supra":
        print(f"car: {cfg.car.name} ({cfg.car.engine})")
    if args.pop:
        cfg.evo.population = args.pop
    if args.n_envs:
        cfg.ppo.n_envs = args.n_envs
    if args.recurrent:
        cfg.ppo.recurrent = True
    # the drift lineage runs more steering lock + quicker hands; apply for both
    # training (--drift) and watching (--watch-drift) so obs/physics stay matched
    if args.drift or args.watch_drift:
        cfg.car.max_steer_angle = math.radians(cfg.ppo.drift_steer_lock_deg)
        cfg.car.steer_rate = math.radians(cfg.ppo.drift_steer_rate_deg)

    if args.train:
        train_headless(cfg, args.train, seed=args.seed, save_path=args.save)
        return
    if args.ppo:
        from supra.ppo import train_ppo
        train_ppo(cfg, iterations=args.ppo, seed=args.seed or 0,
                  save_path=args.ppo_save, resume=not args.fresh,
                  n_workers=args.workers)
        return
    if args.drift:
        from supra.ppo import train_ppo
        train_ppo(cfg, iterations=args.drift, seed=args.seed or 0,
                  save_path=args.drift_save, resume=not args.fresh,
                  n_workers=args.workers, objective="drift")
        return
    if args.ppo_live:
        from supra.ppo import watch_ppo_live
        if args.touge:
            cfg.track = cfg.track.__class__(half_width=cfg.track.touge_half_width)
        watch_ppo_live(cfg, seed=args.seed or 0, save_path=args.ppo_save,
                       resume=not args.fresh)
        return

    app = App(cfg, seed=args.seed, save_path=args.save)
    watch_path = (args.ppo_save if args.watch_ppo else
                  args.drift_save if args.watch_drift else None)
    if watch_path is not None:
        import torch
        from supra.ppo import TorchBrain
        from supra.ppo_env import obs_dim
        kind = "drift" if args.watch_drift else "race"
        train_tip = "--drift 1500" if args.watch_drift else "--ppo 1500"
        expected_obs = obs_dim(cfg, kind)        # drift watch expects the 30-d obs
        try:
            ckpt = torch.load(watch_path, map_location="cpu", weights_only=False)
        except FileNotFoundError:
            print(f"no policy at {watch_path}. Train one first: python3 run.py {train_tip}")
            return
        if ckpt.get("n_obs") != expected_obs:
            print(f"{watch_path} expects {ckpt.get('n_obs')} inputs but the current "
                  f"config needs {expected_obs} (observation layout changed). "
                  f"Retrain: python3 run.py {train_tip}")
            return
        brains = [TorchBrain(watch_path, stochastic=(i > 0))
                  for i in range(cfg.evo.population)]
        app.sim.set_population(brains, app.evo.colors())
        app.evolve_enabled = False
        if args.reload:
            app.reload_ppo_path = watch_path
            print(f"watching {watch_path} ({kind}) — LIVE: hot-reloading as it trains")
        else:
            print(f"loaded {watch_path} — watching the {kind} policy (car 1 = deterministic)")
    elif args.watch:
        from supra.brain import MLPGenome
        genome = MLPGenome.load(args.save)
        if genome.sizes[0] != cfg.sensors.n_inputs:
            print(f"{args.save} uses {genome.sizes[0]} inputs but the current "
                  f"config has {cfg.sensors.n_inputs}. Retrain: python3 run.py --train 300")
            return
        app.evo.seed_from(genome)
        app.sim.set_population(app.evo.brains, app.evo.colors())
        print(f"loaded {args.save} — watching the champion (+ mutated rivals)")
    if args.touge:
        app.sim.new_track(kind='touge')
        app.sim.reset_episode()
    if args.use_3d:
        from supra.app3d import run_3d
        run_3d(app)
    else:
        app.run()


if __name__ == "__main__":
    main()
