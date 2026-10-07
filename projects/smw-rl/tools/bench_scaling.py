"""Measure how training throughput scales with worker count and rollout length.

The emulator workers spend most of each vectorised step blocked on a sync
barrier, so the useful question is not "how many cores are there" but "how many
envs does it take to hide the per-step overhead".

    python tools/bench_scaling.py --envs 12,16,24,32
"""

from __future__ import annotations

import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import psutil  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from smwrl.env import make_vec_env  # noqa: E402
from smwrl.wrappers import EpisodeConfig, RewardConfig  # noqa: E402


def measure(n_envs: int, n_steps: int, batch_size: int, rollouts: int, device: str):
    venv = make_vec_env(
        "YoshiIsland1",
        n_envs=n_envs,
        reward_cfg=RewardConfig(),
        episode_cfg=EpisodeConfig(),
    )
    model = PPO(
        "CnnPolicy", venv, n_steps=n_steps, batch_size=batch_size,
        n_epochs=4, device=device, verbose=0,
    )
    proc = psutil.Process()

    model.learn(total_timesteps=n_envs * n_steps)          # warm up (JIT, alloc)
    proc.cpu_percent(None)
    for c in proc.children(recursive=True):
        try:
            c.cpu_percent(None)
        except psutil.Error:
            pass

    target = n_envs * n_steps * rollouts
    t0 = time.time()
    model.learn(total_timesteps=target, reset_num_timesteps=False)
    dt = time.time() - t0

    cpu = proc.cpu_percent(None)
    rss = proc.memory_info().rss
    for c in proc.children(recursive=True):
        try:
            cpu += c.cpu_percent(None)
            rss += c.memory_info().rss
        except psutil.Error:
            pass

    venv.close()
    del model
    return target / dt, cpu, rss / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="12,16,24,32")
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    ncpu = psutil.cpu_count()
    print(f"{args.tag}device={args.device}  cores={ncpu}  n_steps={args.n_steps}")
    print(f"{'envs':>5} {'steps/s':>9} {'per-env':>8} {'vs 12':>7} {'CPU%':>7} "
          f"{'of max':>7} {'RSS GB':>7}  {'h per 5M':>9}")
    base = None
    for n in [int(x) for x in args.envs.split(",")]:
        try:
            sps, cpu, rss = measure(n, args.n_steps, args.batch_size, args.rollouts, args.device)
        except Exception as e:  # OOM or spawn failure at high env counts
            print(f"{n:>5}  FAILED: {type(e).__name__}: {str(e)[:60]}")
            continue
        base = base or sps
        print(f"{args.tag}{n:>5} {sps:9.0f} {sps / n:8.1f} {sps / base:6.2f}x {cpu:7.0f} "
              f"{cpu / (ncpu * 100):6.0%} {rss:7.1f}  {5e6 / sps / 3600:8.2f}h")


if __name__ == "__main__":
    main()
