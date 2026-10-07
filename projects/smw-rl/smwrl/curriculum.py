"""Reverse curriculum: learn the end of a level before the start.

An SMW level is ~4,000-5,000 pixels long. From a standing start the agent only
ever sees the first few hundred pixels, so the reward signal for anything later
is effectively unreachable and training plateaus.

The fix is to seed some episodes from save states harvested part-way through the
level. The agent practises the final stretch (a short, easy problem), then the
stretch before it, and the skill chains backwards toward the real start.

    python -m smwrl.curriculum --level YoshiIsland1 --episodes 40

Writes checkpoints/<level>/harvest.pkl.  These are unproven policy-rollout
snapshots for diagnosis/exploration seeds; training deliberately accepts only
the clear-proven curriculum exported by ``smwrl.explore``.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from smwrl.env import make_env
from smwrl.levels import LEVELS
from smwrl.policy import load_policy_runtime
from smwrl.wrappers import EpisodeConfig, RewardConfig

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"

# Each save state is ~430 KB and every worker process gets a copy, so this is
# deliberately small: 24 states is ~10 MB per worker.
MAX_STATES = 24


def harvest(level: str, episodes: int, every: int, deterministic: bool,
            scripted: bool = False) -> list[bytes]:
    """Run the current policy and keep save states from its deepest runs.

    Early in training the scripted run-right agent gets further than the policy
    does, so it makes the better bootstrap harvester -- see --scripted.
    """
    runtime = None
    path = CKPT / level / "latest.zip"
    if scripted:
        print("harvesting with the scripted run-right agent (--scripted)")
    elif path.exists():
        runtime = load_policy_runtime(path, expected_level=level)
        print(f"harvesting with policy {path}")
    else:
        print("no policy yet -- harvesting with a scripted run-right agent")

    env = make_env(level, RewardConfig(), EpisodeConfig(max_steps=1500), monitor=False,
                   obs_cfg=runtime.obs_config if runtime is not None else None)
    found: list[tuple[int, bytes]] = []

    for ep in range(episodes):
        obs, _ = env.reset()
        stack = (runtime.initial_stack(obs) if runtime is not None
                 else np.repeat(obs[None], 4, axis=0))
        step = 0
        while True:
            if runtime is None:
                action = 4 if (step % 10) < 3 else 2
            else:
                batch = runtime.batch(stack)
                a, _ = runtime.model.predict(batch, deterministic=deterministic)
                action = int(np.asarray(a).flat[0])

            obs, _, term, trunc, info = env.step(action)
            stack = (runtime.advance(stack, obs) if runtime is not None
                     else np.concatenate([stack[1:], obs[None]], axis=0))
            step += 1

            st = env.last_state
            if st is not None and st.in_level and step % every == 0:
                found.append((st.x, env.unwrapped.em.get_state()))

            if term or trunc:
                print(f"  ep {ep + 1:>3}/{episodes}  max_x={info.get('max_x', 0):5}  "
                      f"{'CLEARED' if info.get('level_cleared') else ''}")
                break

    env.close()

    # Keep a spread across the level rather than only the deepest states: the
    # curriculum needs rungs all the way down, not just the top one.
    found.sort(key=lambda kv: kv[0])
    if len(found) > MAX_STATES:
        idx = np.linspace(0, len(found) - 1, MAX_STATES).astype(int)
        found = [found[i] for i in idx]
    print(f"kept {len(found)} states spanning x={found[0][0]}..{found[-1][0]}"
          if found else "no states harvested")
    return [blob for _, blob in found]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="YoshiIsland1", choices=sorted(LEVELS))
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--every", type=int, default=25, help="agent steps between snapshots")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--scripted", action="store_true",
                    help="ignore any trained policy and harvest with the scripted agent")
    args = ap.parse_args()

    states = harvest(args.level, args.episodes, args.every, not args.stochastic,
                     scripted=args.scripted)
    if not states:
        return
    out = CKPT / args.level
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "harvest.pkl"
    dest.write_bytes(pickle.dumps(states))
    print(f"wrote unproven harvest {dest} "
          f"({dest.stat().st_size / 1e6:.1f} MB, {len(states)} states); "
          "this does not replace a verified Go-Explore curriculum")


if __name__ == "__main__":
    main()
