from __future__ import annotations

import argparse

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401
from crypt_heist.env import CryptHeistParallelEnv, scripted_random_actions


def main():
    parser = argparse.ArgumentParser(description="Training smoke runner for the PettingZoo-style env.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--cycles", type=int, default=120)
    args = parser.parse_args()
    env = CryptHeistParallelEnv(seed=args.seed, max_cycles=args.cycles)
    obs, _ = env.reset(seed=args.seed)
    total = {agent: 0.0 for agent in env.agents}
    for _ in range(args.cycles):
        actions = scripted_random_actions(env)
        obs, rewards, terms, truncs, infos = env.step(actions)
        for agent, reward in rewards.items():
            total[agent] = total.get(agent, 0.0) + reward
        if not env.agents:
            break
    print("training smoke complete")
    for agent in sorted(total):
        print(f"{agent}: {total[agent]:.3f}")


if __name__ == "__main__":
    main()
