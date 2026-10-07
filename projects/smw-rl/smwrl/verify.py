"""Check that a curriculum is actually usable before training on it.

The backward curriculum starts at the stage nearest the goal and only moves
earlier once the agent can finish from where it is. So if the *last* stage is
not finishable, the whole thing is dead on arrival -- the stage counter just
pins at its maximum and every episode is wasted.

That is not hypothetical. YoshiIsland3's curriculum was ordered by
`room * 100000 + x`, which ranked a vertical side-room above the real route and
put the deepest stage on a vine above a pit. Random play died 12/12 from there
and training spent an hour going backwards.

    python -m smwrl.verify --level YoshiIsland1

Exits non-zero if the goal-side stages cannot be cleared, so it can gate a
pipeline.
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np

from smwrl.actions import N_ACTIONS
from smwrl.curriculum_store import load_verified_curriculum
from smwrl.env import make_env
from smwrl.levels import LEVELS
from smwrl.wrappers import EpisodeConfig, RewardConfig

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"


def state_of(entry) -> bytes:
    """Curricula are (room, x, state) now; older ones were bare state blobs."""
    return entry[2] if isinstance(entry, tuple) else entry


def pos_of(entry):
    return (entry[0], entry[1]) if isinstance(entry, tuple) else None


def inflate(blob: bytes) -> bytes:
    """Accept current compressed curricula and legacy raw snapshots."""
    try:
        return zlib.decompress(blob)
    except zlib.error:
        return blob


def check_stage(env, blob: bytes, episodes: int, rng, action_p, max_steps: int) -> dict:
    """Random rollouts from one curriculum state."""
    outcomes: dict[str, int] = {}
    for _ in range(episodes):
        env.reset()
        env.unwrapped.em.set_state(inflate(blob))
        env.step(0)
        info: dict = {}
        term = trunc = False
        steps = 0
        while steps < max_steps and not (term or trunc):
            action = int(rng.choice(N_ACTIONS, p=action_p))
            for _ in range(int(rng.integers(4, 17))):
                _, _, term, trunc, info = env.step(action)
                steps += 1
                if term or trunc:
                    break
        why = ("cleared" if info.get("level_cleared")
               else "death" if info.get("death")
               else "stuck" if info.get("stuck") else "timeout")
        outcomes[why] = outcomes.get(why, 0) + 1
    return outcomes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="YoshiIsland1", choices=sorted(LEVELS))
    ap.add_argument("--stages", type=int, default=4, help="how many goal-side stages to test")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=1200)
    args = ap.parse_args()

    from smwrl.explore import ACTION_P

    directory = CKPT / args.level
    path = directory / "curriculum.pkl"
    if not path.exists():
        raise SystemExit(f"no curriculum at {path}; run smwrl.explore first")
    try:
        states = load_verified_curriculum(directory / "archive.pkl", path)
    except ValueError as e:
        raise SystemExit(f"unverified curriculum: {e}") from e
    print(f"{args.level}: {len(states)} curriculum stages")

    env = make_env(args.level, RewardConfig(),
                   EpisodeConfig(max_steps=args.max_steps, noop_max=0), monitor=False)
    rng = np.random.default_rng(0)
    ok = bool(states)
    try:
        # Highest index = nearest the goal = where the curriculum begins.
        for idx in range(len(states) - 1, max(-1, len(states) - 1 - args.stages), -1):
            env.reset()
            env.unwrapped.em.set_state(inflate(state_of(states[idx])))
            env.step(0)
            st = env.last_state
            out = check_stage(env, state_of(states[idx]), args.episodes, rng,
                              ACTION_P, args.max_steps)
            clears = out.get("cleared", 0)
            flag = "OK " if clears else "   "
            p = pos_of(states[idx])
            where = f" (archived room {p[0]} x {p[1]})" if p else ""
            print(f"  {flag}stage {idx:>3}{where}  live x={st.x:>5}  "
                  f"random clears {clears}/{args.episodes}  {out}")
            ok = ok and clears > 0
    finally:
        env.close()

    print()
    if ok:
        print("PASS: every tested goal-side stage is finishable, so the backward "
              "curriculum has somewhere to start.")
    else:
        print("FAIL: at least one required goal-side stage could not be cleared.\n"
              "      Training would pin at the maximum stage and learn nothing.\n"
              "      Explore more (`smwrl.explore --resume`) so the archive records\n"
              "      cells that actually lead to the goal.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
