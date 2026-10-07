"""Go-Explore phase 1: push the frontier with random play, no policy needed.

    python -m smwrl.explore --level YoshiIsland3 --iters 400

Repeatedly: pick a cell from the archive, restore that exact emulator state,
then take *sticky random* actions for a while and archive anything new.

Two details do most of the work:

* **Sticky actions.** Holding one action for 4-16 agent steps, rather than
  re-rolling every step, is what produces a committed run-up and a full-height
  jump. Per-step random actions jitter on the spot and clear nothing.
* **A rightward prior.** Uniform sampling over 14 actions wastes most rollouts
  walking left. Weighting the right-moving actions turns random play into
  something that actually traverses a side-scroller.

Writes checkpoints/<level>/archive.pkl plus a curriculum.pkl for training.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np

from smwrl.actions import ACTION_LABELS, N_ACTIONS
from smwrl.archive import Archive
from smwrl.curriculum_store import write_verified_curriculum
from smwrl.env import make_env
from smwrl.levels import LEVELS
from smwrl.wrappers import EpisodeConfig, RewardConfig

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"

# Index order matches smwrl.actions.ACTION_COMBOS. Right-moving and jumping
# actions carry most of the weight; left and idle stay available so the agent
# can still back up for a run-up, but they do not dominate.
ACTION_WEIGHTS = np.array([
    0.5,   # 0  idle
    1.5,   # 1  right
    3.0,   # 2  right + Y (run)
    2.0,   # 3  right + B (jump)
    4.0,   # 4  right + Y + B (run-jump)
    1.5,   # 5  right + Y + A (spin)
    0.6,   # 6  right + down
    0.8,   # 7  B
    1.0,   # 8  Y + B
    0.4,   # 9  left
    0.5,   # 10 left + Y
    0.5,   # 11 left + B
    0.4,   # 12 down
    0.4,   # 13 up
], dtype=np.float64)
ACTION_P = ACTION_WEIGHTS / ACTION_WEIGHTS.sum()


def chronological_route(start_key, visited):
    """Return an excursion trace in increasing step order.

    Archive.relax_route walks this list backwards.  Appending the step-zero
    launch cell after the later visits meant it was processed *first* during
    that reverse walk, so route-to-goal evidence could never propagate all the
    way back to the archive start unless one excursion cleared the whole level.
    """
    return [(start_key, 0), *visited]


def representative_route(archive: Archive, start_cell, visited, base_steps: int):
    """Trace only occurrences whose state is the archive representative.

    A coarse (room, x-bin, y-bin) can be reached with different momentum,
    powerups, or hazard states.  Route proof from a slower/later visit must not
    be attached to the faster snapshot stored under the same key.
    """
    trace = chronological_route(start_cell.key, visited)
    out = []
    seen = set()
    for key, relative_step in trace:
        if key in seen:
            continue
        seen.add(key)
        cell = archive.cells.get(key)
        if cell is not None and cell.steps == base_steps + relative_step:
            out.append((key, relative_step))
    return out


def explore(level: str, iters: int, rollout: int, sticky: tuple[int, int],
            seed_states: list[bytes] | None = None, verbose: bool = True,
            survive: int = 12, archive: Archive | None = None,
            save_path: Path | None = None, save_every: int = 500,
            seed: int = 0) -> Archive:
    """Explore outward from the archive frontier.

    `survive` is the delayed-commit window: a candidate state is only archived
    once the player is still alive that many agent steps later. Without it the
    archive fills with states captured mid-fall into a pit -- they have the
    highest x, so frontier-weighted selection picks them forever and every
    excursion dies on the first step.
    """
    env = make_env(level, RewardConfig(),
                   EpisodeConfig(max_steps=100_000, stuck_steps=100_000, noop_max=0),
                   monitor=False)
    # Extend the archive we were handed rather than starting over: the earlier
    # --resume rebuilt from a handful of seed states and silently threw away
    # every other cell that had been found.
    archive = archive if archive is not None else Archive()
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # Seed the archive with the level start (and anything handed in).
    env.reset(seed=seed)
    st = env.last_state
    archive.consider(0, st.x, max(0, st.y), env.unwrapped.em.get_state(), 0)
    for blob in seed_states or []:
        env.reset()
        env.unwrapped.em.set_state(blob)
        env.step(0)
        st = env.last_state
        if st.in_level:
            archive.consider(0, st.x, max(0, st.y), env.unwrapped.em.get_state(), 0)

    t0 = time.time()
    new_cells = 0
    clears = 0
    for it in range(iters):
        cell = archive.select()
        if cell is None:
            break

        env.reset()
        import zlib

        env.unwrapped.em.set_state(zlib.decompress(cell.state))
        base_steps, base_room = cell.steps, cell.room
        room = base_room
        prev_x = cell.x
        steps = 0
        # (age, room, x, y, state, steps)
        pending: list[tuple[int, int, int, int, bytes, int]] = []
        # Every cell this excursion passed through, with the step it was seen.
        # If we reach the goal, all of them are on a route that works.
        visited: list[tuple[tuple[int, int, int], int]] = []
        dead = cleared = False

        def commit(force_all: bool = False) -> int:
            """Archive candidates that have now survived `survive` steps."""
            added = 0
            keep = []
            for age, rm, cx, cy, blob, stp in pending:
                if force_all or steps - age >= survive:
                    if archive.consider(rm, cx, cy, blob, stp):
                        added += 1
                else:
                    keep.append((age, rm, cx, cy, blob, stp))
            pending[:] = keep
            return added

        while steps < rollout:
            action = int(rng.choice(N_ACTIONS, p=ACTION_P))
            hold = int(rng.integers(sticky[0], sticky[1] + 1))
            for _ in range(hold):
                _, _, term, trunc, info = env.step(action)
                steps += 1
                s = env.last_state
                if term and info.get("death"):
                    dead = True
                    break
                if not s.in_level:
                    continue
                if prev_x - s.x > 400:      # room transition
                    room += 1
                if info.get("level_cleared"):
                    cleared = True
                    break
                prev_x = s.x
                visited.append((archive.key_for(room, s.x, max(0, s.y)), steps))
                pending.append((steps, room, s.x, max(0, s.y),
                                env.unwrapped.em.get_state(), base_steps + steps))
                new_cells += commit()
                if steps >= rollout:
                    break
            if dead or cleared:
                break

        if dead:
            # Everything still pending led into the death; discard it.
            archive.mark_failure(cell.key)
        else:
            # A clear proves even the tail states safe.  Merely reaching the end
            # of the rollout does not: a pending state may still be mid-fall and
            # has not satisfied the delayed-commit survival window.
            new_cells += commit(force_all=cleared)
            archive.mark_success(cell.key)
            if cleared:
                clears += 1
            # Propagate distance-to-goal back along everything just walked
            # through -- not only on a clear, or early cells never get one.
            route = representative_route(archive, cell, visited, base_steps)
            archive.relax_route(route, cleared, steps)

        # Checkpoint the archive so a long run can be stopped at any point
        # without losing what it found.
        if save_path is not None and (it + 1) % save_every == 0:
            archive.save(save_path)

        if verbose and (it + 1) % 50 == 0:
            b = archive.best_cell
            print(f"  iter {it + 1:>5}/{iters}  cells {len(archive.cells):>5}  "
                  f"frontier room {b.room} x {b.x:>5}  clears {clears:>3}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    env.close()
    return archive


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="YoshiIsland3", choices=sorted(LEVELS))
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--rollout", type=int, default=140, help="agent steps per excursion")
    ap.add_argument("--sticky-min", type=int, default=4)
    ap.add_argument("--sticky-max", type=int, default=16)
    ap.add_argument("--curriculum-size", type=int, default=48,
                    help="denser is better: the agent must practise the exact "
                         "obstacle it is stuck on, not one 400px away")
    ap.add_argument("--resume", action="store_true", help="extend an existing archive")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export-only", action="store_true",
                    help="re-export curriculum.pkl from the saved archive without exploring")
    args = ap.parse_args()

    out = CKPT / args.level
    out.mkdir(parents=True, exist_ok=True)
    apath = out / "archive.pkl"

    if args.export_only:
        if not apath.exists():
            raise SystemExit(f"no archive at {apath}")
        archive = Archive.load(apath)
        states = archive.curriculum_entries(args.curriculum_size)
        # Persist migration/invalidation before binding the exported payload.
        archive.save(apath)
        try:
            write_verified_curriculum(apath, out / "curriculum.pkl", archive, states)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        b = archive.best_cell
        print(f"{len(archive.cells)} cells, frontier room {b.room} x {b.x} -> "
              f"curriculum.pkl with {len(states)} states")
        return

    prev = None
    if args.resume and apath.exists():
        prev = Archive.load(apath)
        print(f"resuming: extending archive with {len(prev.cells)} cells "
              f"({len(prev.winning_cells)} on a winning path)")

    print(f"exploring {args.level}: {args.iters} excursions of {args.rollout} steps")
    archive = explore(args.level, args.iters, args.rollout,
                      (args.sticky_min, args.sticky_max), archive=prev,
                      save_path=apath, seed=args.seed)

    best = archive.best_cell
    win = archive.winning_cells
    print(f"\ncells: {len(archive.cells)}  on a winning path: {len(win)}")
    if win:
        print(f"closest to goal: {win[-1].to_goal} steps;  furthest: {win[0].to_goal} steps")
    else:
        print("WARNING: no clear-backed route recorded; curriculum export will be "
              "refused until exploration proves a path to the goal")
    if best:
        print(f"frontier: room {best.room} x {best.x} (reached in {best.steps} steps)")
        print(f"level is ~{LEVELS[args.level].length} px -- "
              f"{best.x / LEVELS[args.level].length:.0%} of room 0")

    archive.save(apath)
    states = archive.curriculum_entries(args.curriculum_size)
    try:
        write_verified_curriculum(apath, out / "curriculum.pkl", archive, states)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    print(f"wrote {apath.name} and curriculum.pkl ({len(states)} stages, "
          f"{sum(len(e[2]) for e in states) / 1e6:.1f} MB compressed)")


if __name__ == "__main__":
    main()
