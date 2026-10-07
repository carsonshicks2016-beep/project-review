"""Prototype: a Go-Explore archive over the WHOLE game, not one level.

Per-level training needs an entry save state for every level, and we have none
for the castles, ghost houses or switch palaces a real playthrough must pass
through. Scoping the archive to the whole game removes that problem: exploration
reaches those levels by *playing into them*, so their states are discovered
rather than supplied.

Cells are keyed by (translevel, room, x_bin, y_bin) so the same x in different
levels never collides. Progress orders by levels cleared first, so finishing a
level always beats running further inside one.

Menus and the overworld stay scripted -- they are deterministic, and letting
random actions wander a map wastes excursions. The agent's random play is spent
only where it can discover something: inside a level.

    python -m smwrl.worldrun --iters 400

The milestone this exists to answer: can exploration cross a level boundary on
its own? If the frontier never leaves the first level, whole-game training is
not viable and we should keep per-level specialists.
"""

from __future__ import annotations

import argparse
import pickle
import random
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import stable_retro as retro

import smwrl.overworld as ow
from smwrl.actions import ACTION_TABLE, N_ACTIONS
from smwrl.explore import ACTION_P
from smwrl.ram import decode
from smwrl.retro_env import GAME, register_integration
from smwrl.world_archive import (MAP_CELL, WorldArchive, WorldCell, game_progress,
                                 load_archive)

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"

X_BIN, Y_BIN = 64, 96


def make_world_env():
    register_integration()
    return retro.make(GAME, state=retro.State.NONE,
                      inttype=retro.data.Integrations.CUSTOM_ONLY, render_mode=None)


def state_of(env) -> tuple[int, int, int, int]:
    """(game_mode, translevel, x, y) straight from RAM."""
    ram = env.get_ram()
    return (int(ram[ow.GAME_MODE]), int(ram[ow.TRANSLEVEL]),
            int(ram[0x0094]) | (int(ram[0x0095]) << 8),
            int(ram[0x0096]) | (int(ram[0x0097]) << 8))


YOSHIS_HOUSE = 40      # tutorial room: no goal tape, so it can never be "cleared"


def advance_overworld(env, move_first: bool = False, avoid=None,
                      max_frames: int = 900) -> bool:
    """Scripted: settle on the map, then walk into a level.

    `avoid` is the map position we entered the *previous* level from. Entering
    the node Mario is standing on right after a clear often walks straight back
    into the level he just finished -- that is why exploration ping-ponged
    between translevels 42 and 39, racking up 277 clears across only two
    levels. If the map has not moved us off `avoid`, we move first.
    """
    # overworld.py raises TimeoutError when the map never becomes controllable.
    # A single bad transition must not end a 10,000-excursion run, so every
    # scripted map operation here is contained and reported as a failed
    # excursion instead.
    try:
        ow.wait_settled(env, stable_frames=20, timeout=400)
    except TimeoutError:
        return False
    try:
        here = ow.position(env)
    except Exception:
        here = None
    if avoid is not None and here == avoid:
        move_first = True          # the map did not advance us; do not re-enter
    if move_first:
        # At boot Mario stands on Yoshi's House. Entering the node he is already
        # on just walks back into a room with no goal tape, which is exactly
        # where the first prototype run got stuck.
        for d in ("RIGHT", "UP", "LEFT", "DOWN"):
            before = ow.position(env)
            try:
                if ow.move(env, d, settle=90) != before:
                    break
            except TimeoutError:
                return False
    for direction in ("RIGHT", "UP", "LEFT", "DOWN", None):
        try:
            if ow.enter_level(env, timeout=240):
                return True
            if direction is None:
                break
            ow.move(env, direction, settle=90)
        except TimeoutError:
            return False
        if ow.mode(env) == ow.MODE_LEVEL:
            return True
    return ow.mode(env) == ow.MODE_LEVEL


NOOP = np.zeros(12, np.uint8)


def choose_branch_and_enter(env, archive, rng, avoid=None) -> bool:
    """Probe every open map direction, prefer an unvisited node, then enter.

    The old version tried RIGHT, UP, LEFT, DOWN and took the first that moved.
    At Yoshi's House that is always RIGHT, so every explorer walked into
    Yoshi's Island 2 and *never once played Yoshi's Island 1* -- which the game
    requires before it will open the path past Yoshi's Island 3. Four parallel
    explorers made the identical wrong turn 24,000 times.

    Probing is cheap because the map state can be snapshotted and restored, so
    we can look down every branch before committing to one.
    """
    # Generous: resuming on a map cell archived mid-animation, or arriving here
    # straight after a clear that opened a path, both mean a long unsettled
    # stretch. A 400-frame limit turned that into a failed excursion, which then
    # threw away everything the excursion had found.
    # Tolerant, for the same reason archive_map_state is: the cells carrying the
    # most game progress are the ones snapshotted while the map was still
    # animating a path open. Refusing to work from an unsettled map made every
    # single excursion from them fail -- 0 successes in 57 attempts on the two
    # most advanced cells in the archive. The probes below have their own waits.
    try:
        ow.wait_settled(env, stable_frames=20, timeout=1800)
    except TimeoutError:
        pass
    snap = env.em.get_state()
    here = ow.position(env)

    options: list[tuple[str | None, tuple[int, int]]] = []
    if avoid is None or here != avoid:
        options.append((None, here))          # entering the node we stand on
    for d in ("RIGHT", "LEFT", "UP", "DOWN"):
        env.em.set_state(snap); env.step(NOOP)
        try:
            ow.wait_settled(env, stable_frames=12, timeout=250)
            after = ow.move(env, d, settle=100)
        except TimeoutError:
            continue
        if after != here:
            options.append((d, after))
    if not options:
        return False

    # Try every option, not one. Committing to a single random branch and giving
    # up if it failed cost the search its whole frontier: the two cells holding
    # the banked events failed 57 times out of 57, because the node Mario stands
    # on after a path opens is often not one you can enter -- and the excursion
    # never looked anywhere else. Unvisited nodes still go first; the rest are a
    # fallback rather than a dead end.
    visited = archive.visited()
    fresh = [o for o in options if o[1] not in visited]
    stale = [o for o in options if o[1] in visited]
    pool = ([fresh[i] for i in rng.permutation(len(fresh))]
            + [stale[i] for i in rng.permutation(len(stale))])

    for direction, node in pool:
        env.em.set_state(snap); env.step(NOOP)
        try:
            ow.wait_settled(env, stable_frames=12, timeout=250)
            if direction is not None:
                ow.move(env, direction, settle=100)
            entered = ow.enter_level(env, timeout=300)
        except TimeoutError:
            continue
        if entered:
            visited.add(node)
            return True
    return False


def archive_map_state(env, archive, cleared: int, steps: int) -> None:
    """Record standing on the map so a different branch can be taken later.

    A settle timeout must never discard the state. This used to `return` when
    the map would not settle in 300 frames, and the map takes far longer than
    that exactly when a clear opens a new path -- which is exactly when
    `cleared` has just risen. So the only mechanism that recorded new progress
    gave up whenever there was new progress to record: measured over one night,
    98 banked events and not one surviving cell in the archive.

    A settled position is still preferable, because it bins to the node Mario
    actually ends on, so wait for one generously -- but archive regardless. The
    emulator snapshot is the valuable part, and anything restoring it settles
    the map itself before acting.
    """
    try:
        ow.wait_settled(env, stable_frames=15, timeout=1800)
    except TimeoutError:
        pass
    try:
        mx, my = ow.position(env)
    except Exception:
        mx, my = 0, 0
    archive.consider(MAP_CELL, 0, mx, my, cleared, env.em.get_state(), steps)


def explore(iters: int, rollout: int, sticky: tuple[int, int],
            verbose: bool = True, save_path: Path | None = None,
            save_every: int = 250, seed_archive: WorldArchive | None = None) -> WorldArchive:
    env = make_world_env()
    # Always reset: the seeded path skips boot_to_map(), which was the only
    # caller of env.reset(), and stable-retro refuses to step before one.
    env.reset()
    archive = seed_archive if seed_archive is not None else WorldArchive()
    rng = np.random.default_rng()

    if archive.cells:
        print(f"seeded with {len(archive.cells)} cells; skipping boot", flush=True)
    else:
        print("booting from power-on ...", flush=True)
        boot = ow.boot_to_map(env)
        if not boot.ok:
            raise SystemExit(f"boot failed: {boot.detail}")
        archive_map_state(env, archive, game_progress(env.get_ram()), 0)
        if not choose_branch_and_enter(env, archive, rng):
            raise SystemExit("booted to the map but could not enter a level")
        mode, tl, x, y = state_of(env)
        if tl == YOSHIS_HOUSE:
            raise SystemExit("still inside Yoshi's House; the map move did not take")
        archive.consider(tl, 0, x, y, game_progress(env.get_ram()), env.em.get_state(), 0)
        print(f"first level loaded: translevel {tl}", flush=True)

    t0 = time.time()
    clears = 0
    # Per-level accounting. "clears 615, events 1" is ambiguous between a level
    # we finish constantly that opens nothing and one we never finish, and those
    # want opposite fixes -- so record where each clear happened and whether it
    # advanced the game.
    cleared_by_level: Counter[int] = Counter()
    scored_by_level: Counter[int] = Counter()
    for it in range(iters):
        cell = archive.select()
        if cell is None:
            break
        try:
            env.em.set_state(zlib.decompress(cell.state))
        except Exception:
            cell.fails += 1
            continue
        room, cleared, prev_x = cell.room, cell.cleared, cell.x
        entry_pos = None                 # map node this level was entered from
        if cell.translevel == MAP_CELL:
            # Resuming on the overworld: choose a branch (biased to unvisited
            # nodes) instead of replaying whichever one boot happened to take.
            if not choose_branch_and_enter(env, archive, rng):
                cell.fails += 1
                continue
            prev_x = 0
            room = 0
        steps, dead = 0, False
        pending: list[tuple[int, tuple, bytes]] = []
        tl_entered = cell.translevel

        try:
          while steps < rollout and not dead:
              action = ACTION_TABLE[int(rng.choice(N_ACTIONS, p=ACTION_P))]
              for _ in range(int(rng.integers(sticky[0], sticky[1] + 1))):
                  env.step(action)
                  steps += 1
                  ram = env.get_ram()
                  st = decode({
                      "x_pos": int(ram[0x0094]) | (int(ram[0x0095]) << 8),
                      "y_pos": int(ram[0x0096]) | (int(ram[0x0097]) << 8),
                      "game_mode": int(ram[ow.GAME_MODE]),
                      "player_anim": int(ram[0x0071]),
                      "lives": int(ram[ow.LIVES]),
                      "end_level_timer": int(ram[0x1493]),
                      "translevel": int(ram[ow.TRANSLEVEL]),
                  })
                  if st.dying:
                      dead = True
                      break
                  if st.cleared:
                      # Level finished -- a goal tape, or a switch palace, whose
                      # switch ends the level the same way. Re-read progress from
                      # RAM rather than counting these: re-clearing a beaten
                      # level ends it again and must not buy frontier priority.
                      clears += 1
                      # Which level, and did finishing it actually advance the
                      # game? Counting clears alone cannot tell "we finish this
                      # level constantly and it opens nothing" apart from "we
                      # never finish it" -- and those want opposite fixes.
                      tl_cleared = st.translevel or tl_entered
                      cleared_by_level[tl_cleared] += 1
                      before = cleared
                      entry_pos = None
                      if ow.wait_for_map(env, timeout=1200):
                          try:
                              # The event counter is written during the map
                              # animation, AFTER game_mode flips to overworld.
                              # Reading at the flip always returned the
                              # pre-clear value, so progress never rose and the
                              # map looked permanently closed.
                              ow.wait_settled(env, stable_frames=20, timeout=400)
                              cleared = game_progress(env.get_ram())
                              if cleared > before:
                                  scored_by_level[tl_cleared] += 1
                                  print(f"  *** translevel {tl_cleared} banked "
                                        f"progress {before} -> {cleared} ***",
                                        flush=True)
                              entry_pos = ow.position(env)
                              archive_map_state(env, archive, cleared,
                                                cell.steps + steps)
                          except TimeoutError:
                              entry_pos = None
                      if choose_branch_and_enter(env, archive, rng, avoid=entry_pos):
                          room, prev_x = 0, 0
                      else:
                          dead = True
                      break
                  if not st.in_level:
                      continue
                  if prev_x - st.x > 400:
                      room += 1
                  prev_x = st.x
                  pending.append((steps, (st.translevel, room, st.x, max(0, st.y)),
                                  env.em.get_state()))
                  if steps >= rollout:
                      break

        except TimeoutError:
            # A transition the scripted map handling could not drive. Treat the
            # excursion as failed and move on rather than ending the run.
            dead = True

        # Delayed commit: only archive states the player survived past, or the
        # archive fills with mid-fall snapshots that kill every excursion.
        if not dead:
            for at, (tl_, rm, cx, cy), blob in pending:
                if steps - at >= 12:
                    archive.consider(tl_, rm, cx, cy, cleared, blob, cell.steps + at)
            cell.succeeds += 1
        else:
            cell.fails += 1

        # Checkpoint: a 10k-excursion run must survive being stopped. The
        # per-level explorer learned this the hard way -- it only saved at the
        # end, so killing it discarded everything it had found.
        if save_path is not None and (it + 1) % save_every == 0:
            archive.save(save_path)

        if verbose and (it + 1) % 25 == 0:
            b = archive.best_cell
            print(f"  iter {it+1:>5}/{iters}  cells {len(archive.cells):>5}  "
                  f"translevels {len(archive.translevels):>2}  "
                  f"frontier tl {b.translevel} room {b.room} x {b.x} (events {b.cleared})  "
                  f"clears {clears} {format_clears(cleared_by_level, scored_by_level)}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)

    env.close()
    archive.clear_report = (cleared_by_level, scored_by_level)
    return archive


def format_clears(cleared: Counter, scored: Counter) -> str:
    """`[39:615*0 42:80*1]` -- level: times finished, times it advanced the game."""
    if not cleared:
        return "[]"
    parts = [f"{tl}:{n}*{scored.get(tl, 0)}"
             for tl, n in sorted(cleared.items(), key=lambda kv: -kv[1])]
    return "[" + " ".join(parts) + "]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--rollout", type=int, default=220)
    ap.add_argument("--sticky-min", type=int, default=4)
    ap.add_argument("--sticky-max", type=int, default=16)
    ap.add_argument("--out", default=str(CKPT / "world_archive.pkl"))
    ap.add_argument("--resume-from", type=Path,
                    help="start from an existing archive instead of booting fresh")
    args = ap.parse_args()

    seed = None
    if args.resume_from and Path(args.resume_from).exists():
        seed = load_archive(args.resume_from)
        print(f"resuming from {args.resume_from}: {len(seed.cells)} cells")
    archive = explore(args.iters, args.rollout, (args.sticky_min, args.sticky_max),
                      save_path=Path(args.out), seed_archive=seed)
    b = archive.best_cell
    print(f"\ncells: {len(archive.cells)}")
    print(f"distinct translevels reached: {sorted(archive.translevels)}")
    if b:
        print(f"frontier: translevel {b.translevel} room {b.room} x {b.x}, "
              f"{b.cleared} completion event(s)")

    cleared_by_level, scored_by_level = getattr(archive, "clear_report", (Counter(), Counter()))
    if cleared_by_level:
        print("\nclears by level (finished / of those, advanced the game):")
        for tl, n in sorted(cleared_by_level.items(), key=lambda kv: -kv[1]):
            scored = scored_by_level.get(tl, 0)
            note = "" if scored else "   <- opens nothing"
            print(f"  translevel {tl:>3}: {n:>5} / {scored}{note}")
    archive.save(Path(args.out))
    print(f"wrote {args.out}")

    print("\nMILESTONE: " + ("PASS -- exploration crossed a level boundary on its own"
                             if len(archive.translevels) > 1 else
                             "FAIL -- never left the first level"))


if __name__ == "__main__":
    main()
