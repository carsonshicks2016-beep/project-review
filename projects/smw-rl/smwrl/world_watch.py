"""Watch the deepest run the whole-game archive has found.

Watching exploration itself is not worth it: it restores a save state, plays a
few hundred random frames, then teleports somewhere else. That is a search, not
a playthrough.

What *is* watchable is the frontier. This restores the deepest cell in the
archive and plays on from there, so you see the furthest point reached and what
the game looks like when it gets there. It re-reads the archive between runs, so
leaving it open while `smwrl.worldrun` explores shows the frontier advancing.

    python -m smwrl.world_watch                # frontier, live-updating
    python -m smwrl.world_watch --tour         # step through the archive in order
"""

from __future__ import annotations

import argparse
import pickle
import time
import zlib
from pathlib import Path

import numpy as np
import pygame

import smwrl.overworld as ow
from smwrl.actions import ACTION_TABLE, N_ACTIONS
from smwrl.explore import ACTION_P
from smwrl.worldrun import make_world_env

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE = ROOT / "checkpoints" / "world_archive.pkl"
SCALE = 3
BG, FG, DIM, ACCENT = (14, 14, 18), (232, 232, 238), (128, 128, 140), (255, 205, 90)


def load_cells(path: Path):
    """Returns (cells, mtime). Raises on a genuinely unreadable archive.

    An earlier version swallowed every exception and returned an empty list,
    so a load failure looked exactly like "no cells yet" and the viewer exited
    silently instead of reporting the problem.
    """
    from smwrl.world_archive import load_archive

    for attempt in range(5):        # the explorer rewrites this file as it goes
        try:
            archive = load_archive(path)
            break
        except (EOFError, pickle.UnpicklingError, OSError):
            time.sleep(0.4)
    else:
        raise SystemExit(f"could not read {path} (still being written?)")
    return sorted(archive.cells.values(), key=lambda c: c.progress), path.stat().st_mtime


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    ap.add_argument("--tour", action="store_true",
                    help="walk the archive from the start instead of the frontier")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--seconds", type=float, default=8.0,
                    help="how long to play from each restored cell")
    args = ap.parse_args()

    if not args.archive.exists():
        raise SystemExit(f"no archive at {args.archive}; run smwrl.worldrun first")

    pygame.init()
    pygame.display.set_caption("SMW whole-game archive")
    screen = pygame.display.set_mode((256 * SCALE, 224 * SCALE + 54))
    font = pygame.font.SysFont("Menlo", 15)
    small = pygame.font.SysFont("Menlo", 13)
    clock = pygame.time.Clock()

    env = make_world_env()
    env.reset()
    rng = np.random.default_rng()
    cells, mtime = load_cells(args.archive)
    idx = 0
    running = True

    try:
        while running and cells:
            # Re-read: the explorer rewrites this file as it goes.
            if args.archive.stat().st_mtime != mtime:
                cells, mtime = load_cells(args.archive)
            cell = cells[idx % len(cells)] if args.tour else cells[-1]
            env.em.set_state(zlib.decompress(cell.state))

            deadline = time.time() + args.seconds
            while running and time.time() < deadline:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT or (
                        e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                        running = False
                # Play on from the restored state the same way exploration does.
                action = ACTION_TABLE[int(rng.choice(N_ACTIONS, p=ACTION_P))]
                for _ in range(int(rng.integers(4, 17))):
                    env.step(action)

                screen.fill(BG)
                frame = env.em.get_screen()
                surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                screen.blit(pygame.transform.scale(surf, (256 * SCALE, 224 * SCALE)), (0, 0))
                ram = env.get_ram()
                y = 224 * SCALE + 6
                mode = int(ram[ow.GAME_MODE])
                where = ("in level" if mode == ow.MODE_LEVEL
                         else "overworld" if mode == ow.MODE_OVERWORLD else f"mode 0x{mode:02X}")
                screen.blit(font.render(
                    f"frontier: translevel {cell.translevel}  room {cell.room}  "
                    f"x {cell.x}   levels cleared {cell.cleared}", True, ACCENT), (10, y))
                screen.blit(small.render(
                    f"archive {len(cells)} cells   live translevel "
                    f"{int(ram[ow.TRANSLEVEL])}   {where}"
                    f"   {'(tour %d/%d)' % (idx % len(cells) + 1, len(cells)) if args.tour else '(deepest cell)'}",
                    True, DIM), (10, y + 22))
                pygame.display.flip()
                clock.tick(args.fps)
            idx += 1
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
