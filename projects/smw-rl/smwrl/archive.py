"""A Go-Explore style frontier archive of emulator save states.

Why this exists
---------------
The first curriculum harvested save states from the *policy's own* rollouts.
That cannot break through a wall: if the policy never passes x=310, every
harvested state sits at x<=310, so the curriculum can only ever rehearse what
the agent already does. YoshiIsland3 sat at x~=310 for 600k steps with a
curriculum spanning 148..310 -- the remedy was structurally incapable of
helping.

This archive is independent of policy quality. It keeps the best known state
per "cell" (a coarse bin of room + x + y), and exploration works by returning to
a promising cell and taking *random* actions from there. Random play from the
frontier stumbles over a hard jump far sooner than PPO finds it through a
-20 death penalty, and every new cell it reaches becomes a new launch point.
"""

from __future__ import annotations

import pickle
import random
import zlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Cell:
    """The best known way to be at one coarse location."""

    key: tuple[int, int, int]
    state: bytes            # zlib-compressed emulator snapshot
    x: int
    room: int
    steps: int              # steps taken to get here; fewer is better
    visits: int = 0         # times chosen as an exploration start
    chosen: int = 0
    fails: int = 0          # excursions from here that ended in death
    succeeds: int = 0       # excursions that survived
    # Fewest steps ever observed from this cell to the goal. None means no
    # excursion through this cell has ever finished the level -- so it is not a
    # safe place to start a curriculum stage, however far right it looks.
    to_goal: int | None = None

    @property
    def progress(self) -> int:
        """Ordering key across rooms: later rooms always beat earlier ones."""
        return self.room * 100_000 + self.x


@dataclass
class Archive:
    x_bin: int = 48
    y_bin: int = 96
    cells: dict[tuple[int, int, int], Cell] = field(default_factory=dict)
    frontier_bias: float = 3.0
    # v1 let repeated occurrences of a coarse cell consume a to_goal value that
    # the same relaxation pass had just written.  Distances collapsed toward
    # zero and curricula became arbitrarily ordered.  Loading v1 invalidates
    # those derived labels while preserving the expensive emulator states.
    route_format: int = 2

    # -- keys -------------------------------------------------------------
    def key_for(self, room: int, x: int, y: int) -> tuple[int, int, int]:
        return (room, x // self.x_bin, y // self.y_bin)

    # -- insertion --------------------------------------------------------
    def consider(self, room: int, x: int, y: int, state: bytes, steps: int) -> bool:
        """Record a visit. Returns True if this created or improved a cell."""
        key = self.key_for(room, x, y)
        cur = self.cells.get(key)
        if cur is None:
            self.cells[key] = Cell(key, zlib.compress(state, 6), x, room, steps)
            return True
        cur.visits += 1
        # Prefer reaching the same cell in fewer steps -- that is what makes the
        # eventual run fast rather than merely successful.
        if steps < cur.steps and cur.to_goal is None:
            cur.state = zlib.compress(state, 6)
            cur.steps = steps
            cur.x = x
            return True
        # Once a representative has a route proof, upstream proofs may depend
        # on that exact emulator snapshot.  Replacing it would require a full
        # dependency graph/recompute; keeping the proven representative is the
        # safe choice.  Speed can still improve through newly discovered bins.
        return False

    # -- selection --------------------------------------------------------
    def select(self) -> Cell | None:
        """Pick a cell to explore from.

        Weighted toward the frontier (high progress) and toward cells we have
        not launched from much, so exploration pushes forward instead of
        re-treading the easy opening of the level.
        """
        if not self.cells:
            return None
        cells = list(self.cells.values())
        best = max(c.progress for c in cells) or 1
        weights = []
        for c in cells:
            frontier = (c.progress / best) ** self.frontier_bias
            novelty = 1.0 / (1.0 + c.chosen) ** 0.5
            # Laplace-smoothed survival rate. A cell that sometimes works keeps
            # its frontier priority (YoshiIsland3 only broke through because a
            # hard cell was retried ~1400 times); a cell that has never once
            # survived is a trap -- typically a mid-fall state -- and gets
            # abandoned rather than soaking up every excursion.
            rate = (c.succeeds + 1) / (c.succeeds + c.fails + 2)
            weights.append(max(1e-9, frontier * novelty * rate ** 1.5))
        pick = random.choices(cells, weights=weights, k=1)[0]
        pick.chosen += 1
        return pick

    # -- reporting / export ------------------------------------------------
    @property
    def best_cell(self) -> Cell | None:
        return max(self.cells.values(), key=lambda c: c.progress, default=None)

    def mark_failure(self, key: tuple[int, int, int]) -> None:
        c = self.cells.get(key)
        if c is not None:
            c.fails += 1

    def mark_success(self, key: tuple[int, int, int]) -> None:
        c = self.cells.get(key)
        if c is not None:
            c.succeeds += 1

    def mark_on_winning_path(self, key: tuple[int, int, int], steps_to_goal: int) -> None:
        c = self.cells.get(key)
        if c is not None and (c.to_goal is None or steps_to_goal < c.to_goal):
            c.to_goal = steps_to_goal

    def relax_route(self, visited: list[tuple[tuple[int, int, int], int]],
                    cleared: bool, end_step: int) -> int:
        """Propagate distance-to-goal backwards along one excursion.

        For each visited cell i:  to_goal_i = min over later j of
        ((step_j - step_i) + to_goal_j), including the goal itself if this
        excursion cleared. Computed as a suffix minimum of (to_goal_j + step_j).

        Marking only on a clear was far too strict: one 220-step excursion
        almost never runs from the level start to the goal, so early cells never
        got a distance and the curriculum covered only the last ~20% of the
        level. Relaxing through cells that *already* know their distance lets
        the information walk backwards a chunk at a time.
        """
        # A Cell stores the earliest/fastest representative for its coarse key.
        # If an excursion loops through the key again, the later dynamic state
        # is not the stored snapshot and cannot prove a shorter route for it.
        # Retain only the first occurrence in this chronological trace.
        unique = []
        seen = set()
        for key, step in visited:
            if key not in seen:
                unique.append((key, step))
                seen.add(key)

        INF = float("inf")
        # Only routes known *before* this excursion are valid anchors.  Reading
        # c.to_goal after updating it earlier in this same reverse pass creates a
        # self-loop whenever a coarse key appears more than once and drives its
        # distance all the way to zero.
        known = {key: c.to_goal for key, c in self.cells.items()}
        best = float(end_step) if cleared else INF     # virtual goal: to_goal 0
        updated = 0
        for key, step in reversed(unique):
            c = self.cells.get(key)
            old_distance = known.get(key)
            if old_distance is not None:
                best = min(best, old_distance + step)
            if best == INF:
                continue
            t = int(best - step)
            if c is not None and t >= 0 and (c.to_goal is None or t < c.to_goal):
                c.to_goal = t
                updated += 1
        return updated

    @property
    def winning_cells(self) -> list[Cell]:
        """Cells from which the goal has actually been reached, far -> near."""
        return sorted((c for c in self.cells.values() if c.to_goal is not None),
                      key=lambda c: -c.to_goal)

    def curriculum_entries(self, n: int = 48) -> list[tuple[int, int, bytes]]:
        """Curriculum as (room, x, compressed_state), ordered start -> goal.

        The positions matter: training promotes a stage once the agent can
        reach the *next* stage, which needs to know where that is.
        """
        # A frontier state is not a curriculum state until an actual excursion
        # proves a route from that exact snapshot to the goal.  Falling back to
        # progress silently exported side rooms and migrated/corrupt v1 cells.
        cells = self.winning_cells
        cells = self._longest_forward_subsequence(cells)
        if not cells:
            return []
        if len(cells) > n:
            idx = sorted({round(i * (len(cells) - 1) / (n - 1)) for i in range(n)})
            cells = [cells[i] for i in idx]
        return [(c.room, c.x, c.state) for c in cells]

    @staticmethod
    def _longest_forward_subsequence(cells: list[Cell]) -> list[Cell]:
        """Keep the largest route-order subsequence with increasing positions.

        The current curriculum success test is positional, so duplicate or
        backward targets would either pass without moving or be unreachable.
        Alternate y bins and equal-distance ties can create those edges even
        with valid route distances.  Longest-subsequence filtering preserves as
        many proven stages as this coordinate model can express and fails safe
        for routes that genuinely require backtracking.
        """
        if not cells:
            return []
        best = [1] * len(cells)
        prev = [-1] * len(cells)
        pos = [(c.room, c.x) for c in cells]
        for i in range(len(cells)):
            for j in range(i):
                if pos[j] < pos[i] and best[j] + 1 > best[i]:
                    best[i] = best[j] + 1
                    prev[i] = j
        idx = max(range(len(cells)), key=lambda i: best[i])
        out = []
        while idx >= 0:
            out.append(cells[idx])
            idx = prev[idx]
        return list(reversed(out))

    def curriculum_states(self, n: int = 24) -> list[bytes]:
        """A spread of states ordered level-start -> goal.

        Prefers cells that are *known to lead to the goal*, ordered by their
        distance from it. Ranking by `progress` instead is unsafe: it assumes a
        higher room number means further along, but an SMW sub-room can be a
        detour or a bonus area. On YoshiIsland3 that put the deepest curriculum
        stage on a vine high above a pit in a vertical side-room, from which
        even random play died 12/12 -- so the backward curriculum could never
        get started.

        Returns no states until a clear-backed route has been recorded.
        """
        cells = self.winning_cells
        if not cells:
            return []
        if len(cells) <= n:
            return [c.state for c in cells]
        idx = [round(i * (len(cells) - 1) / (n - 1)) for i in range(n)]
        return [cells[i].state for i in sorted(set(idx))]

    # -- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(pickle.dumps(self))
        tmp.replace(path)

    @staticmethod
    def load(path: Path) -> "Archive":
        archive = pickle.loads(Path(path).read_bytes())
        # getattr() would see the dataclass class default even when an old
        # pickle has no instance field, incorrectly treating v1 as migrated.
        if archive.__dict__.get("route_format", 1) < 2:
            for cell in archive.cells.values():
                cell.to_goal = None
            archive.route_format = 2
        return archive
