"""Cell types for the whole-game archive, plus a loader that survives `python -m`.

These live in their own module for a specific reason. They used to be defined in
`smwrl/worldrun.py`, which is normally executed as `python -m smwrl.worldrun` --
so at pickling time that module *is* `__main__`, and every archive recorded its
classes as `__main__.WorldArchive`. Nothing else could then load one: the viewer
failed with `Can't get attribute 'WorldArchive' on <module '__main__'>`.

Defining them here means they always pickle under a stable module path, and
`load_archive` still accepts the older `__main__`-labelled files.
"""

from __future__ import annotations

import io
import pickle
import random
import zlib
from dataclasses import dataclass, field
from pathlib import Path

X_BIN, Y_BIN = 64, 96

# Verified by RAM diff across a first level clear on a fresh game: this counter
# steps 0 -> 1 exactly when a completion EVENT fires and the map opens a new
# path. Counting goal tapes instead let re-clears inflate progress -- ~580 of
# them produced two translevels and a closed map.
EVENTS = 0x1F2E

# The four switch-palace flags, in map order. A switch palace opens no path, so
# 0x1F2E never moves for one -- but pressing the switch is real, irreversible
# progress: its blocks turn solid for the rest of the game. Scoring the event
# counter alone made the Yellow Switch Palace unscorable, and the search wedged
# in it: sticky-random play presses that switch in 38 of 240 excursions, yet
# after ~92,000 excursions not one archived cell had the flag set.
# 0x1F28 is confirmed live -- it steps 0 -> 1 the frame the yellow switch is hit.
SWITCHES = (0x1F27, 0x1F28, 0x1F29, 0x1F2A)      # green, yellow, blue, red

# Cells recorded while standing on the overworld map, not inside a level.
# Without these the archive can only ever resume mid-level, so a branch chosen
# once at boot is chosen forever -- which is how every explorer walked into
# Yoshi's Island 2 and never played Yoshi's Island 1.
MAP_CELL = -1


def game_progress(ram) -> int:
    """Irreversible world-state advances: completion events plus switches hit.

    This is the whole search's notion of "further along", so anything the game
    records permanently and cannot undo belongs here.
    """
    return int(ram[EVENTS]) + sum(1 for addr in SWITCHES if ram[addr])


def _keep_counters(winner: "WorldCell", other: "WorldCell") -> None:
    """Carry the higher excursion counters onto the cell we are keeping.

    Deliberately max, not sum. Every shard resumes from the merged archive, so
    both sides already carry the same history, and adding them re-counted that
    shared base once per shard per merge -- compounding every 900 seconds. The
    switch palace ended up recording 699,509 excursions against roughly 15,000
    ever run, and `select`'s novelty prior read the inflated figure as a level
    explored 47 times more thoroughly than it was.
    """
    winner.chosen = max(winner.chosen, other.chosen)
    winner.fails = max(winner.fails, other.fails)
    winner.succeeds = max(winner.succeeds, other.succeeds)


@dataclass
class WorldCell:
    key: tuple
    state: bytes                 # zlib-compressed emulator snapshot
    translevel: int
    room: int
    x: int
    cleared: int                 # completion EVENTS recorded (0x7E1F2E)
    steps: int
    chosen: int = 0
    fails: int = 0
    succeeds: int = 0

    @property
    def local_progress(self) -> int:
        """Depth inside this level alone -- room and x, no game-wide progress.

        Selection ranks each cell against the best in its *own* translevel, and
        the progress term is 10^7 times larger than any x: including it here
        would flatten every cell in a level to the same score and destroy the
        within-level frontier.
        """
        return self.room * 100_000 + self.x

    @property
    def progress(self) -> int:
        """Game-wide progress dominates: a new event beats any x inside a level.

        `cleared` counts completion events and switch palaces, neither of which
        a re-clear can raise, so replaying a beaten level cannot buy priority.
        """
        return self.cleared * 10_000_000 + self.local_progress


@dataclass
class WorldArchive:
    cells: dict = field(default_factory=dict)
    frontier_bias: float = 3.0

    def key_for(self, translevel: int, room: int, x: int, y: int) -> tuple:
        return (translevel, room, x // X_BIN, max(0, y) // Y_BIN)

    def consider(self, translevel, room, x, y, cleared, state, steps) -> bool:
        """Keep the most advanced state at this cell, and the cheapest route to it.

        Progress outranks cheapness, which the first version had backwards: a
        state carrying a newly-pressed switch was discarded in favour of an
        older, cheaper visit to the same spot, so the one discovery the search
        needed was thrown away every time it was made.
        """
        key = self.key_for(translevel, room, x, y)
        cur = self.cells.get(key)
        if cur is None:
            self.cells[key] = WorldCell(key, zlib.compress(state, 6),
                                        translevel, room, x, cleared, steps)
            self.barren()[translevel] = 0
            return True
        if (cleared, -steps) > (cur.cleared, -cur.steps):
            gained = cleared > cur.cleared
            cur.state = zlib.compress(state, 6)
            cur.steps, cur.cleared = steps, cleared
            if gained:
                # Only a new cell or new game progress counts as yield. A
                # cheaper route to a cell we already hold is not a discovery,
                # and crediting it would make a re-clear loop -- which shaves
                # steps off known cells constantly -- look productive forever.
                self.barren()[translevel] = 0
            return True
        return False

    def select(self) -> WorldCell | None:
        """Pick a cell to explore from: frontier-weighted *within each level*,
        novelty-adjusted, and discounted by how often excursions from it died.

        The frontier used to be one global scalar. Because game progress is
        worth 10^7 there, every cell reached after the first completion event
        outranked every other cell in the archive -- so the whole search
        collapsed onto whichever level it happened to be in when that event
        fired. That level was the Yellow Switch Palace, a dead end that opens no
        map path and so can never score again: it took 79% of ~92,000 excursions
        while Yoshi's Island 2 sat on a single unvisited cell.

        Two changes, and the second is what makes this general:

        * Rank each cell against the deepest cell in its *own* translevel, so
          every level keeps a live frontier instead of one owning them all.
        * Spread by how long a level has gone without yielding anything, not by
          how many cells it holds. Cell count barely moved for the switch
          palace -- 41 cells absorbing 72,771 excursions -- so counting cells
          never noticed. Counting all-time excursions instead overcorrected:
          it exiled the palace on sunk cost, down to 1 pick in 500, even though
          one more excursion there was worth a switch. Barrenness resets the
          moment a level produces something, so effort drains away from a dead
          area and flows back if it ever comes alive again.
        """
        if not self.cells:
            return None
        cells = list(self.cells.values())
        barren = self.barren()
        deepest: dict[int, int] = {}
        for c in cells:
            deepest[c.translevel] = max(deepest.get(c.translevel, 1),
                                        c.local_progress, 1)
        weights = []
        for c in cells:
            if c.translevel == MAP_CELL:
                # Every map node is a branch point, not a distance. Ranking them
                # by map x would starve whichever branch sits leftmost.
                frontier = 1.0
            else:
                frontier = (c.local_progress / deepest[c.translevel]) ** self.frontier_bias
            novelty = 1.0 / (1.0 + c.chosen) ** 0.5
            rarity = 1.0 / (1.0 + barren.get(c.translevel, 0)) ** 0.5
            depth = 1.0 + c.cleared          # prefer deeper game states, gently
            rate = (c.succeeds + 1) / (c.succeeds + c.fails + 2)
            weights.append(max(1e-9, frontier * novelty * rarity * depth * rate ** 1.5))
        pick = random.choices(cells, weights=weights, k=1)[0]
        pick.chosen += 1
        barren[pick.translevel] = barren.get(pick.translevel, 0) + 1
        return pick

    @property
    def best_cell(self) -> WorldCell | None:
        return max(self.cells.values(), key=lambda c: c.progress, default=None)

    def visited(self) -> set:
        """Map nodes we have entered a level from.

        Stored lazily so archives pickled before this existed still load.
        """
        nodes = self.__dict__.get("nodes")
        if nodes is None:
            nodes = self.__dict__["nodes"] = set()
        return nodes

    def barren(self) -> dict:
        """Excursions spent in each translevel since it last yielded anything.

        A level absent from here has never been drawn from, which is the most
        attractive state there is. Stored lazily so older archives still load.
        """
        counts = self.__dict__.get("barren_counts")
        if counts is None:
            counts = self.__dict__["barren_counts"] = {}
        return counts

    @property
    def translevels(self) -> set:
        return {c.translevel for c in self.cells.values()}

    def absorb(self, other: "WorldArchive") -> int:
        """Merge another archive in, on the same rule `consider` uses: the most
        advanced state at each cell, and among equals the cheaper route to it.

        Exploration is single-threaded, so the only way to use more cores is to
        run several explorers and union their findings.
        """
        added = 0
        self.visited().update(other.visited())
        # A level is only barren if it has gone quiet for *every* explorer, so
        # take the shortest dry spell. Absent means "never drawn from", which
        # must not be read as a fresh reset for a level this shard never saw.
        mine = self.barren()
        for tl, dry in other.barren().items():
            mine[tl] = min(mine.get(tl, dry), dry)
        for key, cell in other.cells.items():
            cur = self.cells.get(key)
            if cur is None:
                self.cells[key] = cell
                added += 1
            elif (cell.cleared, -cell.steps) > (cur.cleared, -cur.steps):
                _keep_counters(cell, cur)
                self.cells[key] = cell
                added += 1
            else:
                _keep_counters(cur, cell)
        return added

    def save(self, path: Path) -> None:
        tmp = Path(path).with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(self))
        tmp.replace(path)


class _MainRedirect(pickle.Unpickler):
    """Resolve classes an older archive recorded as living in ``__main__``."""

    def find_class(self, module, name):
        if module == "__main__" and name in ("WorldArchive", "WorldCell"):
            return globals()[name]
        return super().find_class(module, name)


def load_archive(path: str | Path) -> WorldArchive:
    return _MainRedirect(io.BytesIO(Path(path).read_bytes())).load()
