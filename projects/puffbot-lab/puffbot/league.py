"""Who each game is against: a CPU ladder plus self-play against the live policy and
frozen snapshots of it.

The learner owns the ladder and writes `league.json`; workers read it whenever they
pick a new setup. A CPU game is credited to the level the game *actually* reported,
never the one requested: in version one a slider that did not take silently put 10% of
games against a level-0 CPU that never acts.
"""
from __future__ import annotations

import json
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .dolphin import Setup

# Matchup focus. Each CPU character's sampling weight is its configured weight times a
# factor from 0.5 (winning comfortably) to 2.0 (losing), from its recent record near the
# frontier. Every character keeps at least half its base share, so easy matchups stay
# in the mix as regression coverage. Replaces the old rule that switched to 60% Puff
# self-play once the pooled level-9 window passed, which traded Falco/Fox practice for
# dittos (GPT audit + step-1 review, 2026-09-25).
MATCHUP_WINDOW = 30
MATCHUP_MIN_GAMES = 8
FOCUS_FLOOR, FOCUS_CEILING = 0.5, 2.0


@dataclass(frozen=True)
class Opponent:
    kind: str                 # 'cpu' | 'mirror' | 'snapshot' | 'phillip'
    setup: Setup
    snapshot: str | None = None


class Ladder:
    def __init__(self, cfg: Config, state: dict | None = None):
        self.cfg = cfg
        self.frontier = int((state or {}).get('frontier', cfg.cpu_start_level))
        self.windows: dict[int, deque] = {}
        self.history = list((state or {}).get('history', []))
        self.mastered = bool((state or {}).get('mastered', False))
        for level, results in ((state or {}).get('windows') or {}).items():
            self.windows[int(level)] = deque(results, maxlen=cfg.ladder_window)
        self.matchups: dict[str, deque] = {
            name: deque(results, maxlen=MATCHUP_WINDOW)
            for name, results in ((state or {}).get('matchups') or {}).items()}

    def window(self, level: int) -> deque:
        return self.windows.setdefault(int(level), deque(maxlen=self.cfg.ladder_window))

    def record(self, level: int, result: str, opponent: str | None = None) -> str | None:
        """Add one CPU result. Returns 'promoted' / 'demoted' when the frontier moves, and
        'mastered' / 'unmastered' when the top level is (no longer) being beaten."""
        if result not in ('win', 'loss', 'draw'):
            return None
        won = 1.0 if result == 'win' else 0.0
        if opponent and level >= self.frontier - 1:
            self.matchups.setdefault(opponent, deque(maxlen=MATCHUP_WINDOW)).append(won)
        w = self.window(level)
        w.append(won)
        if level != self.frontier or len(w) < self.cfg.ladder_window:
            return None
        rate = sum(w) / len(w)
        if self.frontier == self.cfg.cpu_max_level:
            # Informational only, and re-judged on every window: it no longer changes
            # who the policy plays.
            now = rate >= self.cfg.promote_at
            if now != self.mastered:
                self.mastered = now
                self.history.append({'level': self.frontier, 'from_rate': round(rate, 3), 'mastered': now})
                return 'mastered' if now else 'unmastered'
            if now:
                return None
        if rate >= self.cfg.promote_at and self.frontier < self.cfg.cpu_max_level:
            self.frontier += 1
            self.history.append({'level': self.frontier, 'from_rate': round(rate, 3)})
            self.window(self.frontier).clear()
            return 'promoted'
        if rate <= self.cfg.demote_at and self.frontier > self.cfg.cpu_min_level:
            self.frontier -= 1
            self.history.append({'level': self.frontier, 'from_rate': round(rate, 3)})
            self.window(self.frontier).clear()
            return 'demoted'
        return None

    def rates(self) -> dict:
        return {str(k): {'games': len(w), 'win_rate': round(sum(w) / len(w), 3) if w else None}
                for k, w in sorted(self.windows.items())}

    def to_dict(self) -> dict:
        return {'frontier': self.frontier, 'windows': {str(k): list(v) for k, v in self.windows.items()},
                'history': self.history[-50:], 'mastered': self.mastered,
                'matchups': {k: list(v) for k, v in self.matchups.items()}}

    def selfplay_fraction(self) -> float:
        return self.cfg.selfplay_fraction

    def focus(self, name: str) -> float:
        """0.5 for a matchup won every time, 2.0 for one lost every time; unknown or thin
        records get the maximum, so new characters are sampled until they are measured."""
        w = self.matchups.get(name)
        if not w or len(w) < MATCHUP_MIN_GAMES:
            return FOCUS_CEILING
        need = 1.0 - sum(w) / len(w)
        return FOCUS_FLOOR + (FOCUS_CEILING - FOCUS_FLOOR) * need

    def opponent_weights(self) -> dict[str, float]:
        return {n: round(float(base) * self.focus(n), 4) for n, base in self.cfg.opponents.items()}

    def matchup_table(self) -> dict:
        return {n: {'games': len(self.matchups.get(n, ())),
                    'win_rate': round(sum(self.matchups[n]) / len(self.matchups[n]), 3) if self.matchups.get(n) else None,
                    'focus': round(self.focus(n), 2)}
                for n in self.cfg.opponents}


def write_league(path: Path, frontier: int, snapshots: list[str], selfplay_fraction: float | None = None,
                 opponent_weights: dict | None = None):
    tmp = path.with_name(f'.{path.name}.tmp')
    data = {'frontier': frontier, 'snapshots': snapshots}
    if selfplay_fraction is not None:
        data['selfplay_fraction'] = selfplay_fraction
    if opponent_weights:
        data['opponent_weights'] = opponent_weights
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def read_league(path: Path, cfg: Config) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {'frontier': cfg.cpu_start_level, 'snapshots': []}


def level_around(frontier: int, cfg: Config, rng: np.random.Generator) -> int:
    """Mostly the frontier, some easier games to keep old skills, some harder to probe."""
    offsets = np.array([-2, -1, 0, 1])
    weights = np.array([0.1, 0.2, 0.5, 0.2])
    level = frontier + int(rng.choice(offsets, p=weights))
    return int(min(max(level, cfg.cpu_min_level), cfg.cpu_max_level))


def choose(cfg: Config, league: dict, rng: np.random.Generator) -> Opponent:
    me = cfg.character
    snapshots = [s for s in league.get('snapshots', []) if Path(s).exists()]
    if rng.random() < float(league.get('selfplay_fraction', cfg.selfplay_fraction)):
        if not snapshots or rng.random() < cfg.mirror_fraction:
            return Opponent('mirror', Setup(me, me, 0))
        # Half recent, half anywhere in history: keeps old weaknesses from reopening.
        if rng.random() < 0.5:
            pick = snapshots[-min(3, len(snapshots)):]
            snap = pick[int(rng.integers(len(pick)))]
        else:
            snap = snapshots[int(rng.integers(len(snapshots)))]
        return Opponent('snapshot', Setup(me, me, 0), snap)
    table = league.get('opponent_weights') or cfg.opponents
    names = [n for n in table if n in cfg.opponents] or list(cfg.opponents)
    weights = np.array([max(float(table.get(n, cfg.opponents[n])), 1e-6) for n in names])
    name = names[int(rng.choice(len(names), p=weights / weights.sum()))]
    level = level_around(int(league.get('frontier', cfg.cpu_start_level)), cfg, rng)
    return Opponent('cpu', Setup(me, name, level))


def choose_phillip(cfg: Config, rng: np.random.Generator) -> Opponent:
    names = list(cfg.phillip_characters)
    weights = np.array([max(float(cfg.phillip_characters[n]), 1e-6) for n in names])
    name = names[int(rng.choice(len(names), p=weights / weights.sum()))]
    return Opponent('phillip', Setup(cfg.character, name, 0))


def wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a win rate. Ten games is a wide interval; say so."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
