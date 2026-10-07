"""Fixed-protocol evaluation: one checkpoint, fixed opponents and level, N games each.

Training win rates are not a score: the ladder level moves, self-play games mix in and
the policy changes under them. This plays a frozen checkpoint against a fixed field,
spreads the games over parallel emulators, and reports every rate with its 95%
interval. Version one's ten-game samples had intervals 40 points wide; the report makes
that visible instead of letting it pass for a result.

What makes a report trustworthy (GPT audit, 2026-09-25):

* Quotas are enforced. Work is handed out a few games at a time; a batch that ends
  short (a timeout into Sudden Death, a crashed emulator) is topped up with another,
  up to a retry limit. The report says `complete` only when every opponent has its
  full count, otherwise `incomplete`, `failed` or `cancelled`. It never says finished
  with games missing.
* Only real results count: a game must end by the game's rules, at the requested CPU
  level. Everything else is kept in `rejected` with the reason.
* The checkpoint is copied once, fingerprinted (SHA-256), and every worker loads the
  copy, so a training run overwriting the original cannot change what was measured.
  The report stores the protocol: config, the checkpoint's contract, code commit,
  Dolphin build, ISO and library versions.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import platform
import queue as queue_mod
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from .config import ROOT, RUNS, Config
from .dolphin import Setup
from .league import Opponent, wilson

COMPLETED = ('win', 'loss', 'draw')
PER_SETUP = 5           # games per batch handed to a worker
RETRY_FACTOR = 3        # wasted batches allowed per opponent, as a multiple of those planned
MAX_RESTARTS = 2        # per worker slot
STALL_SECONDS = 240.0   # a worker holding a batch this long without news or frames is killed


class Quota:
    """Which games still need playing. Pure bookkeeping, testable without emulators.

    The retry budget counts only *wasted* batches, ones that produced no countable
    game. A batch cut short after a legitimate result (a timeout into Sudden Death ends
    the setup) made progress and costs nothing; the shortfall is simply re-planned.
    (GPT step-1 review R2: six batches of one valid timeout each used to exhaust a
    ten-game quota at 6/10.)"""

    def __init__(self, opponents: list[str], games: int, per_setup: int = PER_SETUP,
                 retry_factor: int = RETRY_FACTOR):
        self.want = {o: games for o in opponents}
        self.done = {o: 0 for o in opponents}
        self.outstanding = {o: 0 for o in opponents}   # games in batches handed out
        self.issued = {o: 0 for o in opponents}
        self.wasted = {o: 0 for o in opponents}
        self.per_setup = max(1, per_setup)
        self.max_wasted = {o: retry_factor * math.ceil(games / self.per_setup) for o in opponents}

    def _short(self, o) -> int:
        return self.want[o] - self.done[o] - self.outstanding[o]

    def plan(self) -> list[tuple[str, int]]:
        """New batches needed to cover what is neither done nor in progress."""
        out = []
        for o in self.want:
            while self._short(o) > 0 and self.wasted[o] < self.max_wasted[o]:
                n = min(self.per_setup, self._short(o))
                self.outstanding[o] += n
                self.issued[o] += 1
                out.append((o, n))
        return out

    def accept(self, game: dict, level: int) -> str | None:
        """Count one finished game, or return why it does not count."""
        o = game.get('opponent')
        if o not in self.want:
            return 'opponent not requested'
        if game.get('result') not in COMPLETED:
            return f"game {game.get('result')}"
        if game.get('opponent_kind') == 'cpu' and int(game.get('cpu_level', -1)) != level:
            return f"played at CPU {game.get('cpu_level')}, not {level}"
        if self.done[o] >= self.want[o]:
            return 'quota already met'
        self.done[o] += 1
        return None

    def batch_finished(self, o: str, requested: int, accepted: int = 0):
        if o in self.outstanding:
            self.outstanding[o] = max(0, self.outstanding[o] - int(requested))
            if accepted <= 0:
                self.wasted[o] += 1

    @property
    def complete(self) -> bool:
        return all(self.done[o] >= self.want[o] for o in self.want)

    @property
    def exhausted(self) -> bool:
        """Quotas unmet, nothing in progress, and the retry budget is spent."""
        return (not self.complete and not any(self.outstanding.values())
                and not any(self._short(o) > 0 and self.wasted[o] < self.max_wasted[o] for o in self.want))


class Coordinator:
    """Hands numbered batches to specific workers and never loses one.

    Each batch is leased to one worker. If that worker dies, or holds a batch while
    sending nothing (or making no frame progress) for `stall` seconds, the lease is
    reclaimed, the batch's unplayed games are re-planned and the worker is restarted up
    to `max_restarts` times. Results are keyed by (batch, game number), so a repeated
    delivery is never counted twice. (GPT step-1 review R1: a worker dying with a batch
    while another waited for work used to hang the evaluation forever.)

    `spawn(i)` returns (process, inbox); processes need is_alive/kill/join, inboxes
    need put. Everything else is plain bookkeeping, so tests drive it with fakes."""

    def __init__(self, quota: Quota, level: int, make_item, spawn, workers: int, *,
                 max_restarts: int = MAX_RESTARTS, stall: float = STALL_SECONDS,
                 deadline: float | None = None, clock=time.monotonic):
        self.quota, self.level, self.make_item, self.spawn = quota, level, make_item, spawn
        self.workers, self.max_restarts, self.stall, self.clock = workers, max_restarts, stall, clock
        self.deadline = None if deadline is None else clock() + deadline
        self.procs, self.inbox, self.lease = {}, {}, {}
        self.seen, self.progress, self.frames = {}, {}, {}
        self.restarts = defaultdict(int)
        self.retired = set()
        self.batches: dict[int, dict] = {}
        self.backlog: list[int] = []
        self.counted_keys = set()
        self.results, self.rejected, self.errors = [], [], []
        self.next_id = 0
        self.timed_out = False

    # ------------------------------------------------------------- scheduling
    def start(self):
        self._refill()
        for i in range(self.workers):
            self._launch(i)
        self._assign()

    def _refill(self):
        for o, n in self.quota.plan():
            self.next_id += 1
            self.batches[self.next_id] = {'opponent': o, 'games': n, 'accepted': 0, 'worker': None, 'done': False}
            self.backlog.append(self.next_id)

    def _launch(self, i):
        self.procs[i], self.inbox[i] = self.spawn(i)
        now = self.clock()
        self.seen[i] = self.progress[i] = now
        self.frames.pop(i, None)
        self.lease.pop(i, None)

    def _assign(self):
        for i, p in self.procs.items():
            if not self.backlog:
                return
            if i in self.retired or self.lease.get(i) is not None or not p.is_alive():
                continue
            bid = self.backlog.pop(0)
            b = self.batches[bid]
            b['worker'] = i
            self.inbox[i].put(self.make_item(bid, b['opponent'], b['games']))
            self.lease[i] = bid
            self.seen[i] = self.progress[i] = self.clock()

    def _release(self, bid):
        b = self.batches.get(bid)
        if b is None or b['done']:
            return
        b['done'] = True
        self.quota.batch_finished(b['opponent'], b['games'], b['accepted'])

    # --------------------------------------------------------------- messages
    def handle(self, msg: dict) -> str | None:
        """Apply one worker message. Returns a progress line worth printing, if any."""
        i = msg.get('worker')
        now = self.clock()
        if i is not None:
            self.seen[i] = now
        kind = msg.get('kind')
        if kind == 'status':
            if msg.get('frames') != self.frames.get(i):
                self.frames[i] = msg.get('frames')
                self.progress[i] = now
        elif kind == 'game':
            self.progress[i] = now
            key = (msg.get('batch'), msg.get('game_no'))
            if key[0] is not None and key in self.counted_keys:
                self.rejected.append({**msg, 'rejected': 'duplicate delivery'})
                return None
            self.counted_keys.add(key)
            why = self.quota.accept(msg, self.level)
            if why:
                self.rejected.append({**msg, 'rejected': why})
                return f"not counted ({why}): {msg.get('opponent')} {msg.get('result')}"
            self.results.append(msg)
            b = self.batches.get(msg.get('batch'))
            if b is not None:
                b['accepted'] += 1
            who = 'Phillip' if msg.get('opponent_kind') == 'phillip' else f"CPU {msg.get('cpu_level')}"
            return (f"[{len(self.results)}/{sum(self.quota.want.values())}] {msg['opponent']} "
                    f"{who}: {msg['result']} ({msg.get('stocks_left')}-{msg.get('opp_stocks_left')} stocks)")
        elif kind == 'setup_done':
            bid = msg.get('batch')
            if self.lease.get(i) == bid:
                self.lease[i] = None
            self._release(bid)
            self._refill()
            self._assign()
        elif kind in ('error', 'fatal'):
            self.errors.append(msg)
            return f"worker {i}: {msg.get('error')}"
        return None

    # ----------------------------------------------------------- supervision
    def tick(self):
        now = self.clock()
        for i, p in list(self.procs.items()):
            if i in self.retired:
                continue
            if p.is_alive():
                bid = self.lease.get(i)
                if bid is not None and (now - self.seen[i] > self.stall or now - self.progress[i] > self.stall):
                    self.errors.append({'worker': i, 'error': f'batch {bid}: no progress for {self.stall:.0f}s; worker killed'})
                    p.kill()
                continue
            bid = self.lease.pop(i, None)
            if bid is not None:
                self.errors.append({'worker': i, 'error': f'worker died holding batch {bid}; its games are re-planned'})
                self._release(bid)
            try:
                p.join(timeout=1)
            except Exception:
                pass
            if self.restarts[i] < self.max_restarts and not self.quota.complete:
                self.restarts[i] += 1
                self._launch(i)
            else:
                self.retired.add(i)
        self._refill()
        self._assign()
        if self.deadline is not None and now > self.deadline:
            self.timed_out = True

    def state(self) -> str | None:
        if self.quota.complete:
            return 'complete'
        if self.quota.exhausted or self.timed_out:
            return 'incomplete'
        if self.procs and all(i in self.retired for i in self.procs):
            return 'incomplete' if self.results else 'failed'
        return None

    def shutdown(self):
        for i, box in self.inbox.items():
            try:
                box.put(None)
            except Exception:
                pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ''


def protocol(cfg: Config, payload: dict, notes: list[str]) -> dict:
    import torch
    iso = Path(cfg.iso)
    commit = _run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'])
    dirty = bool(_run(['git', '-C', str(ROOT), 'status', '--porcelain', '--', 'puffbot']))
    try:
        from importlib.metadata import version as dist_version
        libmelee = dist_version('melee')
    except Exception:
        libmelee = None
    return {
        'config': cfg.to_dict(),
        'contract': payload.get('contract'), 'contract_notes': notes,
        'code': {'commit': commit or None, 'uncommitted_changes': dirty},
        'dolphin': {'path': cfg.dolphin, 'version': _run([cfg.dolphin, '--version'])},
        'iso': {'path': str(iso), 'bytes': iso.stat().st_size if iso.exists() else None},
        'versions': {'python': sys.version.split()[0], 'torch': torch.__version__,
                     'libmelee': libmelee, 'platform': platform.platform()},
    }


PORT_OFFSETS = {'eval': 100, 'watch': 150, 'scorecard': 200}   # training uses base_port + 0..15


def evaluate(cfg: Config, checkpoint: str, opponents: list[str], level: int, games: int,
             greedy: bool = False, out: str | None = None, label: str = 'eval',
             progress=print, out_dir: str | None = None, max_hours: float | None = None,
             kind: str = 'cpu') -> dict:
    from . import contract
    from .learner import _reap_dolphins, _worker_entry
    from .model import load_policy
    src = Path(checkpoint).resolve()
    stamp = time.strftime('%Y%m%d-%H%M%S')
    run_dir = Path(out_dir) if out_dir else RUNS / 'evals' / f'{stamp}-{label}'
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen = run_dir / 'checkpoint.pt'
    shutil.copy2(src, frozen)
    payload = load_policy(frozen)
    saved_every = contract.upgrade(payload.get('contract')).get('act_every')
    if saved_every:
        cfg.act_every = int(saved_every)        # play it at the timing it was trained with
    notes = contract.check(payload.get('migrated_from') or payload.get('contract'), cfg.act_every,
                           what=src.name, migrate=True)
    # Separate Slippi ports so an evaluation can run beside a training run, and a
    # watch window or an automatic scorecard beside a dashboard evaluation.
    cfg.base_port += PORT_OFFSETS.get(label, PORT_OFFSETS['eval'])

    if kind == 'phillip':
        from .phillip import SUPPORTED
        cfg.phillip_workers = 0          # evaluation workers start Phillip on their first batch
        missing = [p for p in (cfg.phillip_python, cfg.phillip_model) if not Path(p).exists()]
        unsupported = [o for o in opponents if o not in SUPPORTED]
        if missing or unsupported:
            raise ValueError(f'Cannot evaluate against Phillip: missing {missing}, unsupported {unsupported}.')
    quota = Quota(opponents, games, per_setup=min(games, PER_SETUP))
    ctx = mp.get_context('spawn')
    q = ctx.Queue(maxsize=1024)
    stop = ctx.Event()
    generation = defaultdict(int)

    def spawn(i):
        inbox = ctx.Queue()
        seed = cfg.seed + i + 7919 * generation[i]
        generation[i] += 1
        plan = {'checkpoint': str(frozen), 'work': inbox, 'greedy': greedy}
        p = ctx.Process(target=_worker_entry, args=(i, cfg.to_dict(), str(run_dir), q, stop, seed, plan))
        p.start()
        time.sleep(0.4)          # stagger boots
        return p, inbox

    if kind not in ('cpu', 'phillip'):
        raise ValueError("kind must be 'cpu' or 'phillip'")

    def make_item(bid, o, n):
        setup = Setup(cfg.character, o, level if kind == 'cpu' else 0)
        return {'batch': bid, 'opponent': Opponent(kind, setup), 'games': n}

    status = {'state': 'running', 'label': label, 'checkpoint': str(src), 'checkpoint_copy': str(frozen),
              'checkpoint_sha256': sha256(frozen), 'checkpoint_version': payload.get('version'),
              'checkpoint_frames': payload.get('frames'), 'checkpoint_hidden': payload.get('hidden'),
              'kind': kind, 'level': level if kind == 'cpu' else None, 'games_each': games,
              'opponents': opponents, 'greedy': greedy,
              'started': time.time(), 'protocol': protocol(cfg, payload, notes)}
    for note in notes:
        progress(f'note: {note}')
    n_workers = max(1, min(cfg.workers, len(opponents) * math.ceil(games / quota.per_setup)))
    coord = Coordinator(quota, level, make_item, spawn, n_workers,
                        deadline=None if max_hours is None else max_hours * 3600)
    _write(run_dir, status, coord)          # a running report exists before any worker does
    try:
        coord.start()
        while True:
            try:
                line = coord.handle(q.get(timeout=0.5))
                if line:
                    progress(line)
                    _write(run_dir, status, coord)
            except queue_mod.Empty:
                pass
            coord.tick()
            state = coord.state()
            if state:
                status['state'] = state
                break
    except KeyboardInterrupt:
        status['state'] = 'cancelled'
    finally:
        coord.shutdown()
        stop.set()
        deadline = time.monotonic() + 10
        procs = list(coord.procs.values())
        while time.monotonic() < deadline and any(p.is_alive() for p in procs):
            time.sleep(0.1)
        for p in procs:
            if p.is_alive():
                p.kill()
            p.join(timeout=2)
        _reap_dolphins(run_dir)
        q.cancel_join_thread()
        for box in coord.inbox.values():
            box.cancel_join_thread()
    status['finished'] = time.time()
    if status['state'] == 'running':
        status['state'] = 'cancelled'
    report = _write(run_dir, status, coord)
    if out:
        Path(out).write_text(json.dumps(report, indent=2, default=float))
    return report


def summarize(games: list[dict]) -> dict:
    n = len(games)
    wins = sum(g['result'] == 'win' for g in games)
    lo, hi = wilson(wins, n)
    lost = sum(g['stocks_lost'] for g in games)
    sd = sum(g['self_destructs'] for g in games)
    rests = sum(g['rests'] for g in games)
    return {
        'games': n, 'wins': wins, 'win_rate': round(wins / n, 3) if n else None,
        'win_rate_95ci': [round(lo, 3), round(hi, 3)],
        'stocks_taken_per_game': round(sum(g['stocks_taken'] for g in games) / n, 2) if n else None,
        'stocks_lost_per_game': round(lost / n, 2) if n else None,
        'damage_dealt_per_game': round(sum(g['damage_dealt'] for g in games) / n, 1) if n else None,
        'damage_received_per_game': round(sum(g['damage_received'] for g in games) / n, 1) if n else None,
        'self_destruct_share': round(sd / lost, 3) if lost else None,
        'rests_per_game': round(rests / n, 2) if n else None,
        'rest_hit_rate': round(sum(g['rest_hits'] for g in games) / rests, 3) if rests else None,
        'techs': sum(g.get('techs', 0) for g in games),
        'missed_techs': sum(g.get('missed_techs', 0) for g in games),
        'offstage_airdodges': sum(g.get('offstage_airdodges', 0) for g in games),
        'aerial_landing_lag': (round(sum(g.get('aerial_landing_frames', 0) for g in games) /
                                     max(1, sum(g.get('aerial_landings', 0) for g in games)), 1)),
    }


def _write(run_dir: Path, status: dict, coord: 'Coordinator') -> dict:
    quota = coord.quota
    by = defaultdict(list)
    for g in coord.results:
        by[g['opponent']].append(g)
    report = dict(status, overall=summarize(coord.results),
                  by_opponent={o: summarize(by[o]) for o in quota.want if by[o]},
                  counted={o: quota.done[o] for o in quota.want}, requested=dict(quota.want),
                  batches_issued=dict(quota.issued), batches_wasted=dict(quota.wasted),
                  worker_restarts=dict(coord.restarts),
                  errors=[e.get('error') for e in coord.errors][-20:], games=coord.results,
                  rejected=[{k: r.get(k) for k in ('opponent', 'result', 'cpu_level', 'rejected', 'worker', 'batch')}
                            for r in coord.rejected])
    tmp = run_dir / '.report.tmp'
    tmp.write_text(json.dumps(report, indent=2, default=float))
    tmp.replace(run_dir / 'report.json')
    return report
