"""A worker: one emulator, one local copy of the policy, and no waiting on anyone.

Version one stepped every emulator in lockstep through a vectorised environment, so a
single emulator in a menu froze all the others (the "every Fox freezes when one game
ends" bug, and the 17 FPS drop). Here each worker plays at its own pace and ships
fixed-length trajectory fragments to the learner through a queue. It picks up new
weights between fragments. A worker that crashes or wedges is restarted by the learner
while the rest keep playing.
"""
from __future__ import annotations

import os
import queue as queue_mod
import signal
import threading
import time
import traceback
from pathlib import Path

import melee
import numpy as np
import torch

from . import actions, league, observation, phillip, reward
from .config import Config
from .dolphin import Emulator, EmulatorError
from .model import Actor, load_policy

STATUS_EVERY = 2.0
IN_GAME = (melee.Menu.IN_GAME,)


class Side:
    """One port the policy (live or frozen) is playing."""

    def __init__(self, port: int, cfg: Config, learn: bool, net: Actor):
        self.port = port
        self.learn = learn
        self.net = net
        self.exec = actions.Executor(cfg.act_every, cfg.auto_lcancel, cfg.auto_tech)
        self.prev_action = -1
        self.open: dict | None = None

    def accumulate(self, r: float, gamma: float):
        if self.open is not None:
            self.open['r'] += self.open['disc'] * r
            self.open['disc'] *= gamma


class Buffer:
    """Closed decisions for one port, waiting to become a fragment."""

    def __init__(self):
        self.rows: list[dict] = []

    def __len__(self):
        return len(self.rows)


class Worker:
    def __init__(self, index: int, cfg: Config, run_dir: Path, queue, stop, seed: int,
                 evaluate: dict | None = None):
        self.index = index
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.queue = queue
        self.stop = stop
        self.rng = np.random.default_rng(seed)
        self.evaluate = evaluate            # {'checkpoint':..., 'setups': [...], 'greedy': bool}
        home = self.run_dir / 'workers' / f'w{index:02d}'
        home.mkdir(parents=True, exist_ok=True)
        replays = home / 'replays' if cfg.save_replays else None
        # The visible emulators are the last ones: Phillip takes the first, and a window
        # should show the ordinary mix of games.
        self.emu = Emulator(cfg, index, home / 'User', visible=index >= cfg.workers - cfg.visible_workers,
                            replay_dir=replays, log_path=home / 'dolphin.log')
        self.actor = Actor(cfg.hidden)
        self.policy_path = self.run_dir / 'policy.pt'
        self.policy_stamp = None
        self.frozen: dict[str, Actor] = {}
        self.buffers = {1: Buffer(), 2: Buffer()}
        # Where the current game's decisions start in each buffer. A game that is cut
        # short (emulator failure, safety cap) has its unfinished rows discarded: it has
        # no true ending to learn from, and pretending it ended would teach a false one.
        self.game_row0 = {1: 0, 2: 0}
        self.cut_games = 0
        self.gamma = cfg.gamma
        self.phase = 'starting'
        self.frames = self.games = self.errors = self.dropped = 0
        self.last_error = None
        self.setup_label = ''
        self.status_at = 0.0
        self.status_frames = 0
        self.fps = 0.0
        self.menu_seconds = None
        self.batch = None          # evaluation batch being played, if any
        self.batch_games = 0
        self.game_stats = None
        self.sides: list[Side] = []
        self.phillip: phillip.Phillip | None = None
        self.phillip_log = home / 'phillip.log'

    # --------------------------------------------------------------- messaging
    def send(self, msg: dict, block: float = 1.0) -> bool:
        try:
            self.queue.put(msg, timeout=block)
            return True
        except Exception:
            self.dropped += 1
            return False

    def send_reliable(self, msg: dict) -> bool:
        """For messages the coordinator cannot do without (batch start/end, results):
        keep trying until delivered or the run is stopping."""
        while True:
            try:
                self.queue.put(msg, timeout=1.0)
                return True
            except Exception:
                if self.stop.is_set():
                    self.dropped += 1
                    return False

    def status(self, force=False):
        now = time.monotonic()
        if not force and now - self.status_at < STATUS_EVERY:
            return
        span = max(now - self.status_at, 1e-6)
        if self.status_at:
            self.fps = (self.frames - self.status_frames) / span
        self.status_at, self.status_frames = now, self.frames
        self.send({'kind': 'status', 'worker': self.index, 'phase': self.phase, 'fps': round(self.fps, 1),
                   'frames': self.frames, 'games': self.games, 'errors': self.errors,
                   'launches': self.emu.launches, 'setup': self.setup_label, 'version': self.actor.version,
                   'last_error': self.last_error, 'dropped': self.dropped, 'pid': os.getpid(),
                   'visible': self.emu.visible, 'menu_seconds': self.menu_seconds,
                   'cut_games': self.cut_games, 'batch': self.batch,
                   'phillip_ms': round(1000 * self.phillip.step_seconds / self.phillip.frames, 2)
                   if self.phillip is not None and self.phillip.frames else None}, block=0.2)

    # ------------------------------------------------------------------ policy
    def refresh_policy(self, wait: bool = False):
        if self.evaluate:
            return
        deadline = time.monotonic() + 120
        while True:
            try:
                stamp = self.policy_path.stat().st_mtime_ns
            except FileNotFoundError:
                stamp = None
            if stamp is not None and stamp != self.policy_stamp:
                try:
                    payload = load_policy(self.policy_path)
                    self.actor.load_state(payload['state'], payload['version'])
                    self.policy_stamp = stamp
                    return
                except Exception:
                    pass  # mid-replace on a slow disk; try again shortly
            if not wait or self.actor.version >= 0 or self.stop.is_set():
                return
            if time.monotonic() > deadline:
                raise RuntimeError('No policy.pt from the learner after 120s.')
            time.sleep(0.2)

    def frozen_net(self, path: str) -> Actor:
        if path not in self.frozen:
            if len(self.frozen) >= 4:
                self.frozen.pop(next(iter(self.frozen)))
            payload = load_policy(Path(path))
            net = Actor(payload.get('hidden', self.cfg.hidden))
            net.load_state(payload['state'], payload['version'])
            self.frozen[path] = net
        return self.frozen[path]

    # --------------------------------------------------------------- decisions
    def close(self, side: Side, terminal: bool):
        step, side.open = side.open, None
        if step is None or not side.learn:
            return
        step['discount'] = 0.0 if terminal else step['disc']
        step['frames'] = side.exec.t
        self.buffers[side.port].rows.append(step)

    def emit(self, port: int, boot):
        rows = self.buffers[port].rows
        floats, ids, mask = boot
        frag = {
            'kind': 'frag', 'worker': self.index, 'port': port,
            'version': min(r['version'] for r in rows),
            'frames': int(sum(r['frames'] for r in rows)),
            'floats': np.stack([r['floats'] for r in rows] + [floats]),
            'ids': np.stack([r['ids'] for r in rows] + [ids]),
            'masks': np.stack([r['mask'] for r in rows] + [mask]),
            'actions': np.array([r['action'] for r in rows], np.int64),
            'logp': np.array([r['logp'] for r in rows], np.float32),
            'rewards': np.array([r['r'] for r in rows], np.float32),
            'discounts': np.array([r['discount'] for r in rows], np.float32),
        }
        self.buffers[port].rows = []
        self.game_row0[port] = 0
        self.send(frag, block=2.0)
        self.refresh_policy()

    def decide(self, g):
        need = [s for s in self.sides if s.open is None or s.exec.done]
        if not need:
            return
        by_net: dict[int, list] = {}
        for s in need:
            self.close(s, terminal=False)
            floats, ids = observation.encode(g, s.port, s.prev_action, self.cfg.max_game_frames)
            mask = actions.legal_mask(g.players[s.port])
            if s.learn and len(self.buffers[s.port]) >= self.cfg.fragment:
                self.emit(s.port, (floats, ids, mask))
            by_net.setdefault(id(s.net), []).append((s, floats, ids, mask))
        greedy = bool(self.evaluate and self.evaluate.get('greedy'))
        for group in by_net.values():
            net = group[0][0].net
            acts, logps = net.act(np.stack([x[1] for x in group]), np.stack([x[2] for x in group]),
                                  np.stack([x[3] for x in group]), self.rng, greedy=greedy)
            for (s, floats, ids, mask), a, lp in zip(group, acts, logps):
                s.exec.start(int(a))
                s.prev_action = int(a)
                s.open = {'floats': floats, 'ids': ids, 'mask': mask, 'action': int(a), 'logp': float(lp),
                          'version': net.version, 'r': 0.0, 'disc': 1.0}
                if s.port == 1 and self.game_stats is not None:
                    self.game_stats.actions[int(a)] += 1

    # -------------------------------------------------------------------- play
    def play(self, opp: league.Opponent, games_wanted: int | None = None):
        cfg = self.cfg
        limit = games_wanted or cfg.games_per_setup
        self.setup_label = (f'{opp.setup.p2} CPU {opp.setup.cpu_level}' if opp.kind == 'cpu'
                            else 'mirror (self-play)' if opp.kind == 'mirror'
                            else f'Phillip {opp.setup.p2}' if opp.kind == 'phillip'
                            else f'snapshot {Path(opp.snapshot).stem}')
        bot = self.phillip_client() if opp.kind == 'phillip' else None
        self.phase = 'booting'
        self.status(force=True)
        self.emu.launch()
        self.phase = 'menus'
        t0 = time.monotonic()
        g = self.emu.enter_match(opp.setup)
        menu_seconds = time.monotonic() - t0
        self.sides = [Side(1, self.cfg, not self.evaluate, self.actor)]
        if opp.kind == 'mirror':
            self.sides.append(Side(2, cfg, not self.evaluate, self.actor))
        elif opp.kind == 'snapshot':
            self.sides.append(Side(2, cfg, False, self.frozen_net(opp.snapshot)))
        controlled = {s.port for s in self.sides}
        self.phase = 'playing'
        games = 0
        prev = None
        ended = False
        self.game_stats = reward.GameStats(1)
        started = time.monotonic()
        cpu_seen = int(g.players[2].cpu_level) if 2 in g.players else 0
        self.mark_game_start()
        try:
            while games < limit and not self.stop.is_set():
                live = (g.menu_state in IN_GAME and 1 in g.players and 2 in g.players
                        and g.frame >= 0 and not ended)
                if live:
                    self.decide(g)
                    for s in self.sides:
                        actions.write(self.emu.pads[s.port - 1], s.exec.next_input(g.players[s.port]))
                else:
                    for s in self.sides:
                        self.emu.pads[s.port - 1].release_all()
                        s.exec.released()
                for p in (1, 2):
                    if p in controlled:
                        continue
                    if bot is not None and p == 2 and g.menu_state in IN_GAME and 1 in g.players and 2 in g.players:
                        # Every frame, countdown included: Phillip resets on frame -123 and
                        # carries its own recurrent state and reaction delay.
                        phillip.write(self.emu.pads[1], bot.step(g))
                    else:
                        self.emu.pads[p - 1].release_all()
                g2 = self.emu.frame()
                self.frames += 1
                if g2.menu_state == melee.Menu.SUDDEN_DEATH:
                    # Time ran out with stocks tied. Score it as a timeout and end the
                    # setup: after Sudden Death the console goes to a results screen that
                    # Instant Match does not skip, and a fresh launch is quicker.
                    if not ended and prev is not None:
                        self.finish(prev, opp, cpu_seen, started, timeout=True)
                    return
                if g2.menu_state not in IN_GAME or 1 not in g2.players or 2 not in g2.players:
                    if ended:
                        return   # the game was already scored; the console just left it
                    raise EmulatorError(f'Left the match unexpectedly ({g2.menu_state.name}).')
                if prev is not None and g2.frame < prev.frame:
                    # Instant Match started the next game.
                    if not ended:
                        self.finish(prev, opp, cpu_seen, started, timeout=True)
                    games += 1
                    ended, prev = False, None
                    self.game_stats = reward.GameStats(1)
                    started = time.monotonic()
                    cpu_seen = int(g2.players[2].cpu_level)
                    self.mark_game_start()
                elif g2.frame >= 0 and not ended:
                    if prev is not None:
                        over = int(g2.players[1].stock) == 0 or int(g2.players[2].stock) == 0
                        for s in self.sides:
                            r = reward.frame_reward(prev, g2, s.port, self.gamma)
                            if over:
                                r += reward.result_reward(reward.result_of(g2, s.port))
                            s.accumulate(r, self.gamma)
                        self.game_stats.update(prev, g2)
                        if over:
                            self.finish(g2, opp, cpu_seen, started)
                            ended = True
                        elif g2.frame >= cfg.max_game_frames:
                            # The game's own 8:00 timer should have ended this already.
                            self.cut(g2, opp, cpu_seen, started, 'capped')
                            return
                    prev = g2
                g = g2
                self.status()
        except EmulatorError:
            self.cut(prev if not ended else None, opp, cpu_seen, started, 'interrupted')
            raise
        finally:
            self.sides = []
            # Port 2 only learns in mirror games; its leftover rows would be many policy
            # versions stale by the next mirror setup and dropped by the learner anyway.
            self.buffers[2].rows = []
            self.menu_seconds = round(menu_seconds, 2)

    def mark_game_start(self):
        self.game_row0 = {p: len(self.buffers[p].rows) for p in (1, 2)}

    def finish(self, last, opp, cpu_seen, started, timeout=False):
        """The game ended by the game's rules: close every side's last decision as terminal."""
        for s in self.sides:
            if timeout and s.open is not None:
                s.open['r'] += s.open['disc'] * reward.result_reward(reward.result_of(last, s.port))
            self.close(s, terminal=True)
        # This game's rows are final. Move the boundary past them at once, so a crash
        # before the next game starts cannot trim them (GPT step-1 review, R3).
        self.mark_game_start()
        self.games += 1
        self.report(last, opp, cpu_seen, started, reward.result_of(last, 1), timeout=timeout)

    def cut(self, last, opp, cpu_seen, started, reason: str):
        """The game did not end by the game's rules. Keep its record, discard its
        unfinished training data (rows already sent in fragments were complete steps)."""
        for s in self.sides:
            s.open = None
        if last is None:
            return          # no game in progress (between games): nothing to trim
        for p in (1, 2):
            self.buffers[p].rows = self.buffers[p].rows[:self.game_row0[p]]
        self.cut_games += 1
        self.report(last, opp, cpu_seen, started, reason)

    def report(self, last, opp, cpu_seen, started, result, timeout=False):
        msg = {'kind': 'game', 'worker': self.index, 'opponent_kind': opp.kind,
               'opponent': opp.setup.p2, 'cpu_requested': opp.setup.cpu_level,
               'cpu_level': cpu_seen if opp.kind == 'cpu' else 0,
               'snapshot': Path(opp.snapshot).stem if opp.snapshot else None,
               'result': result, 'timeout': timeout, 'stocks_left': int(last.players[1].stock),
               'opp_stocks_left': int(last.players[2].stock), 'version': self.actor.version,
               'wall_seconds': round(time.monotonic() - started, 1), 'time': time.time(),
               **self.game_stats.summary()}
        if self.evaluate:
            self.batch_games += 1
            msg.update(checkpoint=self.evaluate.get('checkpoint'), batch=self.batch, game_no=self.batch_games)
            self.send_reliable(msg)
        else:
            self.send(msg, block=2.0)

    # -------------------------------------------------------------------- loop
    def loop(self):
        self.game_stats = None
        if self.evaluate:
            payload = load_policy(Path(self.evaluate['checkpoint']))
            # Build the network the checkpoint was trained with, not the local config's.
            self.actor = Actor(int(payload.get('hidden', self.cfg.hidden)))
            self.actor.load_state(payload['state'], payload['version'])
        else:
            self.refresh_policy(wait=True)
        fails = 0
        while not self.stop.is_set():
            item = self.next_setup()
            if item is None:
                break
            opp, games = item['opponent'], item['games']
            before = self.games
            failed = None
            if self.evaluate:
                self.batch, self.batch_games = item['batch'], 0
                self.send_reliable({'kind': 'setup_start', 'worker': self.index, 'batch': self.batch})
            try:
                self.play(opp, games)
                fails = 0
            except EmulatorError as exc:
                fails += 1
                self.errors += 1
                failed = self.last_error = str(exc)[:300]
                if isinstance(exc, phillip.PhillipError):
                    self.close_phillip()      # a fresh one is started for the next setup
                self.phase = 'recovering'
                self.send({'kind': 'error', 'worker': self.index, 'error': self.last_error,
                           'setup': self.setup_label, 'time': time.time()}, block=0.5)
                self.emu.stop()
            if self.evaluate:
                # The coordinator decides what to replay; a worker never repeats games itself.
                self.send_reliable({'kind': 'setup_done', 'worker': self.index, 'batch': self.batch,
                                    'opponent': opp.setup.p2, 'requested': games,
                                    'completed': self.games - before, 'error': failed})
                self.batch = None
            if failed:
                self.status(force=True)
                end = time.monotonic() + min(30.0, 1.5 ** fails)
                while time.monotonic() < end and not self.stop.is_set():
                    time.sleep(0.1)
        self.phase = 'stopped'
        self.status(force=True)

    def phillip_client(self) -> 'phillip.Phillip':
        """This worker's Phillip, started on first use and kept across setups."""
        if self.phillip is not None and not self.phillip.alive:
            self.phillip.close()
            self.phillip = None
        if self.phillip is None:
            self.phase = 'loading Phillip'
            self.status(force=True)
            self.phillip = phillip.Phillip(self.cfg.phillip_python, self.cfg.phillip_model, self.phillip_log)
        return self.phillip

    def close_phillip(self):
        if self.phillip is not None:
            self.phillip.close()
            self.phillip = None

    def next_setup(self):
        """{'opponent', 'games'[, 'batch']} to play next, or None when there is nothing left."""
        if not self.evaluate:
            if self.index < self.cfg.phillip_workers:
                opp = league.choose_phillip(self.cfg, self.rng)
            else:
                opp = league.choose(self.cfg, league.read_league(self.run_dir / 'league.json', self.cfg), self.rng)
            return {'opponent': opp, 'games': self.cfg.games_per_setup}
        while not self.stop.is_set():
            try:
                return self.evaluate['work'].get(timeout=0.5)   # None means finished
            except queue_mod.Empty:
                self.phase = 'waiting'
                self.status()
        return None


def run(index: int, cfg_dict: dict, run_dir: str, queue, stop, seed: int, evaluate: dict | None = None):
    """Process entry point."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # the learner owns shutdown
    torch.set_num_threads(1)
    cfg = Config(**cfg_dict)
    code = 0
    worker = None
    try:
        worker = Worker(index, cfg, Path(run_dir), queue, stop, seed, evaluate)
        worker.loop()
    except Exception as exc:
        code = 1
        try:
            queue.put({'kind': 'fatal', 'worker': index, 'error': f'{type(exc).__name__}: {exc}',
                       'traceback': traceback.format_exc()[-3000:], 'time': time.time()}, timeout=1)
        except Exception:
            pass
    finally:
        if worker is not None:
            worker.emu.stop()
            worker.close_phillip()
        # Flush queued messages, but never let a full pipe hold this process open, and
        # never let multiprocessing's atexit join a child that will not exit: that join
        # is exactly how version one wedged a whole run for hours without an error.
        try:
            queue.close()
            t = threading.Thread(target=queue.join_thread, daemon=True)
            t.start()
            t.join(timeout=3)
        except Exception:
            pass
        os._exit(code)
