"""The learner: trains on fragments as they arrive and supervises the workers.

It is the main process of a training run. Workers are independent processes it starts,
watches and restarts; if one dies the others never notice. The learner publishes new
weights to `policy.pt` after every update and writes `status.json` once a second for the
dashboard. `control.json` ({"stop": true}) or SIGTERM ends the run cleanly with a
checkpoint.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import queue as queue_mod
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import torch

from . import actions, contract, league
from .config import ROOT, RUNS, Config
from .model import Policy, load_policy, save_policy
from .vtrace import ppo_loss, vtrace

STATUS_EVERY = 1.0
COMPLETED = ('win', 'loss', 'draw')
RESTART_BACKOFF = (2, 5, 15, 30, 60)
# Longer than a worker's worst legitimate silence: boot (45 s) + menus (45 s) + error
# backoff (30 s), with room to spare.
STALL_SECONDS = 240.0
# The stalling watch (a policy hiding offstage, not a silent worker) steps down a level
# only this far below the threshold that raised it, so it does not flap on noise.
STALLING_HYSTERESIS = 0.03
SCORECARD_OPPONENTS = 'FOX,FALCO,MARTH,CPTFALCON,PEACH,JIGGLYPUFF'   # the fixed 60-game scorecard


class EventLog(deque):
    """The Events panel's recent entries, each also appended to events.jsonl so what
    happened overnight (ladder moves, alarms, scorecards) outlives the panel."""

    def __init__(self, path: Path, maxlen: int = 40):
        super().__init__(maxlen=maxlen)
        self.path = path
        try:
            path.touch(exist_ok=True)            # an empty log still says "nothing happened"
            for line in path.read_text().splitlines()[-maxlen:]:
                super().append(json.loads(line))
        except (OSError, ValueError):
            pass

    def append(self, event: dict):
        super().append(event)
        try:
            with open(self.path, 'a') as f:
                f.write(json.dumps(event, default=str) + '\n')
        except OSError:
            pass


def pick_device(name: str) -> torch.device:
    if name == 'auto':
        name = 'mps' if torch.backends.mps.is_available() else 'cpu'
    return torch.device(name)


def atomic_json(path: Path, data):
    tmp = path.with_name(f'.{path.name}.tmp')
    tmp.write_text(json.dumps(data, default=float))
    os.replace(tmp, path)


class Learner:
    def __init__(self, cfg: Config, run_dir: Path, resume: bool = False, init_from: str | None = None):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.run_dir / 'league').mkdir(exist_ok=True)
        torch.manual_seed(cfg.seed)
        torch.set_num_threads(2)
        self.device = pick_device(cfg.device)
        self.net = Policy(cfg.hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate, eps=1e-5)
        self.version = 0
        self.frames_trained = 0
        self.decisions_trained = 0
        self.frames_received = 0
        # Frames of actual game time. frames_trained counts experience, and a mirror
        # game gives two players' experience per frame of game, so it runs ahead of the
        # clock. Port 1 is present in every game, so its frames are the game's frames.
        self.world_frames_trained = 0
        self.world_received = 0
        self.world_estimated = False
        self.updates = 0
        self.dropped_stale = 0
        self.snapshots: list[str] = []
        self.last_snapshot_frames = 0
        self.ladder = league.Ladder(cfg)
        self.started = time.time()
        self.elapsed_before = 0.0
        self.errors_on_start: list[str] = []
        self.critic_warmup_left = 0
        self.stall: dict = {'level': 'ok'}
        self.last_ok_checkpoint: str | None = None    # newest checkpoint saved while not stalling
        if resume:
            self._restore(self.run_dir / 'checkpoints' / 'latest.pt')
        elif init_from:
            state, notes, trained_gamma = read_weights(Path(init_from), cfg.act_every)
            self.net.load_state_dict(state)
            self.errors_on_start += notes
            self._start_warmup(trained_gamma)
        # Only the value side trains during a critic warm-up; everything else (including
        # the input embeddings both sides share) keeps its weights and optimizer state.
        self.policy_side = [p for n, p in self.net.named_parameters() if not n.startswith(('v_body', 'v_head'))]
        cfg.save(self.run_dir / 'config.json')
        self.publish_policy()
        self.write_league()

        self.ctx = mp.get_context('spawn')
        self.queue = self.ctx.Queue(maxsize=512)
        self.stop_event = self.ctx.Event()
        self.procs: dict[int, mp.Process] = {}
        self.restarts = Counter()
        self.restart_at: dict[int, float] = {}
        self.worker_status: dict[int, dict] = {}
        self.worker_seen: dict[int, float] = {}
        self.worker_progress: dict[int, float] = {}   # last time frames advanced
        self.worker_frames: dict[int, int] = {}
        self.fatal: dict[int, dict] = {}
        self.errors = EventLog(self.run_dir / 'events.jsonl')
        for note in self.errors_on_start:
            self.errors.append({'worker': -1, 'time': time.time(), 'error': note})
        self.games = deque(maxlen=3000)
        self.metrics = deque(maxlen=300)
        self.last_metrics: dict = {}
        self.pending: list[dict] = []
        self.pending_steps = 0
        self.throughput = deque(maxlen=120)   # (time, frames_received, world_received)
        self.stopping = False
        self.cpu_games_since_league = 0
        self.crashed = None
        self.state = 'starting'
        self.last_checkpoint = time.monotonic()
        self.last_policy_file: Path | None = None
        self.scorecard_proc: subprocess.Popen | None = None
        self.scorecard_dir: Path | None = None
        self.last_scorecard = getattr(self, 'last_scorecard', None) or time.time()
        self._load_games()
        self.stall_check()

    # --------------------------------------------------------------- persistence
    def _restore(self, path: Path):
        from . import migrate
        ck = torch.load(path, map_location='cpu', weights_only=False)
        for note in contract.check(ck.get('contract'), self.cfg.act_every, what='resumed checkpoint', migrate=True):
            self.errors_on_start.append(note)
        if migrate.is_v1(ck.get('contract')):
            # The network's shape changed, so the optimizer's moment estimates no longer
            # line up with it: start Adam fresh, keep everything else.
            self.net.load_state_dict(migrate.v1_to_current(ck['net']))
            self.errors_on_start.append('optimizer state reset after conversion')
        else:
            self.net.load_state_dict(ck['net'])
            self.opt.load_state_dict(ck['opt'])
        s = ck['state']
        self.version = s['version']
        self.frames_trained = s['frames_trained']
        self.decisions_trained = s.get('decisions_trained', 0)
        if 'world_frames_trained' in s:
            self.world_frames_trained = s['world_frames_trained']
            self.world_estimated = s.get('world_estimated', False)
        else:
            # Saved before game time was tracked: reconstruct it from the game log.
            self.world_frames_trained = _logged_game_frames(self.run_dir / 'games.jsonl', self.version)
            self.world_estimated = True
        self.updates = s['updates']
        self.snapshots = [p for p in s.get('snapshots', []) if Path(p).exists()]
        self.last_snapshot_frames = s.get('last_snapshot_frames', 0)
        self.ladder = league.Ladder(self.cfg, s.get('ladder'))
        self.elapsed_before = s.get('elapsed', 0.0)
        self.last_ok_checkpoint = s.get('last_ok_checkpoint')
        self.last_scorecard = s.get('last_scorecard')
        trained = (ck.get('config') or {}).get('gamma')
        if trained is not None and math.isclose(trained, self.cfg.gamma):
            self.critic_warmup_left = s.get('critic_warmup_left', 0)
        else:
            self._start_warmup(trained)

    def _start_warmup(self, trained_gamma: float | None):
        n = self.cfg.critic_warmup_updates
        if not n or (trained_gamma is not None and math.isclose(trained_gamma, self.cfg.gamma)):
            return
        was = f'{trained_gamma:g}' if trained_gamma is not None else 'not recorded'
        self.critic_warmup_left = n
        self.errors_on_start.append(f'Planning horizon changed (gamma {was} -> {self.cfg.gamma:g}): the policy is '
                                    f'frozen for {n} updates while the value estimate adjusts')

    def _load_games(self):
        path = self.run_dir / 'games.jsonl'
        if path.exists():
            for line in path.read_text().splitlines()[-3000:]:
                try:
                    self.games.append(json.loads(line))
                except ValueError:
                    pass

    def checkpoint(self):
        # A standalone policy file per checkpoint: what the dashboard lists, evaluates
        # and promotes. Small enough to keep every one.
        name = f'policy-{self.frames_trained / 1e6:08.2f}M-v{self.version}.pt'
        policy_path = self.run_dir / 'checkpoints' / name
        save_policy(policy_path, self.net, self.version, self.frames_trained,
                    {'ladder_frontier': self.ladder.frontier, 'gamma': self.cfg.gamma}, act_every=self.cfg.act_every)
        self.last_policy_file = policy_path
        if self.stall['level'] == 'ok':
            self.last_ok_checkpoint = str(policy_path.resolve())
        state = {'version': self.version, 'frames_trained': self.frames_trained,
                 'decisions_trained': self.decisions_trained, 'updates': self.updates,
                 'world_frames_trained': self.world_frames_trained, 'world_estimated': self.world_estimated,
                 'snapshots': self.snapshots, 'last_snapshot_frames': self.last_snapshot_frames,
                 'ladder': self.ladder.to_dict(), 'elapsed': self.elapsed(),
                 'critic_warmup_left': self.critic_warmup_left, 'last_ok_checkpoint': self.last_ok_checkpoint,
                 'last_scorecard': self.last_scorecard}
        payload = {'net': {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
                   'opt': self.opt.state_dict(), 'state': state, 'config': self.cfg.to_dict(),
                   'contract': contract.current(self.cfg.act_every)}
        tmp = self.run_dir / 'checkpoints' / '.latest.tmp'
        torch.save(payload, tmp)
        os.replace(tmp, self.run_dir / 'checkpoints' / 'latest.pt')
        self.last_checkpoint = time.monotonic()
        self._prune_checkpoints()

    def _prune_checkpoints(self, recent: int = 12, hourly: float = 3600.0):
        """Keep the newest few and one per hour of wall time; a 5 MB file every ten
        minutes would otherwise be the only thing a week-long run grows without bound."""
        files = sorted((self.run_dir / 'checkpoints').glob('policy-*.pt'), key=lambda p: p.stat().st_mtime)
        keep, last_kept = set(files[-recent:]), None
        for p in files:
            t = p.stat().st_mtime
            if last_kept is None or t - last_kept >= hourly:
                keep.add(p)
                last_kept = t
        protected = _referenced_checkpoints()
        if self.last_ok_checkpoint:
            protected.add(str(Path(self.last_ok_checkpoint).resolve()))
        for p in files:
            if p not in keep and str(p.resolve()) not in protected:
                p.unlink(missing_ok=True)

    def publish_policy(self):
        save_policy(self.run_dir / 'policy.pt', self.net, self.version, self.frames_trained,
                    act_every=self.cfg.act_every)

    def write_league(self):
        league.write_league(self.run_dir / 'league.json', self.ladder.frontier, self.snapshots,
                            self.ladder.selfplay_fraction(), self.ladder.opponent_weights())

    def maybe_snapshot(self):
        if self.frames_trained - self.last_snapshot_frames < self.cfg.snapshot_every_frames:
            return
        self.last_snapshot_frames = self.frames_trained
        path = self.run_dir / 'league' / f'snap-{self.frames_trained / 1e6:08.2f}M.pt'
        save_policy(path, self.net, self.version, self.frames_trained, act_every=self.cfg.act_every)
        if str(path) not in self.snapshots:
            self.snapshots.append(str(path))
        while len(self.snapshots) > self.cfg.max_snapshots:
            self.snapshots.pop(0)
        self.write_league()

    # ------------------------------------------------------------------ workers
    def start_worker(self, i: int):
        seed = self.cfg.seed * 1000 + i * 97 + self.restarts[i] * 7919
        p = self.ctx.Process(target=_worker_entry, name=f'puff-worker-{i}',
                             args=(i, self.cfg.to_dict(), str(self.run_dir), self.queue, self.stop_event, seed))
        p.start()
        self.procs[i] = p
        self.worker_seen[i] = self.worker_progress[i] = time.monotonic()
        self.worker_frames.pop(i, None)

    def supervise(self):
        now = time.monotonic()
        for i, p in list(self.procs.items()):
            if p.is_alive():
                # Alive is not the same as working: a worker that has sent nothing for
                # this long is stuck somewhere its own deadlines do not cover.
                # A heartbeat is not progress: status messages with the same frame count
                # are a stuck worker that is still talking (GPT step-1 review, R6).
                silent = now - self.worker_seen.get(i, now)
                stalled = now - self.worker_progress.get(i, now)
                if max(silent, stalled) > STALL_SECONDS and not self.stopping:
                    self.errors.append({'worker': i, 'time': time.time(),
                                        'error': f'no progress for {STALL_SECONDS:.0f}s; restarting'})
                    p.kill()
                    self.worker_seen[i] = self.worker_progress[i] = now
                continue
            if i not in self.restart_at:
                k = min(self.restarts[i], len(RESTART_BACKOFF) - 1)
                self.restart_at[i] = now + RESTART_BACKOFF[k]
                self.errors.append({'worker': i, 'time': time.time(),
                                    'error': f'worker exited with code {p.exitcode}; restarting'})
                st = self.worker_status.setdefault(i, {})
                st['phase'] = 'restarting'
            elif now >= self.restart_at[i] and not self.stopping:
                del self.restart_at[i]
                self.restarts[i] += 1
                p.join(timeout=0.1)
                self.start_worker(i)

    # ------------------------------------------------------------------ messages
    def handle(self, msg: dict):
        kind = msg.get('kind')
        if 'worker' in msg and msg['worker'] >= 0:
            now = time.monotonic()
            self.worker_seen[msg['worker']] = now
            if kind != 'status' or msg.get('frames') != self.worker_frames.get(msg['worker']):
                self.worker_progress[msg['worker']] = now
                self.worker_frames[msg['worker']] = msg.get('frames', self.worker_frames.get(msg['worker']))
        if kind == 'frag':
            self.frames_received += msg['frames']
            if msg.get('port', 1) == 1:
                self.world_received += msg['frames']
            self.throughput.append((time.monotonic(), self.frames_received, self.world_received))
            if msg['version'] < self.version - self.cfg.max_policy_lag:
                self.dropped_stale += 1
                return
            self.pending.append(msg)
            self.pending_steps += len(msg['actions'])
        elif kind == 'game':
            self.games.append(msg)
            with open(self.run_dir / 'games.jsonl', 'a') as f:
                f.write(json.dumps({k: v for k, v in msg.items() if k != 'kind'}) + '\n')
            self.stall_check()
            if msg['opponent_kind'] == 'cpu' and msg['cpu_level'] > 0:
                moved = self.ladder.record(msg['cpu_level'], msg['result'], msg.get('opponent'))
                self.cpu_games_since_league += 1
                if moved:
                    note = {'mastered': f'Top level beaten: CPU {self.ladder.frontier} over the last window',
                            'unmastered': f'CPU {self.ladder.frontier} no longer beaten over the last window'
                            }.get(moved, f'Ladder {moved}: frontier is now CPU {self.ladder.frontier}')
                    self.errors.append({'worker': -1, 'time': time.time(), 'error': note})
                if moved or self.cpu_games_since_league >= 10:
                    self.write_league()      # also refreshes matchup-focus weights
                    self.cpu_games_since_league = 0
        elif kind == 'status':
            self.worker_status[msg['worker']] = msg
        elif kind == 'error':
            self.errors.append(msg)
        elif kind == 'fatal':
            self.fatal[msg['worker']] = msg
            self.errors.append({'worker': msg['worker'], 'time': msg['time'], 'error': 'FATAL ' + msg['error']})

    # ------------------------------------------------------------------- update
    def entropy_coef(self) -> float:
        c = self.cfg
        frac = min(1.0, self.frames_trained / max(1, c.entropy_anneal_frames))
        return c.entropy + (c.entropy_final - c.entropy) * frac

    def update(self):
        cfg = self.cfg
        take, steps = [], 0
        while self.pending and steps < cfg.batch_steps:
            f = self.pending.pop(0)
            self.pending_steps -= len(f['actions'])
            if f['version'] < self.version - cfg.max_policy_lag:
                self.dropped_stale += 1      # went stale while it waited
                continue
            take.append(f)
            steps += len(f['actions'])
        if not take:
            return
        t0 = time.monotonic()
        dev = self.device
        T = len(take[0]['actions'])
        floats = torch.from_numpy(np.stack([f['floats'] for f in take])).to(dev)
        ids = torch.from_numpy(np.stack([f['ids'] for f in take])).to(dev)
        masks = torch.from_numpy(np.stack([f['masks'] for f in take])).to(dev)
        acts = torch.from_numpy(np.stack([f['actions'] for f in take]))
        blogp = torch.from_numpy(np.stack([f['logp'] for f in take]))
        rewards = torch.from_numpy(np.stack([f['rewards'] for f in take]))
        discounts = torch.from_numpy(np.stack([f['discounts'] for f in take]))
        N = len(take)
        with torch.no_grad():
            logits, values = self.net(floats.reshape(N * (T + 1), -1), ids.reshape(N * (T + 1), -1),
                                      masks.reshape(N * (T + 1), -1))
            logits = logits.reshape(N, T + 1, -1)[:, :T].float().cpu()
            values = values.reshape(N, T + 1).float().cpu()
            prox = torch.log_softmax(logits, -1).gather(-1, acts[..., None]).squeeze(-1)
        vs, adv = vtrace(values[:, :T], values[:, T], rewards, discounts, prox - blogp, cfg.lam, cfg.rho_clip)
        flat = lambda x: x.reshape(N * T, *x.shape[2:])
        b_floats, b_ids, b_masks = flat(floats[:, :T]), flat(ids[:, :T]), flat(masks[:, :T])
        b_acts, b_prox, b_adv, b_vs = (flat(x).to(dev) for x in (acts, prox, adv, vs))
        ent = self.entropy_coef()
        warm = self.critic_warmup_left > 0
        stats = defaultdict(float)
        n_mb = 0
        self.net.train()
        for _ in range(cfg.epochs):
            perm = torch.randperm(N * T, device=dev)
            for start in range(0, N * T, cfg.minibatch):
                idx = perm[start:start + cfg.minibatch]
                lg, v = self.net(b_floats[idx], b_ids[idx], b_masks[idx])
                loss, st = ppo_loss(lg, v, b_acts[idx], b_prox[idx], b_adv[idx], b_vs[idx],
                                    cfg.clip, cfg.value_coef, ent, policy_coef=0.0 if warm else 1.0)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if warm:
                    for p in self.policy_side:
                        p.grad = None        # Adam skips a parameter with no gradient: no momentum drift
                gn = torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()
                for k, v_ in st.items():
                    stats[k] += v_
                stats['grad_norm'] += float(gn)
                n_mb += 1
        self.net.eval()
        self.version += 1
        self.updates += 1
        if warm:
            self.critic_warmup_left -= 1
            if not self.critic_warmup_left:
                self.errors.append({'worker': -1, 'time': time.time(),
                                    'error': 'Value warm-up finished: the policy is learning again'})
        frames = sum(f['frames'] for f in take)
        self.frames_trained += frames
        self.world_frames_trained += sum(f['frames'] for f in take if f.get('port', 1) == 1)
        self.decisions_trained += N * T
        self.publish_policy()
        lags = [self.version - 1 - f['version'] for f in take]
        m = {k: round(v / max(n_mb, 1), 5) for k, v in stats.items()}
        m.update(update=self.updates, version=self.version, frames_trained=self.frames_trained,
                 seconds=round(time.monotonic() - t0, 3), entropy_coef=round(ent, 5),
                 mean_lag=round(float(np.mean(lags)), 2), max_lag=int(max(lags)),
                 reward_per_decision=round(float(rewards.mean()), 5),
                 value_mean=round(float(values.mean()), 4), critic_warmup=self.critic_warmup_left,
                 time=time.time())
        self.last_metrics = m
        self.metrics.append(m)
        with open(self.run_dir / 'metrics.jsonl', 'a') as f:
            f.write(json.dumps(m) + '\n')
        self.maybe_snapshot()

    # ------------------------------------------------------------------- status
    def elapsed(self) -> float:
        return self.elapsed_before + (time.time() - self.started)

    def rate(self, window=30.0, world=False) -> float:
        if len(self.throughput) < 2:
            return 0.0
        now = time.monotonic()
        pts = [p for p in self.throughput if now - p[0] <= window] or list(self.throughput)[-2:]
        if len(pts) < 2 or pts[-1][0] == pts[0][0]:
            return 0.0
        k = 2 if world else 1
        return (pts[-1][k] - pts[0][k]) / (pts[-1][0] - pts[0][0])

    def summary(self) -> dict:
        # Games cut short (emulator failure, safety cap) have no result; never count them.
        recent = [g for g in list(self.games)[-400:] if g['result'] in COMPLETED]
        by_level, by_opp, other, by_phillip = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        for gm in recent:
            win = 1.0 if gm['result'] == 'win' else 0.0
            if gm['opponent_kind'] == 'cpu':
                by_level[gm['cpu_level']].append(win)
                by_opp[gm['opponent']].append(win)
            elif gm['opponent_kind'] == 'phillip':
                by_phillip[gm['opponent']].append(win)
                other['phillip'].append(win)
            else:
                other[gm['opponent_kind']].append(win)
        fmt = lambda d: {str(k): {'games': len(v), 'win_rate': round(sum(v) / len(v), 3)} for k, v in sorted(d.items())}
        last = recent[-200:]
        lost = sum(g_['stocks_lost'] for g_ in last)
        sd = sum(g_['self_destructs'] for g_ in last)
        rests = sum(g_['rests'] for g_ in last)
        hits = sum(g_['rest_hits'] for g_ in last)
        techs = sum(g_.get('techs', 0) for g_ in last)
        missed = sum(g_.get('missed_techs', 0) for g_ in last)
        landings = sum(g_.get('aerial_landings', 0) for g_ in last)
        acts = np.zeros(actions.COUNT)
        for g_ in last:
            if len(g_.get('actions', ())) == actions.COUNT:
                acts += np.array(g_['actions'])
        return {
            'by_level': fmt(by_level), 'by_opponent': fmt(by_opp), 'selfplay': fmt(other),
            'by_phillip': fmt(by_phillip),
            'self_destruct_share': round(sd / lost, 3) if lost else None,
            'rest_hit_rate': round(hits / rests, 3) if rests else None,
            'rests_per_game': round(rests / len(last), 2) if last else None,
            'damage_ratio': round(sum(g_['damage_dealt'] for g_ in last) /
                                  max(1.0, sum(g_['damage_received'] for g_ in last)), 3) if last else None,
            'action_share': {n: round(float(a / acts.sum()), 4) for n, a in zip(actions.NAMES, acts)} if acts.sum() else {},
            'tech_rate': round(techs / (techs + missed), 3) if techs + missed else None,
            'offstage_airdodges': sum(g_.get('offstage_airdodges', 0) for g_ in last),
            'aerial_landing_lag': round(sum(g_.get('aerial_landing_frames', 0) for g_ in last) / landings, 1) if landings else None,
        }

    def stall_check(self):
        """Watch for stalling: a policy that learns to hide offstage to postpone losses.
        The 2026-09-26 run gave no warning in any learning metric; time offstage was the
        signal (25-48% healthy, 65-78% stalling). Announces each change of level."""
        cfg = self.cfg
        done = [g for g in list(self.games)[-3 * cfg.stall_window:] if g['result'] in COMPLETED][-cfg.stall_window:]
        frames = sum(g.get('frames', 0) for g in done)
        share = sum(g.get('offstage_frames', 0) for g in done) / frames if frames else None
        minutes = frames / len(done) / 3600 if done else None
        prev = self.stall.get('level', 'ok')
        level = 'ok'
        if share is not None and len(done) >= cfg.stall_window:
            level = 'alarm' if share >= cfg.stall_offstage_alarm else 'warn' if share >= cfg.stall_offstage_warn else 'ok'
            rank = ('ok', 'warn', 'alarm')
            bar = cfg.stall_offstage_alarm if prev == 'alarm' else cfg.stall_offstage_warn
            if rank.index(level) < rank.index(prev) and share > bar - STALLING_HYSTERESIS:
                level = prev             # step down only once clearly below the line
        ok = Path(self.last_ok_checkpoint).name if self.last_ok_checkpoint else None
        self.stall = {'level': level, 'since': self.stall.get('since') if level == prev else time.time(),
                      'offstage_share': None if share is None else round(share, 3),
                      'game_minutes': None if minutes is None else round(minutes, 2),
                      'games': len(done), 'window': cfg.stall_window,
                      'warn': cfg.stall_offstage_warn, 'alarm': cfg.stall_offstage_alarm,
                      'last_ok_checkpoint': self.last_ok_checkpoint}
        if level == prev:
            return
        where = f'{share:.0%} of the last {len(done)} games offstage; games average {minutes:.1f} min'
        note = {
            'warn': f'Stalling watch: {where} (healthy is under {cfg.stall_offstage_warn:.0%}). '
                    f'Last checkpoint before this: {ok or "none yet"}',
            'alarm': f'Stalling ALARM: {where}. Consider stopping and restarting from {ok or "an earlier checkpoint"}',
            'ok': f'Stalling watch back to normal: {share:.0%} offstage over the last {len(done)} games',
        }[level]
        self.errors.append({'worker': -1, 'time': time.time(), 'error': note})

    def write_status(self):
        workers = []
        for i in range(self.cfg.workers):
            st = dict(self.worker_status.get(i, {}))
            p = self.procs.get(i)
            st.update(worker=i, alive=bool(p and p.is_alive()), restarts=self.restarts[i])
            if i in self.fatal:
                st['fatal'] = self.fatal[i]['error']
            workers.append(st)
        rate = self.rate()
        world_rate = self.rate(world=True)
        status = {
            'state': self.state, 'run': self.run_dir.name, 'pid': os.getpid(), 'device': str(self.device),
            'elapsed': round(self.elapsed(), 1), 'updated': time.time(),
            'version': self.version, 'updates': self.updates,
            'frames_trained': self.frames_trained, 'decisions_trained': self.decisions_trained,
            'world_frames_trained': self.world_frames_trained, 'world_frames_estimated': self.world_estimated,
            'game_hours_played': round(self.world_frames_trained / 60 / 3600, 2),
            'experience_hours': round(self.frames_trained / 60 / 3600, 2),
            'game_hours_trained': round(self.frames_trained / 60 / 3600, 2),   # = experience_hours (old name)
            'frames_per_second': round(rate, 1),
            'world_frames_per_second': round(world_rate, 1),
            'realtime_multiple': round(world_rate / 60.0, 2),
            'emulator_fps': round(sum(w.get('fps', 0) for w in workers if w.get('alive')), 1),
            'pending_steps': self.pending_steps, 'dropped_stale': self.dropped_stale,
            'games': len(self.games), 'ladder': {'frontier': self.ladder.frontier, 'levels': self.ladder.rates(),
                                                  'mastered': self.ladder.mastered,
                                                  'selfplay_fraction': self.ladder.selfplay_fraction(),
                                                  'matchups': self.ladder.matchup_table(),
                                                  'history': self.ladder.history[-10:]},
            'snapshots': [Path(s).stem for s in self.snapshots],
            'metrics': self.last_metrics, 'workers': workers, 'errors': list(self.errors)[-15:],
            'recent_games': [{k: v for k, v in g.items() if k != 'actions'} for g in list(self.games)[-25:]],
            'summary': self.summary(), 'config': self.cfg.to_dict(),
            'stall': self.stall, 'critic_warmup_left': self.critic_warmup_left,
            'scorecard': {'running': self.scorecard_proc is not None,
                          'dir': self.scorecard_dir.name if self.scorecard_dir else None,
                          'next': (self.last_scorecard + self.cfg.scorecard_hours * 3600) if self.cfg.scorecard_hours else None},
        }
        atomic_json(self.run_dir / 'status.json', status)

    def control(self) -> bool:
        path = self.run_dir / 'control.json'
        try:
            return bool(json.loads(path.read_text()).get('stop'))
        except (OSError, ValueError):
            return False

    # --------------------------------------------------------------------- run
    def run(self):
        (self.run_dir / 'control.json').unlink(missing_ok=True)
        stop_signal = []
        signal.signal(signal.SIGTERM, lambda *_: stop_signal.append(1))
        signal.signal(signal.SIGINT, lambda *_: stop_signal.append(1))
        # Hold the Mac awake (not the display) for as long as this process lives.
        try:
            import subprocess
            subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError:
            pass
        for i in range(self.cfg.workers):
            self.start_worker(i)
            time.sleep(0.4)  # stagger boots; four Dolphins booting at once starve each other
        self.state = 'running'
        last_status = 0.0
        try:
            while True:
                deadline = time.monotonic() + 0.05
                while time.monotonic() < deadline:
                    try:
                        self.handle(self.queue.get(timeout=0.02))
                    except queue_mod.Empty:
                        break
                if self.pending_steps >= self.cfg.batch_steps:
                    self.update()
                self.supervise()
                now = time.monotonic()
                if now - last_status >= STATUS_EVERY:
                    last_status = now
                    self.write_status()
                    if stop_signal or self.control():
                        break
                    if self.cfg.total_frames and self.frames_trained >= self.cfg.total_frames:
                        break
                if now - self.last_checkpoint >= self.cfg.checkpoint_minutes * 60:
                    self.checkpoint()
                    self.maybe_scorecard()
                if self.scorecard_proc is not None and self.scorecard_proc.poll() is not None:
                    self._scorecard_finished()
        except Exception:
            self.crashed = traceback.format_exc()[-3000:]
            self.errors.append({'worker': -1, 'time': time.time(), 'error': 'LEARNER ' + self.crashed[-1500:]})
            raise
        finally:
            self.shutdown()

    # --------------------------------------------------------------- scorecards
    def maybe_scorecard(self):
        """Every `scorecard_hours`, the newest checkpoint plays the fixed scorecard (six
        characters x `scorecard_games` vs CPU `scorecard_level`) in a separate evaluation
        process on its own ports, beside training. Called right after a checkpoint."""
        cfg = self.cfg
        if (not cfg.scorecard_hours or self.scorecard_proc is not None or self.last_policy_file is None
                or time.time() - self.last_scorecard < cfg.scorecard_hours * 3600):
            return
        ck = self.last_policy_file
        out = RUNS / 'evals' / f"{time.strftime('%Y%m%d-%H%M%S')}-scorecard-{self.run_dir.name}-v{self.version}"
        out.mkdir(parents=True, exist_ok=True)
        args = [sys.executable, '-u', '-m', 'puffbot', 'evaluate', str(ck.resolve()),
                '--level', str(cfg.scorecard_level), '--games', str(cfg.scorecard_games),
                '--workers', str(cfg.scorecard_workers), '--opponents', SCORECARD_OPPONENTS,
                '--label', 'scorecard', '--dir', str(out)]
        self.scorecard_proc = self._spawn_scorecard(args, out)
        self.scorecard_dir = out
        self.last_scorecard = time.time()
        self.errors.append({'worker': -1, 'time': time.time(),
                            'error': f'Scorecard started: {ck.name} vs CPU {cfg.scorecard_level}, '
                                     f'{6 * cfg.scorecard_games} games on {cfg.scorecard_workers} extra emulators'})

    def _spawn_scorecard(self, args: list[str], out: Path) -> subprocess.Popen:
        with open(out / 'evaluate.log', 'ab') as log:
            return subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL)

    def _scorecard_finished(self):
        rep = {}
        try:
            rep = json.loads((self.scorecard_dir / 'report.json').read_text())
        except (OSError, ValueError, TypeError):
            pass
        o = rep.get('overall') or {}
        name = Path(rep.get('checkpoint', '?')).name
        if rep.get('state') == 'complete':
            lo, hi = o.get('win_rate_95ci') or (0, 0)
            note = (f"Scorecard {name}: won {o.get('wins')}/{o.get('games')} ({o.get('win_rate', 0):.0%}, "
                    f"95% CI {lo:.0%}-{hi:.0%}) vs CPU {rep.get('level')}")
        else:
            note = f"Scorecard {name} ended {rep.get('state', 'without a report')} ({o.get('games', 0)} games)"
        self.errors.append({'worker': -1, 'time': time.time(), 'error': note})
        self.scorecard_proc = None
        self.scorecard_dir = None

    def _stop_scorecard(self):
        p = self.scorecard_proc
        if p is None or p.poll() is not None:
            return
        p.send_signal(signal.SIGINT)          # the evaluation cancels cleanly and reaps its Dolphins
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()
            _reap_dolphins(self.scorecard_dir)

    def shutdown(self):
        """Every step is independent: a failure in one must never leave workers or
        emulators running, which is how a crashed run used to hold the machine."""
        self.stopping = True
        self.state = 'stopping'
        self.stop_event.set()
        _quietly(self.write_status)
        _quietly(self._stop_scorecard)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and any(p.is_alive() for p in self.procs.values()):
            try:
                self.handle(self.queue.get(timeout=0.1))
            except queue_mod.Empty:
                pass
            except Exception:
                pass
        for p in self.procs.values():
            if p.is_alive():
                p.kill()
        for p in self.procs.values():
            p.join(timeout=2)
        _quietly(_reap_dolphins, self.run_dir)
        _quietly(self.checkpoint)
        self.state = 'crashed' if self.crashed else 'stopped'
        _quietly(self.write_status)
        # Drop the queue's feeder thread so interpreter exit cannot block on it.
        _quietly(self.queue.cancel_join_thread)


def _referenced_checkpoints() -> set[str]:
    """Checkpoints an evaluation report or a champion points at: never pruned."""
    from .config import ROOT, RUNS
    refs = set()
    for report in (RUNS / 'evals').glob('*/report.json'):
        try:
            ck = json.loads(report.read_text()).get('checkpoint')
            if ck:
                refs.add(str(Path(ck).resolve()))
        except (OSError, ValueError):
            pass
    try:
        for c in json.loads((ROOT / 'champions' / 'champions.json').read_text()):
            refs.add(str(Path(c['source']).resolve()))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return refs


def _logged_game_frames(path: Path, up_to_version: int | None = None) -> int:
    """Game frames of logged games, only those played by policies up to the checkpoint
    being restored (a log can run ahead of an older checkpoint)."""
    total = 0
    try:
        for line in path.read_text().splitlines():
            g = json.loads(line)
            if up_to_version is None or int(g.get('version', 0)) <= up_to_version:
                total += int(g.get('frames', 0))
    except (OSError, ValueError):
        pass
    return total


def _quietly(fn, *args):
    try:
        fn(*args)
    except Exception:
        traceback.print_exc()


def read_weights(path: Path, act_every: int | None = None) -> tuple[dict, list[str], float | None]:
    """Network weights from a policy file or a full training checkpoint, plus notes and
    the gamma they were trained with (None if the file predates recording it)."""
    from . import migrate
    payload = torch.load(path, map_location='cpu', weights_only=False)
    notes = contract.check(payload.get('contract'), act_every, what=Path(path).name, migrate=True)
    weights = payload['state'] if 'state' in payload and 'net' not in payload else payload['net']
    if migrate.is_v1(payload.get('contract')):
        weights = migrate.v1_to_current(weights)
    gamma = payload.get('gamma', (payload.get('config') or {}).get('gamma'))
    return weights, notes, gamma


def _reap_dolphins(run_dir: Path):
    """Kill any Dolphin a worker recorded and did not take down with it."""
    from .dolphin import _is_dolphin, kill_pid
    for pidfile in Path(run_dir).glob('workers/*/dolphin.pid'):
        try:
            pid = int(pidfile.read_text().strip())
            if _is_dolphin(pid):
                kill_pid(pid)
        except (ValueError, OSError):
            pass
        pidfile.unlink(missing_ok=True)


def _worker_entry(*args):
    from .actor import run
    run(*args)
