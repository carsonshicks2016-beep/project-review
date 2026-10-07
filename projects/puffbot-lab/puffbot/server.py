"""Dashboard: watch training, start and stop runs, evaluate and crown checkpoints.

The dashboard never owns a game loop. Training, evaluation and watching are separate
processes it launches and then only reads files from, so closing the browser or
restarting the dashboard never touches a run.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from collections import defaultdict
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import ROOT, RUNS, Config

WEB = Path(__file__).parent / 'web'
CHAMPIONS = ROOT / 'champions'
app = FastAPI(title='Puff Bot')
app.mount('/static', StaticFiles(directory=WEB), name='static')


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _tail_jsonl(path: Path, n: int) -> list:
    if not path.exists():
        return []
    with open(path, 'rb') as f:
        f.seek(0, 2)
        size = f.tell()
        block = min(size, max(65536, n * 1500))
        f.seek(size - block)
        lines = f.read().decode(errors='ignore').splitlines()
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def _pid_alive(pid) -> bool:
    try:
        # A finished child of this server stays a zombie (and looks alive) until reaped.
        if os.waitpid(int(pid), os.WNOHANG)[0]:
            return False
    except (ChildProcessError, OSError, TypeError, ValueError):
        pass
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _run_dir(run: str) -> Path:
    d = (RUNS / run).resolve()
    if RUNS.resolve() not in d.parents or not d.is_dir():
        raise HTTPException(404, f'No run {run}')
    return d


def _live_state(status: dict | None) -> str:
    if not status:
        return 'unknown'
    state = status.get('state', 'unknown')
    if state in ('running', 'starting', 'stopping') and not _pid_alive(status.get('pid')):
        return 'crashed' if state != 'stopping' else 'stopped'
    return state


def _spawn(args: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, 'ab') as out:
        p = subprocess.Popen([sys.executable, '-u', '-m', 'puffbot', *args], cwd=ROOT, stdout=out,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    return p.pid


def _active_training() -> str | None:
    for d in sorted(RUNS.glob('*'), reverse=True):
        st = _read_json(d / 'status.json')
        if st and _live_state(st) in ('running', 'starting', 'stopping'):
            return d.name
    return None


@app.get('/')
def index():
    return FileResponse(WEB / 'index.html')


@app.get('/api/runs')
def runs():
    out = []
    if RUNS.exists():
        for d in sorted(RUNS.iterdir(), reverse=True):
            if not d.is_dir() or d.name in ('evals', 'bench'):
                continue
            st = _read_json(d / 'status.json') or {}
            out.append({'run': d.name, 'state': _live_state(st) if st else 'empty',
                        'frames_trained': st.get('frames_trained', 0),
                        'game_hours_played': st.get('game_hours_played'),
                        'game_hours_trained': st.get('game_hours_trained', 0),
                        'frontier': (st.get('ladder') or {}).get('frontier'),
                        'games': st.get('games', 0), 'updated': st.get('updated')})
    return out


def _legacy_game_time(d: Path, st: dict) -> dict:
    """Statuses written before game time was tracked separately only have player
    experience (both players of a mirror game). Estimate game time read-only from the
    game log and say so; never show experience under the game-time label."""
    if 'game_hours_played' in st:
        return st
    frames = 0
    for g in _tail_jsonl(d / 'games.jsonl', 10 ** 7):
        frames += int(g.get('frames', 0))
    st['game_hours_played'] = round(frames / 60 / 3600, 2)
    st['world_frames_estimated'] = True
    st['experience_hours'] = st.get('game_hours_trained')
    st['realtime_multiple'] = None            # the old figure counted experience, not game time
    st['world_frames_per_second'] = None
    return st


@app.get('/api/runs/{run}/status')
def status(run: str):
    d = _run_dir(run)
    st = _read_json(d / 'status.json')
    if st is None:
        raise HTTPException(404, 'No status yet')
    st = _legacy_game_time(d, st)
    st['state'] = _live_state(st)
    st['age'] = round(time.time() - st.get('updated', time.time()), 1)
    return JSONResponse(st)


@app.get('/api/runs/{run}/metrics')
def metrics(run: str, tail: int = 600):
    return _tail_jsonl(_run_dir(run) / 'metrics.jsonl', tail)


@app.get('/api/runs/{run}/games')
def games(run: str, tail: int = 1500):
    rows = _tail_jsonl(_run_dir(run) / 'games.jsonl', tail)
    for r in rows:
        r.pop('actions', None)
    return rows


# ------------------------------------------------------------ reports & panels
COMPLETED = ('win', 'loss', 'draw')
SCORECARD = {'kind': 'cpu', 'level': 9, 'games_each': 10,
             'opponents': ['FOX', 'FALCO', 'MARTH', 'CPTFALCON', 'PEACH', 'JIGGLYPUFF']}
_games_lock = threading.Lock()
_games_cache: dict[Path, tuple[int, list]] = {}


def _all_games(d: Path) -> list[dict]:
    """Every game of a run, read incrementally: the log only grows, so each request
    parses just the lines appended since the last one."""
    path = d / 'games.jsonl'
    try:
        size = path.stat().st_size
    except OSError:
        return []
    with _games_lock:
        offset, rows = _games_cache.get(path, (0, []))
        if size < offset:
            offset, rows = 0, []
        if size > offset:
            with open(path, 'rb') as f:
                f.seek(offset)
                chunk = f.read(size - offset)
            end = chunk.rfind(b'\n') + 1          # a line still being written waits
            for line in chunk[:end].splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
            offset += end
            _games_cache[path] = (offset, rows)
        return rows


def _rate(wins: int, n: int):
    return round(wins / n, 3) if n else None


def _bucket_stats(rows: list[dict]) -> dict:
    """What a morning report says about a stretch of games."""
    done = [g for g in rows if g.get('result') in COMPLETED]
    cpu = [g for g in done if g.get('opponent_kind') == 'cpu']
    cpu9 = [g for g in cpu if g.get('cpu_level') == 9]
    ph = [g for g in done if g.get('opponent_kind') == 'phillip']
    selfp = [g for g in done if g.get('opponent_kind') in ('mirror', 'snapshot')]
    wins = lambda gs: sum(g['result'] == 'win' for g in gs)
    frames = sum(g.get('frames', 0) for g in done)
    techs = sum(g.get('techs', 0) for g in done)
    missed = sum(g.get('missed_techs', 0) for g in done)
    return {
        'games': len(done), 'game_hours': round(frames / 216000, 2),
        'cpu9_wins': wins(cpu9), 'cpu9_games': len(cpu9), 'cpu9_rate': _rate(wins(cpu9), len(cpu9)),
        'cpu_wins': wins(cpu), 'cpu_games': len(cpu), 'cpu_rate': _rate(wins(cpu), len(cpu)),
        'cpu_level': round(sum(g.get('cpu_level') or 0 for g in cpu) / len(cpu), 2) if cpu else None,
        'phillip_wins': wins(ph), 'phillip_games': len(ph),
        'phillip_taken': round(sum(g.get('stocks_taken', 0) for g in ph) / len(ph), 2) if ph else None,
        'self_wins': wins(selfp), 'self_games': len(selfp),
        'tech_rate': _rate(techs, techs + missed),
        'offstage': round(sum(g.get('offstage_frames', 0) for g in done) / frames, 3) if frames else None,
        'minutes': round(frames / len(done) / 3600, 2) if done else None,
        'damage_ratio': round(sum(g.get('damage_dealt', 0) for g in cpu) /
                              max(1.0, sum(g.get('damage_received', 0) for g in cpu)), 2) if cpu else None,
        'sd_per_game': round(sum(g.get('self_destructs', 0) for g in done) / len(done), 2) if done else None,
    }


def _version_of(name: str):
    m = re.search(r'-v(\d+)\.pt$', name)       # policy-00048.17M-v3304.pt
    return int(m.group(1)) if m else None


def _scorecards() -> list[dict]:
    """Every evaluation run with the fixed scorecard protocol, oldest first."""
    out = []
    for d in sorted((RUNS / 'evals').glob('*')):
        rep = _read_json(d / 'report.json')
        if not rep:
            continue
        if rep.get('kind') is None:
            rep['kind'] = 'cpu'           # reports from before Phillip existed
        if any(rep.get(k) != v for k, v in SCORECARD.items()):
            continue
        ck = Path(rep.get('checkpoint', ''))
        o = rep.get('overall') or {}
        state = rep.get('state')
        if state == 'finished':          # reports before 'complete' existed: judge by the count
            state = 'complete' if o.get('games', 0) >= 60 else 'incomplete'
        out.append({'id': d.name, 'state': state, 'label': rep.get('label'),
                    'started': rep.get('started'), 'finished': rep.get('finished'),
                    'checkpoint': str(ck), 'checkpoint_name': ck.name,
                    'run': ck.parent.parent.name if ck.parent.name == 'checkpoints' else None,
                    'version': rep.get('checkpoint_version') or _version_of(ck.name),
                    'frames': rep.get('checkpoint_frames'),
                    'wins': o.get('wins'), 'games': o.get('games'), 'win_rate': o.get('win_rate'),
                    'ci': o.get('win_rate_95ci'), 'stocks_taken': o.get('stocks_taken_per_game'),
                    'stocks_lost': o.get('stocks_lost_per_game'),
                    'techs': o.get('techs'), 'missed_techs': o.get('missed_techs'),
                    'by_opponent': {k: {'wins': v.get('wins'), 'games': v.get('games')}
                                    for k, v in (rep.get('by_opponent') or {}).items()}})
    return out


def _events(d: Path) -> list[dict]:
    return _tail_jsonl(d / 'events.jsonl', 10 ** 6)


DISPLAY = {'CPTFALCON': 'Falcon', 'JIGGLYPUFF': 'Puff', 'GAMEANDWATCH': 'Mr. Game & Watch',
           'DOC': 'Dr. Mario', 'YLINK': 'Young Link', 'POPO': 'Ice Climbers'}


def _name(char: str) -> str:
    return DISPLAY.get(char, char.title())


def _pct(x) -> str:
    return f'{x:.0%}' if x is not None else '-'


def _clock(t: float) -> str:
    return time.strftime('%H:%M', time.localtime(t))


def _headline(total: dict, hours: list[dict], events: list[dict] | None, cards: list[dict], span_h: float,
              ph_wins: list[dict], warn: float) -> list[str]:
    """The report in plain English, most important first."""
    gh = total['game_hours']
    lines = [f"Puff played {total['games']:,} games in the last {span_h:.1f} hours "
             f"({gh:.0f} hour{'s' if round(gh) != 1 else ''} of Melee)."]
    cpu9 = [h for h in hours if h['cpu9_games'] >= 10]
    if total['cpu9_games']:
        s = f"Against level-9 CPUs she won {_pct(total['cpu9_rate'])} ({total['cpu9_wins']}/{total['cpu9_games']})"
        if len(cpu9) >= 2:
            best = max(cpu9, key=lambda h: h['cpu9_rate'])
            s += (f"; hour by hour it went from {_pct(cpu9[0]['cpu9_rate'])} to {_pct(cpu9[-1]['cpu9_rate'])}, "
                  f"best {_pct(best['cpu9_rate'])} at {best['label']}")
        lines.append(s + '.')
    if total['phillip_games']:
        s = (f"Against Phillip she won {total['phillip_wins']} of {total['phillip_games']} games and took "
             f"{total['phillip_taken']:.1f} stocks per game")
        ph = [h for h in hours if h['phillip_games'] >= 5]
        if len(ph) >= 2:
            s += f" ({ph[0]['phillip_taken']:.1f} in the first hour, {ph[-1]['phillip_taken']:.1f} in the last)"
        if ph_wins:
            chars = defaultdict(int)
            for g in ph_wins:
                chars[_name(g['opponent'])] += 1
            s += '; wins against ' + ', '.join(f'{c} ×{n}' if n > 1 else c for c, n in chars.items())
        lines.append(s + '.')
    for c in cards:
        if c['state'] == 'complete':
            lines.append(f"Scorecard at {_clock(c['started'])} ({c['checkpoint_name']}): "
                         f"{c['wins']}/{c['games']} vs CPU 9 ({_pct(c['win_rate'])}).")
    # Judged from the games themselves, so runs from before the stalling watch get it too.
    high = [h for h in hours if h['games'] >= 20 and (h['offstage'] or 0) >= warn]
    if high:
        lines.append(f"Possible stalling: time offstage was over the {_pct(warn)} warning line in {len(high)} "
                     f"hour{'s' if len(high) > 1 else ''}, from {high[0]['label']} (peak {_pct(max(h['offstage'] for h in high))}). "
                     f"See the stalling chart on the Live tab.")
    else:
        lines.append(f"No sign of stalling: time offstage averaged {_pct(total['offstage'])} "
                     f"and never reached the {_pct(warn)} warning line in any hour.")
    if total['tech_rate'] is not None:
        lines.append(f"She teched {_pct(total['tech_rate'])} of knockdowns and self-destructed "
                     f"{total['sd_per_game']:.2f} times per game.")
    if events is not None:                # runs before 2026-09-26 kept no event log
        crashes = [e for e in events if 'FATAL' in e['error'] or e['error'].startswith('LEARNER')]
        errors = [e for e in events if e.get('worker', -1) >= 0]
        lines.append(f"Health: {len(errors)} emulator error{'s' if len(errors) != 1 else ''} (each one recovers "
                     f"on its own){f', {len(crashes)} crash' + ('es' if len(crashes) != 1 else '') if crashes else ', no crashes'}.")
    return lines


@app.get('/api/runs/{run}/report')
def report(run: str, hours: float = 12.0):
    """A morning report: what happened over the last `hours` (0 = the whole run)."""
    d = _run_dir(run)
    rows = _all_games(d)
    if not rows:
        return {'headline': ['No games yet.'], 'hours': [], 'events': [], 'total': None}
    end = rows[-1]['time']
    start = rows[0]['time'] if hours <= 0 else max(rows[0]['time'], end - hours * 3600)
    window = [g for g in rows if g['time'] >= start]
    by_hour = defaultdict(list)
    for g in window:
        lt = time.localtime(g['time'])
        by_hour[time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, lt.tm_hour, 0, 0, 0, 0, -1))].append(g)
    table = []
    for t0 in sorted(by_hour):
        st = _bucket_stats(by_hour[t0])
        st.update(start=t0, label=_clock(t0),
                  versions=[min(g.get('version', 0) for g in by_hour[t0]), max(g.get('version', 0) for g in by_hour[t0])])
        table.append(st)
    total = _bucket_stats(window)
    logged = (d / 'events.jsonl').exists()
    events = [e for e in _events(d) if e.get('time', 0) >= start]
    warn = (_read_json(d / 'config.json', {}) or {}).get('stall_offstage_warn', 0.5)
    cards = [c for c in _scorecards() if c['run'] == d.name and (c['started'] or 0) >= start]
    ph_wins = [g for g in window if g.get('opponent_kind') == 'phillip' and g.get('result') == 'win']
    notable = [e for e in events if e.get('worker', -1) < 0]
    return {'window': {'start': start, 'end': end, 'hours': round((end - start) / 3600, 2)},
            'total': total, 'hours': table, 'scorecards': cards,
            'phillip_wins': [{k: g.get(k) for k in ('time', 'opponent', 'stocks_left', 'stocks_taken', 'version')}
                             for g in ph_wins],
            'events': notable[-60:], 'emulator_errors': len(events) - len(notable), 'event_log': logged,
            'warn': warn,
            'headline': _headline(total, table, events if logged else None, cards, (end - start) / 3600,
                                  ph_wins, warn)}


@app.get('/api/runs/{run}/moves')
def moves(run: str, buckets: int = 40):
    """Share of decisions per move, in consecutive blocks of games."""
    from .actions import NAMES
    rows = [g for g in _all_games(_run_dir(run)) if len(g.get('actions') or ()) == len(NAMES)]
    if not rows:
        return {'names': list(NAMES), 'buckets': []}
    size = max(50, -(-len(rows) // max(1, buckets)))
    out = []
    for i in range(0, len(rows), size):
        block = rows[i:i + size]
        counts = [sum(g['actions'][j] for g in block) for j in range(len(NAMES))]
        total = sum(counts) or 1
        out.append({'first_game': i + 1, 'last_game': i + len(block), 'time': block[-1]['time'],
                    'version': block[-1].get('version'), 'share': [round(c / total, 4) for c in counts]})
    return {'names': list(NAMES), 'size': size, 'buckets': out}


@app.get('/api/runs/{run}/phillip')
def phillip_panel(run: str):
    games = [g for g in _all_games(_run_dir(run))
             if g.get('opponent_kind') == 'phillip' and g.get('result') in COMPLETED]
    keep = ('time', 'opponent', 'result', 'stocks_taken', 'stocks_lost', 'stocks_left', 'damage_dealt',
            'damage_received', 'version', 'frames')
    per = defaultdict(list)
    for g in games:
        per[g['opponent']].append(g)

    def summary(gs):
        n = len(gs)
        if not n:
            return None
        w = sum(g['result'] == 'win' for g in gs)
        return {'games': n, 'wins': w, 'win_rate': _rate(w, n),
                'taken': round(sum(g['stocks_taken'] for g in gs) / n, 2),
                'lost': round(sum(g['stocks_lost'] for g in gs) / n, 2),
                'damage_ratio': round(sum(g['damage_dealt'] for g in gs) / max(1.0, sum(g['damage_received'] for g in gs)), 2),
                'last_win': max((g['time'] for g in gs if g['result'] == 'win'), default=None)}
    return {'overall': summary(games), 'first100': summary(games[:100]), 'last100': summary(games[-100:]),
            'characters': {k: summary(v) for k, v in sorted(per.items(), key=lambda kv: -len(kv[1]))},
            'games': [{k: g.get(k) for k in keep} for g in games[-3000:]],
            'wins': [{k: g.get(k) for k in keep} for g in games if g['result'] == 'win'][-30:]}


@app.get('/api/scorecards')
def scorecards():
    return _scorecards()


@app.get('/api/runs/{run}/checkpoints')
def checkpoints(run: str):
    d = _run_dir(run) / 'checkpoints'
    champs = {c.get('source') for c in _champions()}
    out = []
    for p in sorted(d.glob('policy-*.pt'), reverse=True):
        # policy-0012.34M-v321.pt
        stem = p.stem.split('-')
        out.append({'name': p.name, 'path': str(p), 'frames_m': float(stem[1].rstrip('M')),
                    'version': int(stem[2][1:]), 'mtime': p.stat().st_mtime,
                    'champion': str(p) in champs})
    return out


@app.post('/api/train')
def train(body: dict = Body(...)):
    active = _active_training()
    if active:
        raise HTTPException(409, f'Run {active} is already training. Stop it first.')
    args = ['train']
    if body.get('resume'):
        d = _run_dir(body['resume'])
        args += ['--resume', str(d)]
        log = d / 'train.log'
    else:
        name = ''.join(c for c in (body.get('name') or 'puff') if c.isalnum() or c in '-_')[:40] or 'puff'
        d = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
        d.mkdir(parents=True)
        args += ['--run-dir', str(d)]
        if body.get('init'):
            args += ['--init', str(body['init'])]
        if body.get('level'):
            args += ['--level', str(int(body['level']))]
        log = d / 'train.log'
    for key, flag in (('workers', '--workers'), ('visible', '--visible'), ('selfplay', '--selfplay'),
                      ('visible_speed', '--visible-speed'), ('phillip_workers', '--phillip-workers')):
        if body.get(key) is not None and body.get(key) != '':
            args += [flag, str(body[key])]
    pid = _spawn(args, log)
    return {'run': d.name, 'pid': pid}


@app.post('/api/runs/{run}/stop')
def stop(run: str):
    d = _run_dir(run)
    (d / 'control.json').write_text(json.dumps({'stop': True, 'time': time.time()}))
    return {'ok': True}


@app.post('/api/runs/{run}/kill')
def kill(run: str):
    """Last resort when a run ignores stop: SIGTERM the learner, which still shuts down cleanly."""
    st = _read_json(_run_dir(run) / 'status.json') or {}
    pid = st.get('pid')
    if pid and _pid_alive(pid):
        os.kill(int(pid), signal.SIGTERM)
    return {'ok': True}


def _champions() -> list:
    return _read_json(CHAMPIONS / 'champions.json', []) or []


@app.get('/api/champions')
def champions():
    return _champions()


def qualifying_eval(src: Path, evals_dir: Path | None = None) -> dict | None:
    """The newest *complete* evaluation of exactly these bytes (matched by SHA-256)."""
    from .evaluate import sha256
    digest = sha256(src)
    for d in sorted((evals_dir or RUNS / 'evals').glob('*'), reverse=True):
        rep = _read_json(d / 'report.json')
        if rep and rep.get('state') == 'complete' and rep.get('checkpoint_sha256') == digest:
            return {'id': d.name, 'level': rep.get('level'), 'games_each': rep.get('games_each'),
                    'opponents': rep.get('opponents'), 'overall': rep.get('overall')}
    return None


@app.post('/api/champions')
def crown(body: dict = Body(...)):
    src = Path(body['path']).resolve()
    if RUNS.resolve() not in src.parents or not src.exists():
        raise HTTPException(400, 'Checkpoint must live under runs/.')
    evidence = qualifying_eval(src)
    if evidence is None and not body.get('force'):
        # GPT audit: promotion must rest on a complete evaluation of this exact file.
        raise HTTPException(409, 'No complete evaluation of this exact checkpoint yet. '
                                 'Evaluate it first, or crown it anyway without evidence.')
    CHAMPIONS.mkdir(exist_ok=True)
    name = f"{src.parent.parent.name}--{src.name}"
    shutil.copy2(src, CHAMPIONS / name)
    rows = [c for c in _champions() if c.get('source') != str(src)]
    rows.append({'name': name, 'path': str(CHAMPIONS / name), 'source': str(src),
                 'note': body.get('note', ''), 'crowned': time.time(),
                 'evaluation': evidence, 'without_evaluation': evidence is None})
    (CHAMPIONS / 'champions.json').write_text(json.dumps(rows, indent=2))
    return rows


@app.post('/api/evaluate')
def start_eval(body: dict = Body(...)):
    ck = Path(body['checkpoint']).resolve()
    if not ck.exists():
        raise HTTPException(400, 'No such checkpoint')
    args = ['evaluate', str(ck), '--level', str(int(body.get('level', 9))),
            '--games', str(int(body.get('games', 10))),
            '--opponents', ','.join(body.get('opponents') or ['FOX', 'FALCO', 'MARTH', 'CPTFALCON', 'PEACH', 'JIGGLYPUFF'])]
    if body.get('workers'):
        args += ['--workers', str(int(body['workers']))]
    if body.get('kind') in ('cpu', 'phillip'):
        args += ['--kind', body['kind']]
    if body.get('greedy'):
        args.append('--greedy')
    pid = _spawn(args, ROOT / '.runtime' / 'evaluate.log')
    return {'pid': pid}


VIEWER = ROOT / '.runtime' / 'viewer.json'
ENDLESS = 99          # "until stopped": more games than anyone watches in a sitting


def _viewer() -> dict | None:
    """The viewer window the dashboard opened, if it is still open."""
    v = _read_json(VIEWER)
    if not v or not _pid_alive(v.get('pid')):
        return None
    rep = _read_json(Path(v['dir']) / 'report.json') or {}
    games = rep.get('games') or []
    v['played'] = len(games)
    v['wins'] = sum(g.get('result') == 'win' for g in games)
    v['last'] = {k: games[-1].get(k) for k in ('result', 'stocks_left', 'opp_stocks_left')} if games else None
    v['stopping'] = bool(v.get('stop_requested'))
    v['state'] = rep.get('state')
    v['results'] = [g.get('result') for g in games][-40:]
    return v


def _stop_viewer(v: dict):
    """Ask the viewer to cancel (it reaps its own Dolphin); force it after 15 s."""
    from .learner import _reap_dolphins
    try:
        os.kill(int(v['pid']), signal.SIGINT)
    except OSError:
        return

    def force():
        deadline = time.time() + 15
        while time.time() < deadline and _pid_alive(v['pid']):
            time.sleep(0.5)
        if _pid_alive(v['pid']):
            try:
                os.kill(int(v['pid']), signal.SIGKILL)
            except OSError:
                pass
        _reap_dolphins(Path(v['dir']))
    threading.Thread(target=force, daemon=True).start()


@app.post('/api/watch')
def watch(body: dict = Body(...)):
    ck = Path(body['checkpoint']).resolve()
    if not ck.exists():
        raise HTTPException(400, 'No such checkpoint')
    kind = body.get('kind', 'cpu')
    if kind not in ('cpu', 'phillip'):
        raise HTTPException(400, "kind must be 'cpu' or 'phillip'")
    games = int(body.get('games', 3)) or ENDLESS
    out = RUNS / 'evals' / f"{time.strftime('%Y%m%d-%H%M%S')}-watch"
    out.mkdir(parents=True, exist_ok=True)
    opponent = str(body.get('opponent', 'FOX')).upper()
    args = ['watch', str(ck), '--opponent', opponent, '--kind', kind,
            '--level', str(int(body.get('level', 9))), '--games', str(games),
            '--speed', str(float(body.get('speed', 1.0))), '--dir', str(out)]
    pid = _start_viewer(args, {'dir': str(out), 'checkpoint': str(ck), 'name': ck.name, 'kind': kind,
                               'opponent': opponent, 'level': int(body.get('level', 9)), 'games': games})
    return {'pid': pid}


def _start_viewer(args: list[str], meta: dict) -> int:
    """Open one Dolphin window for a person to look at (watch) or play (versus),
    closing whichever one is already open."""
    old = _viewer()
    if old:
        _stop_viewer(old)
        deadline = time.time() + 15
        while time.time() < deadline and _pid_alive(old['pid']):
            time.sleep(0.3)
    pid = _spawn(args, ROOT / '.runtime' / f"{meta['kind'] if meta['kind'] == 'versus' else 'watch'}.log")
    VIEWER.write_text(json.dumps({'pid': pid, 'started': time.time(), **meta}))
    return pid


_controllers_cache: dict = {'at': 0.0, 'found': []}


@app.get('/api/play/options')
def play_options(refresh: bool = False):
    from .dolphin import detect_controllers
    from .versus import CHARACTERS
    if refresh or time.time() - _controllers_cache['at'] > 30:
        _controllers_cache.update(at=time.time(), found=detect_controllers())
    models, seen = [], set()
    for i, r in enumerate(showcase()['rows']):
        models.append({'path': r['path'], 'label': f"#{i + 1} · {r['run'] or 'checkpoint'} v{r['version']} · {r['score']:.0f} pts"})
        seen.add(r['path'])
    for c in _champions():
        if c.get('source') not in seen and Path(c['path']).exists():
            models.append({'path': c['path'], 'label': f"👑 {c['name']}"})
            seen.add(c.get('source'))
    for d in sorted((p for p in RUNS.glob('*') if (p / 'checkpoints').is_dir()), reverse=True)[:2]:
        for p in sorted((d / 'checkpoints').glob('policy-*.pt'), key=lambda q: q.stat().st_mtime, reverse=True)[:6]:
            if str(p) not in seen:
                models.append({'path': str(p), 'label': f"{d.name} v{_version_of(p.name)} · untested"})
    return {'controllers': _controllers_cache['found'], 'characters': list(CHARACTERS), 'models': models,
            'viewer': _viewer()}


@app.post('/api/versus')
def start_versus(body: dict = Body(...)):
    from .versus import CHARACTERS
    ck = Path(body['checkpoint']).resolve()
    if not ck.exists():
        raise HTTPException(400, 'No such checkpoint')
    character = str(body.get('character', 'FOX')).upper()
    if character not in CHARACTERS:
        raise HTTPException(400, f'Pick one of {", ".join(CHARACTERS)}')
    layout = body.get('layout', 'switch')
    if layout not in ('switch', 'xbox'):
        raise HTTPException(400, "layout must be 'switch' or 'xbox'")
    device = body.get('device') or (_controllers_cache['found'] or [None])[0]
    if not device:
        from .dolphin import detect_controllers
        found = detect_controllers()
        _controllers_cache.update(at=time.time(), found=found)
        if not found:
            raise HTTPException(409, 'No game controller found. Turn it on / reconnect it, then press Check again.')
        device = found[0]
    delay = max(0, min(30, int(body.get('delay', 0))))
    out = RUNS / 'evals' / f"{time.strftime('%Y%m%d-%H%M%S')}-versus"
    out.mkdir(parents=True, exist_ok=True)
    args = ['versus', str(ck), '--character', character, '--layout', layout, '--device', device,
            '--delay', str(delay), '--dir', str(out)]
    pid = _start_viewer(args, {'kind': 'versus', 'dir': str(out), 'checkpoint': str(ck), 'name': ck.name,
                               'opponent': character, 'level': 0, 'games': ENDLESS, 'layout': layout,
                               'delay': delay, 'device': device})
    return {'pid': pid}


@app.get('/api/watch')
def watch_status():
    return {'viewer': _viewer()}


@app.post('/api/watch/stop')
def watch_stop():
    v = _viewer()
    if v:
        _stop_viewer(v)
        VIEWER.write_text(json.dumps({**{k: x for k, x in v.items() if k not in ('played', 'wins', 'last', 'stopping', 'state')},
                                      'stop_requested': time.time()}))
    return {'ok': True, 'was_open': bool(v)}


# ------------------------------------------------------------------ showcase
def _report_rows() -> list[tuple[Path, dict]]:
    out = []
    for d in sorted((RUNS / 'evals').glob('*')):
        rep = _read_json(d / 'report.json')
        if not rep or d.name.endswith('-watch') or rep.get('label') == 'watch':
            continue
        if rep.get('kind') is None:
            rep['kind'] = 'cpu'
        o = rep.get('overall') or {}
        if rep.get('state') == 'finished':
            rep['state'] = 'complete' if o.get('games', 0) >= len(rep.get('opponents') or []) * (rep.get('games_each') or 0) else 'incomplete'
        out.append((d, rep))
    return out


def _pool(reps: list[dict]) -> dict:
    """Combine several evaluations of the same file into one record."""
    games = [g for r in reps for g in (r.get('games') or [])]
    by = defaultdict(lambda: [0, 0, 0])
    for g in games:
        b = by[g['opponent']]
        b[0] += g['result'] == 'win'
        b[1] += 1
        b[2] += g.get('stocks_taken', 0)
    n = len(games)
    return {'games': n, 'wins': sum(g['result'] == 'win' for g in games),
            'taken': sum(g.get('stocks_taken', 0) for g in games) / n if n else 0,
            'lost': sum(g.get('stocks_lost', 0) for g in games) / n if n else 0,
            'by': {k: {'wins': v[0], 'games': v[1], 'taken': round(v[2] / v[1], 2)} for k, v in by.items()}}


def impressiveness(cpu: dict | None, ph: dict | None) -> dict:
    """0-100. Up to 60 from the CPU-9 scorecard (50 for the win rate, 10 for winning by a
    lot of stocks) and up to 40 from Phillip (25 for stocks taken, 15 for wins): a stock
    off a learned opponent is worth far more than a CPU win."""
    cpu_pts = ph_pts = 0.0
    if cpu and cpu['games']:
        cpu_pts = 50 * cpu['wins'] / cpu['games'] + 10 * max(0.0, cpu['taken'] - cpu['lost']) / 4
    if ph and ph['games']:
        ph_pts = 25 * min(1.0, ph['taken'] / 4) + 15 * ph['wins'] / ph['games']
    return {'score': round(cpu_pts + ph_pts, 1), 'cpu_points': round(cpu_pts, 1), 'phillip_points': round(ph_pts, 1)}


def _badges(cpu, ph, crowned) -> list[str]:
    out = ['👑 Crowned'] if crowned else []
    if cpu and cpu['games']:
        out.append(f"{cpu['wins'] / cpu['games']:.0%} vs level-9 CPUs")
        perfect = [_name(k) for k, v in sorted(cpu['by'].items()) if v['games'] >= 5 and v['wins'] == v['games']]
        if perfect:
            out.append('Perfect vs ' + ', '.join(perfect))
    if ph and ph['games']:
        for k, v in sorted(ph['by'].items(), key=lambda kv: -kv[1]['wins']):
            if v['wins']:
                out.append(f"🤖 Beat Phillip's {_name(k)}" + (f" ×{v['wins']}" if v['wins'] > 1 else ''))
        out.append(f"{ph['taken']:.1f} stocks per game off Phillip")
    return out


@app.get('/api/showcase')
def showcase():
    from .phillip import SUPPORTED
    groups = defaultdict(lambda: {'cpu': [], 'phillip': [], 'evals': []})
    for d, rep in _report_rows():
        key = rep.get('checkpoint_sha256') or rep.get('checkpoint')   # early reports had no hash
        if rep.get('state') != 'complete' or not key:
            continue
        g = groups[key]
        g['evals'].append(rep)
        if rep['kind'] == 'phillip':
            g['phillip'].append(rep)
        elif all(rep.get(k) == v for k, v in SCORECARD.items()):
            g['cpu'].append(rep)
    champs = {c.get('source') for c in _champions()}
    rows = []
    for sha, g in groups.items():
        if not g['cpu'] and not g['phillip']:
            continue
        last = g['evals'][-1]
        src = Path(last.get('checkpoint', ''))
        path = src if src.exists() else Path(last.get('checkpoint_copy', ''))
        cpu = _pool(g['cpu']) if g['cpu'] else None
        ph = _pool(g['phillip']) if g['phillip'] else None
        crowned = str(src) in champs
        rows.append({'sha': sha, 'path': str(path), 'name': src.name,
                     'run': src.parent.parent.name if src.parent.name == 'checkpoints' else None,
                     'version': last.get('checkpoint_version') or _version_of(src.name),
                     'frames': last.get('checkpoint_frames'), 'tested': max(r.get('started') or 0 for r in g['evals']),
                     'crowned': crowned, 'cpu': cpu, 'phillip': ph, **impressiveness(cpu, ph),
                     'badges': _badges(cpu, ph, crowned)})
    rows.sort(key=lambda r: (-r['score'], -(r['tested'] or 0)))
    return {'rows': rows, 'viewer': _viewer(),
            'cpu_characters': list(Config.load().opponents), 'phillip_characters': list(SUPPORTED)}


@app.get('/api/evals')
def evals():
    out = []
    for d in sorted((RUNS / 'evals').glob('*'), reverse=True)[:40]:
        rep = _read_json(d / 'report.json')
        if not rep or d.name.endswith('-watch'):
            continue
        rep.pop('games', None)
        rep['rejected'] = len(rep.get('rejected') or [])
        rep['id'] = d.name
        out.append(rep)
    return out


@app.get('/api/config')
def get_config():
    return Config.load().to_dict()


@app.post('/api/config')
def set_config(body: dict = Body(...)):
    path = ROOT / 'config.local.json'
    current = _read_json(path, {}) or {}
    current.update(body)
    try:
        Config(**{**Config().to_dict(), **current}).validate()
    except Exception as exc:
        raise HTTPException(400, str(exc))
    path.write_text(json.dumps(current, indent=2))
    return current


@app.get('/api/system')
def system():
    load = os.getloadavg()
    usage = shutil.disk_usage(ROOT)
    return {'load': [round(x, 2) for x in load], 'cores': os.cpu_count(),
            'disk_free_gb': round(usage.free / 1e9, 1), 'active_run': _active_training()}


def serve(port: int = 8777, open_browser: bool = True):
    import uvicorn
    if open_browser:
        import threading
        threading.Timer(1.0, lambda: webbrowser.open(f'http://127.0.0.1:{port}')).start()
    uvicorn.run(app, host='127.0.0.1', port=port, log_level='warning')
