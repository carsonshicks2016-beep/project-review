"""A companion window that opens with training and fills whatever screen the emulators leave.

The emulators are tiled across the desktop by `layout`, which leaves one large empty
band. This claims that band: a chrome-less Chrome window pinned to an exact rectangle,
showing the run's lineage, its trend, and every slot's health at a glance.

It reads the dashboard's API and nothing else -- no model weights, no trainer imports,
no shared state with the run. Every failure here is swallowed: an optional window must
never be able to interrupt training.

One window serves every run. It renders whichever run is active, so the restarts that
punctuate a training session retarget it for free rather than piling up windows.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .config import ROOT
from . import layout

CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
MINIMUM = (380, 200)          # below this the leftover is not worth a window
HISTORY = 300                 # matches sent to the chart
# Points kept clear beneath the emulators. Tall enough for the chart to say something,
# short enough that ten emulators still get a usable render area.
BAND = 300


def enabled(root=ROOT):
    return read(Path(root)/'panel-settings.json').get('enabled', True) is not False


def read(path, default=None):
    try: return json.loads(Path(path).read_text())
    except (OSError, ValueError): return {} if default is None else default


# --- where it goes -----------------------------------------------------------

def remainder(boxes, screen=None):
    """The largest band the tiled emulators leave free.

    Deliberately not a general empty-rectangle search: the emulators are laid out as a
    grid, so the leftovers are the strips beside and beneath that grid's bounding box.
    Picking among those three gives the same answer with a result you can predict by
    looking at the screen, and it never proposes a sliver between two windows.
    """
    screen = screen or layout.SCREEN
    top = layout.MENU_BAR
    if not boxes:
        return dict(x=layout.MARGIN, y=top, width=screen[0]-2*layout.MARGIN,
                    height=screen[1]-top-layout.MARGIN)
    left = min(b['x'] for b in boxes)
    right = max(b['x']+b['width'] for b in boxes)
    bottom = max(b['y']+b['height']+layout.TITLE_BAR for b in boxes)
    used_top = min(b['y'] for b in boxes)
    gap = layout.GAP

    candidates = [
        dict(x=layout.MARGIN, y=bottom+gap,
             width=screen[0]-2*layout.MARGIN, height=screen[1]-layout.MARGIN-(bottom+gap)),
        dict(x=right+gap, y=top,
             width=screen[0]-layout.MARGIN-(right+gap), height=screen[1]-top-layout.MARGIN),
        dict(x=layout.MARGIN, y=top,
             width=left-gap-layout.MARGIN, height=screen[1]-top-layout.MARGIN),
        dict(x=layout.MARGIN, y=top,
             width=screen[0]-2*layout.MARGIN, height=used_top-gap-top),
    ]
    usable = [c for c in candidates if c['width'] >= MINIMUM[0] and c['height'] >= MINIMUM[1]]
    if not usable: return None
    return max(usable, key=lambda c: c['width']*c['height'])


# --- what it shows -----------------------------------------------------------

def lineage(directory, root):
    """The chain of checkpoints this run descends from, oldest first.

    Walks the frozen `checkpoint_identity` recorded at each resume rather than the
    checkpoint's current metadata: latest.zip keeps being rewritten, so its step count
    today is not what the run that consumed it actually started from.
    """
    chain, seen = [], set()
    directory, root = Path(directory), Path(root)
    current, inherited = directory, None
    while len(chain) < 32 and current not in seen:
        seen.add(current)
        request = read(current/'request.json')
        status = read(current/'status.json')
        meta = read(current/'latest.json')
        input_meta = read(current/'input-policy.json')
        recurrent = bool(request.get('recurrent') or status.get('architecture') == 'recurrent' or meta.get('architecture') == 'recurrent' or input_meta.get('architecture') == 'recurrent')
        chain.append(dict(
            id=current.name,
            live=current == directory,
            steps=status.get('steps') if current == directory else (inherited or {}).get('steps'),
            kind='Anchored Recurrent PPO' if (request.get('anchor') and recurrent) else
                 'Recurrent PPO' if recurrent else
                 'Imitation' if meta.get('imitation_samples') else
                 'Anchored PPO' if request.get('anchor') else 'PPO',
            recurrent=recurrent,
            frames=meta.get('imitation_samples'),
            missing=not current.is_dir()))
        identity = request.get('checkpoint_identity') or {}
        source = identity.get('source') or request.get('checkpoint')
        if not source: break
        parent = (root/source).resolve()
        if not parent.is_relative_to(root) or parent.parent == current: break
        inherited = identity
        current = parent.parent
    return list(reversed(chain))


def payload(manager):
    """One poll's worth of everything the window draws."""
    runs = manager.list_runs(include_slots=True)
    active = next((r for r in runs if r.get('status') in ('starting', 'running', 'paused')), None)
    run = active or (runs[0] if runs else None)
    if not run:
        return dict(run=None, note='No runs yet')
    directory = manager.runs/run['id']
    rows, _ = manager.matches.read(directory/'matches.jsonl')
    played = [r for r in rows if r.get('kind') in ('ppo', 'self-play')]
    history = [dict(n=r['number'], cpu=r.get('cpu_level'), result=r.get('result'),
                    us=(r.get('players') or {}).get('1', {}).get('stocks'),
                    them=(r.get('players') or {}).get('2', {}).get('stocks'),
                    ret=r.get('return'))
               for r in played[-HISTORY:]]
    request = read(directory/'request.json')
    status = read(directory/'status.json')
    initial = run.get('initial_steps')
    if initial is None:
        initial = (request.get('checkpoint_identity') or {}).get('steps', 0)
    
    input_meta = read(directory/'input-policy.json')
    latest_meta = read(directory/'latest.json')
    recurrent = bool(request.get('recurrent') or status.get('architecture') == 'recurrent' or run.get('architecture') == 'recurrent' or input_meta.get('architecture') == 'recurrent' or latest_meta.get('architecture') == 'recurrent')
    slots = run.get('slots') or []
    live_frame = status.get('frame') or (slots[0].get('frame') if slots else 0) or 0
    raw_action = status.get('action') or (slots[0].get('action') if slots else None) or 'neutral'

    # Compute curriculum promotion progress
    cpu_level = run.get('cpu_level', 1) or 1
    at_level = [m for m in history if m.get('cpu') == cpu_level]
    recent_level = at_level[-20:]
    level_wins = sum(1 for m in recent_level if m.get('result') == 'win')
    level_matches = len(recent_level)
    level_wr = (level_wins / level_matches) if level_matches > 0 else 0.0
    streak = 0
    for m in reversed(at_level):
        if m.get('result') == 'win':
            if streak >= 0: streak += 1
            else: break
        elif m.get('result') == 'loss':
            if streak <= 0: streak -= 1
            else: break
        else: break

    # 10 wins out of 20 is the 50% threshold to promote
    wins_needed = max(0, 10 - level_wins)
    eval_pct = min(100.0, (level_matches / 20.0) * 100.0)
    level_up = dict(
        level=cpu_level,
        next_level=min(9, cpu_level + 1),
        matches=level_matches,
        wins=level_wins,
        win_rate=level_wr,
        target_matches=20,
        wins_needed=wins_needed,
        progress_pct=eval_pct,
        streak=streak,
        ready=bool(level_matches >= 20 and level_wr >= 0.5 and cpu_level < 9)
    )

    live_players = status.get('players') or (slots[0].get('players') if slots else None) or {}

    # Compute real-time match advantage (-100 to +100)
    p1 = live_players.get('1') or {}
    p2 = live_players.get('2') or {}
    stocks_diff = (p1.get('stocks', 3) - p2.get('stocks', 3)) * 80.0
    pct_diff = p2.get('percent', 0.0) - p1.get('percent', 0.0)
    stage_ctrl = 0.0
    if abs(p1.get('x', 0.0)) > 85.56: stage_ctrl -= 35.0
    if abs(p2.get('x', 0.0)) > 85.56: stage_ctrl += 35.0
    advantage_score = max(-100.0, min(100.0, (stocks_diff + pct_diff + stage_ctrl) / 2.0))

    # Milestones across match history
    total_stocks_taken = sum(max(0, 3 - int(m.get('them') or 3)) for m in history)
    best_return = max((float(m.get('ret') or -999) for m in history), default=0.0)
    wins_total = sum(1 for m in history if m.get('result') == 'win')
    milestones = dict(
        stocks_taken=total_stocks_taken,
        best_return=round(best_return, 2),
        wins=wins_total,
        matches=len(history)
    )

    return dict(
        run=dict(id=run['id'], status=run.get('status'), phase=run.get('phase'),
                 steps=run.get('steps', 0), initial=initial or 0,
                 budget=run.get('requested_steps'), elapsed=run.get('elapsed'),
                 cpu=cpu_level, matches=run.get('matches'), wins=run.get('wins'),
                 character=(run.get('config') or {}).get('character', 'Agent'),
                 anchor=Path(request['anchor']).stem if request.get('anchor') else None,
                 anchor_weight=run.get('anchor_weight'),
                 anchor_samples=run.get('anchor_samples'),
                 recurrent=recurrent,
                 lstm_hidden_size=request.get('lstm_hidden_size', 128) if recurrent else None,
                 live_frame=int(live_frame),
                 action=raw_action,
                 players=live_players,
                 advantage=round(advantage_score, 1),
                 milestones=milestones,
                 level_up=level_up,
                 live=run.get('status') in ('starting', 'running', 'paused')),
        slots=slots,
        history=history,
        lineage=lineage(directory, manager.root),
        live_players=live_players,
        advantage=round(advantage_score, 1),
        milestones=milestones,
        version=5,
        served=time.time())


# --- opening it --------------------------------------------------------------

def server_port(root=ROOT):
    return read(Path(root)/'.runtime/server.json').get('port')


def running(record):
    pid = record.get('pid')
    if not pid: return False
    try: os.kill(int(pid), 0); return True
    except (ProcessLookupError, ValueError, TypeError): return False
    except PermissionError: return True


def open_window(run_directory, root=ROOT, envs=None):
    """Open the companion, or leave the existing one alone.

    Never raises. The window is a convenience; a training run that dies because a
    browser would not start is a strictly worse trade than no window.
    """
    try:
        root = Path(root).resolve()
        if sys.platform != 'darwin' or not Path(CHROME).exists(): return None
        if not enabled(root): return None
        runtime = root/'.runtime'; runtime.mkdir(exist_ok=True)
        record = read(runtime/'panel.json')
        # A live window already follows whichever run is active, so a restart needs
        # nothing from us.
        if running(record): return record.get('pid')
        port = server_port(root)
        if not port: return None
        count = envs or read(Path(run_directory)/'request.json').get('envs') or 1
        box = remainder(layout.load(int(count), root/'window-layout.json', reserve=BAND))
        if not box: return None
        command = [CHROME, f'--app=http://127.0.0.1:{port}/panel',
                   f"--window-position={box['x']},{box['y']}",
                   f"--window-size={box['width']},{box['height']}",
                   f"--user-data-dir={runtime/'panel-profile'}",
                   '--no-first-run', '--no-default-browser-check', '--disable-extensions']
        with (runtime/'panel.log').open('a') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        (runtime/'panel.json').write_text(json.dumps(dict(pid=child.pid, box=box, opened=time.time())))
        return child.pid
    except Exception:
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Open the training companion window')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--envs', type=int)
    args = parser.parse_args()
    pid = open_window(args.run_dir, envs=args.envs)
    print(f'panel pid {pid}' if pid else 'panel not opened (no Chrome, no server, or no free space)')


if __name__ == '__main__':
    main()
