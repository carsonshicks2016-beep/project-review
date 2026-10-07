"""Me vs AI: a person on port 1 with a real controller, a trained checkpoint as Puff on port 2.

The harness picks both characters itself (the person's port listens to the harness's
pipe as well as the controller, see dolphin.human_bindings), then lets go of port 1 for
the rest of the session. Games repeat on Final Destination, the only stage the policy
was trained on, until the process is stopped (SIGINT from the dashboard's Stop button).

The policy plays Puff because that is all it was trained as; the person may pick any
character the menus can select. Its observation of the opponent is the same as in
training, so a human opponent is just another opponent to it.
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import melee
import numpy as np

from . import actions, contract, observation, reward
from .config import RUNS, Config
from .dolphin import EmulatorError, Emulator, Setup
from .model import Actor, load_policy

LIVE = (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH)
PORT_OFFSET = 250                      # clear of training (0-15), eval (100), watch (150), scorecards (200)
BOT, HUMAN = 2, 1
# Characters the menus can pick for a human-controlled port (not Sheik or Ice Climbers).
CHARACTERS = ('FOX', 'FALCO', 'MARTH', 'CPTFALCON', 'PEACH', 'JIGGLYPUFF', 'SAMUS', 'PIKACHU',
              'LUIGI', 'DOC', 'MARIO', 'YOSHI', 'GANONDORF', 'LINK', 'YLINK', 'ZELDA', 'ROY', 'MEWTWO',
              'PICHU', 'NESS', 'KIRBY', 'BOWSER', 'DK', 'GAMEANDWATCH')
CHARACTERS = tuple(c for c in CHARACTERS if c in melee.Character.__members__)


def versus(checkpoint: str, character: str, device: str, layout: str = 'switch', delay: int = 0,
           out_dir: str | None = None, config: str | None = None, progress=print) -> dict:
    character = character.upper()
    if character not in CHARACTERS:
        raise ValueError(f'character must be one of {CHARACTERS}')
    cfg = Config.load(config, workers=1, visible_workers=1, visible_speed=1.0).validate()
    cfg.base_port += PORT_OFFSET
    src = Path(checkpoint).resolve()
    payload = load_policy(src)
    act_every = contract.upgrade(payload.get('contract')).get('act_every') or cfg.act_every
    actor = Actor(int(payload.get('hidden', cfg.hidden)))
    actor.load_state(payload['state'], payload['version'])
    out = Path(out_dir) if out_dir else RUNS / 'evals' / f"{time.strftime('%Y%m%d-%H%M%S')}-versus"
    out.mkdir(parents=True, exist_ok=True)
    report = {'label': 'versus', 'kind': 'versus', 'state': 'starting', 'checkpoint': str(src),
              'checkpoint_version': payload.get('version'), 'human_character': character, 'layout': layout,
              'delay_frames': delay, 'started': time.time(), 'games': []}

    def save():
        tmp = out / '.report.tmp'
        tmp.write_text(json.dumps(report, indent=2, default=float))
        tmp.replace(out / 'report.json')

    save()
    emu = Emulator(cfg, 0, out / 'User', visible=True, log_path=out / 'dolphin.log',
                   human={'device': device, 'layout': layout})
    setup = Setup(character, 'JIGGLYPUFF', 0)
    rng = np.random.default_rng()
    try:
        progress('Starting Dolphin…')
        emu.launch()
        progress('Picking characters (hands off the controller for a few seconds)…')
        g = emu.enter_match(setup)
        report['state'] = 'playing'
        save()
        progress('Go!')
        ex = actions.Executor(act_every, cfg.auto_lcancel, cfg.auto_tech)
        seen: deque = deque(maxlen=delay + 1)          # what the bot "sees", `delay` frames late
        prev_action, prev, ended, first = -1, None, False, True
        while True:
            emu.pads[HUMAN - 1].release_all()             # the harness never touches port 1 in a game
            live = g.menu_state in LIVE and HUMAN in g.players and BOT in g.players and g.frame >= 0 and not ended
            if live:
                seen.append(g)
                me = g.players[BOT]
                if first or ex.done:
                    view = seen[0]
                    floats, ids = observation.encode(view, BOT, prev_action, cfg.max_game_frames)
                    mask = actions.legal_mask(me)             # legality from now, not the delayed view
                    a, _ = actor.act(floats[None], ids[None], mask[None], rng)
                    ex.start(int(a[0]))
                    prev_action, first = int(a[0]), False
                actions.write(emu.pads[BOT - 1], ex.next_input(me))
            else:
                emu.pads[BOT - 1].release_all()
                ex.released()
            g2 = emu.frame()
            if g2.menu_state not in LIVE or HUMAN not in g2.players or BOT not in g2.players:
                # Results screen after Sudden Death, or the person quit out: set it up again.
                progress('Back to the match…')
                g = emu.enter_match(setup)
                prev, ended, first, prev_action = None, False, True, -1
                seen.clear()
                continue
            if prev is not None and g2.frame < prev.frame:     # Instant Match: the rematch began
                prev, ended, first, prev_action = None, False, True, -1
                seen.clear()
            elif g2.frame >= 0 and not ended:
                if prev is not None and (int(g2.players[1].stock) == 0 or int(g2.players[2].stock) == 0):
                    res = reward.result_of(g2, BOT)
                    report['games'].append({
                        'result': res, 'time': time.time(), 'stocks_left': int(g2.players[BOT].stock),
                        'opp_stocks_left': int(g2.players[HUMAN].stock),
                        'damage_left': float(g2.players[BOT].percent), 'opp_damage_left': float(g2.players[HUMAN].percent)})
                    save()
                    you = sum(x['result'] == 'loss' for x in report['games'])
                    progress(f"Game {len(report['games'])}: {'Puff' if res == 'win' else 'You'} won. "
                             f"Score: you {you} – Puff {len(report['games']) - you}")
                    ended = True
                prev = g2
            g = g2
    except KeyboardInterrupt:
        report['state'] = 'stopped'
    except EmulatorError as exc:
        report['state'] = 'error'
        report['error'] = str(exc)
        progress(f'Emulator error: {exc}')
    finally:
        emu.stop()
        if report['state'] in ('starting', 'playing'):
            report['state'] = 'stopped'
        report['finished'] = time.time()
        save()
    return report
