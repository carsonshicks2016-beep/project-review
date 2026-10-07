"""Command line: train, evaluate, watch, bench, serve."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import RUNS, Config


def _new_run_dir(name: str | None) -> Path:
    stamp = time.strftime('%Y%m%d-%H%M%S')
    return RUNS / (f'{stamp}-{name}' if name else stamp)


def _hard_exit(code: int):
    """Leave no child behind and never wait on one: interpreter exit joins every
    multiprocessing child, and a stuck one would hold this process forever."""
    import multiprocessing as mp
    import os
    for child in mp.active_children():
        child.kill()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def cmd_train(a):
    import traceback
    code = 0
    try:
        _train(a)
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        _hard_exit(code)


def _train(a):
    from .learner import Learner
    overrides = dict(workers=a.workers, visible_workers=a.visible, visible_speed=a.visible_speed,
                     total_frames=a.frames,
                     device=a.device, selfplay_fraction=a.selfplay, cpu_start_level=a.level,
                     phillip_workers=a.phillip_workers)
    if a.resume:
        run_dir = Path(a.resume)
        if not run_dir.is_absolute() and not run_dir.exists():
            run_dir = RUNS / a.resume
        cfg = Config.load(run_dir / 'config.json', **overrides).validate()
        Learner(cfg, run_dir, resume=True).run()
    else:
        cfg = Config.load(a.config, **overrides).validate()
        run_dir = Path(a.run_dir) if a.run_dir else _new_run_dir(a.name)
        print(f'Run: {run_dir}', flush=True)
        Learner(cfg, run_dir, init_from=a.init).run()


def cmd_evaluate(a):
    from .evaluate import evaluate
    cfg = Config.load(a.config, workers=a.workers).validate()
    report = evaluate(cfg, a.checkpoint, opponents=a.opponents.split(','), level=a.level,
                      games=a.games, greedy=a.greedy, out=a.out, out_dir=a.dir, kind=a.kind, label=a.label)
    print(json.dumps({'state': report['state'], 'counted': report['counted'], **report['overall']}, indent=2))
    if report['state'] != 'complete':
        raise SystemExit(f"Evaluation {report['state']}: not every opponent reached {a.games} games.")


def cmd_watch(a):
    from .evaluate import evaluate
    cfg = Config.load(a.config, workers=1, visible_workers=1, visible_speed=a.speed, viewer_audio=True).validate()
    evaluate(cfg, a.checkpoint, opponents=[a.opponent], level=a.level, games=a.games, greedy=False,
             out=None, label='watch', kind=a.kind, out_dir=a.dir)


def cmd_versus(a):
    from .dolphin import detect_controllers
    from .versus import versus
    device = a.device
    if not device:
        found = detect_controllers()
        if not found:
            raise SystemExit('No game controller found. Connect it (Bluetooth or USB) and try again.')
        device = found[0]
    print(f'Controller: {device} ({a.layout} layout)', flush=True)
    versus(a.checkpoint, a.character, device, layout=a.layout, delay=a.delay, out_dir=a.dir, config=a.config,
           progress=lambda m: print(m, flush=True))


def cmd_bench(a):
    from .bench import bench
    cfg = Config.load(a.config).validate()
    bench(cfg, [int(x) for x in a.counts.split(',')], a.seconds)


def cmd_serve(a):
    from .server import serve
    serve(a.port, open_browser=not a.no_browser)


def _guarded(fn):
    def run(a):
        import signal
        import traceback
        # A process started in the background by a script inherits SIGINT *ignored*; the
        # dashboard's Stop relies on it (evaluations and viewers cancel cleanly on it).
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, signal.default_int_handler)
        code = 0
        try:
            fn(a)
        except KeyboardInterrupt:
            code = 130
        except SystemExit as exc:
            if exc.code not in (None, 0):
                print(exc.code if isinstance(exc.code, str) else '', file=sys.stderr)
                code = exc.code if isinstance(exc.code, int) else 2
        except Exception:
            traceback.print_exc()
            code = 1
        finally:
            _hard_exit(code)
    return run


def main(argv=None):
    p = argparse.ArgumentParser(prog='puffbot', description='Melee Bot v2: Jigglypuff specialist')
    p.add_argument('--config', default=None, help='config JSON (default: config.local.json)')
    sub = p.add_subparsers(dest='cmd', required=True)

    t = sub.add_parser('train', help='start or resume a training run')
    t.add_argument('--name')
    t.add_argument('--run-dir', help='explicit directory for a new run')
    t.add_argument('--resume', help='run directory (or name under runs/) to continue')
    t.add_argument('--init', help='start from these weights (policy .pt or checkpoint)')
    t.add_argument('--workers', type=int)
    t.add_argument('--visible', type=int, help='workers to render in a window')
    t.add_argument('--visible-speed', type=float, help='speed of visible workers: 0 unlimited, 1 real time')
    t.add_argument('--frames', type=int, help='stop after this many trained frames')
    t.add_argument('--device', choices=['auto', 'cpu', 'mps'])
    t.add_argument('--selfplay', type=float)
    t.add_argument('--level', type=int, help='starting CPU frontier')
    t.add_argument('--phillip-workers', type=int, help='workers that always play Phillip (Slippi-AI)')
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser('evaluate', help='fixed-protocol evaluation of a checkpoint')
    e.add_argument('checkpoint')
    e.add_argument('--opponents', default='FOX,FALCO,MARTH,CPTFALCON,PEACH,JIGGLYPUFF')
    e.add_argument('--level', type=int, default=9, help='CPU level (cpu opponents only)')
    e.add_argument('--kind', choices=['cpu', 'phillip'], default='cpu',
                   help="'phillip' plays Slippi-AI medium-v2 instead of the game's CPU")
    e.add_argument('--games', type=int, default=10, help='games per opponent')
    e.add_argument('--workers', type=int)
    e.add_argument('--greedy', action='store_true')
    e.add_argument('--out', help='also write the report JSON here')
    e.add_argument('--dir', help='evaluation directory (default: runs/evals/<time>-<label>)')
    e.add_argument('--label', default='eval', choices=['eval', 'scorecard'],
                   help="'scorecard' marks an automatic scorecard (own ports, charted as one)")
    e.set_defaults(fn=_guarded(cmd_evaluate))

    w = sub.add_parser('watch', help='play a checkpoint in a visible window')
    w.add_argument('checkpoint')
    w.add_argument('--opponent', default='FOX')
    w.add_argument('--level', type=int, default=9)
    w.add_argument('--games', type=int, default=3)
    w.add_argument('--speed', type=float, default=1.0, help='1.0 = real time, 0 = unlimited')
    w.add_argument('--kind', choices=['cpu', 'phillip'], default='cpu', help="'phillip' watches it play Slippi-AI")
    w.add_argument('--dir', help='directory for the viewer\'s files (default: runs/evals/<time>-watch)')
    w.set_defaults(fn=_guarded(cmd_watch))

    v = sub.add_parser('versus', help='play against a checkpoint yourself, with a game controller')
    v.add_argument('checkpoint')
    v.add_argument('--character', default='FOX', help='your character (the bot is always Puff)')
    v.add_argument('--layout', choices=['switch', 'xbox'], default='switch',
                   help="'switch': A right, B bottom, X top, Y left (Switch Pro positions); 'xbox': as printed")
    v.add_argument('--device', help='Dolphin device name, e.g. "SDL/0/Xbox One S Controller" (default: detect)')
    v.add_argument('--delay', type=int, default=0, help='frames the bot sees late (human reaction is ~15-20)')
    v.add_argument('--dir', help='directory for the session (default: runs/evals/<time>-versus)')
    v.set_defaults(fn=_guarded(cmd_versus))

    b = sub.add_parser('bench', help='measure emulator throughput at several worker counts')
    b.add_argument('--counts', default='1,4,6,8')
    b.add_argument('--seconds', type=float, default=30)
    b.set_defaults(fn=_guarded(cmd_bench))

    s = sub.add_parser('serve', help='dashboard')
    s.add_argument('--port', type=int, default=8777)
    s.add_argument('--no-browser', action='store_true')
    s.set_defaults(fn=cmd_serve)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    main(sys.argv[1:])
