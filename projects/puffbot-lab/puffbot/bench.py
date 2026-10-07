"""Emulator throughput at several worker counts: the number that decides `workers`.

Each emulator plays Jigglypuff (random macros) against a level-9 Fox, at unlimited
speed with no video, exactly as a training worker would. Measured on the M2 Pro this
project was built on (mainline Slippi 4.0 beta 19): 1 -> 257 fps, 4 -> 831, 6 -> 1082,
8 -> 1207, 12 -> 1259. Version one's Slippi 3.6.4 was locked to 60 fps per emulator
and topped out near 190 fps of training across the whole machine.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

from .config import RUNS, Config


def _one(i: int, cfg_dict: dict, seconds: float, out_dir: str, q):
    import numpy as np
    from . import actions
    from .dolphin import Emulator, Setup
    cfg = Config(**cfg_dict)
    emu = Emulator(cfg, i, Path(out_dir) / f'w{i:02d}' / 'User', visible=False,
                   log_path=Path(out_dir) / f'w{i:02d}.log')
    rng = np.random.default_rng(i)
    try:
        t0 = time.monotonic()
        emu.launch()
        boot = time.monotonic() - t0
        g = emu.enter_match(Setup(cfg.character, 'FOX', 9))
        menus = time.monotonic() - t0 - boot
        ex = actions.Executor(cfg.act_every)
        start, frames = time.monotonic(), 0
        while time.monotonic() - start < seconds:
            if 1 in g.players:
                if ex.done or frames == 0:
                    mask = actions.legal_mask(g.players[1])
                    ex.start(int(rng.choice(np.flatnonzero(mask))))
                actions.write(emu.pads[0], ex.next_input(g.players[1]))
            emu.pads[1].release_all()
            g = emu.frame()
            frames += 1
        q.put({'worker': i, 'fps': frames / (time.monotonic() - start), 'boot': boot, 'menus': menus})
    except Exception as exc:
        q.put({'worker': i, 'error': f'{type(exc).__name__}: {exc}'})
    finally:
        emu.stop()


def bench(cfg: Config, counts: list[int], seconds: float) -> list[dict]:
    ctx = mp.get_context('spawn')
    rows = []
    out_dir = RUNS / 'bench'
    for n in counts:
        q = ctx.Queue()
        procs = [ctx.Process(target=_one, args=(i, cfg.to_dict(), seconds, str(out_dir), q)) for i in range(n)]
        for p in procs:
            p.start()
            time.sleep(0.3)
        res = [q.get(timeout=seconds + 120) for _ in procs]
        for p in procs:
            p.join(timeout=10)
        ok = [r for r in res if 'fps' in r]
        row = {'workers': n, 'total_fps': round(sum(r['fps'] for r in ok), 1),
               'per_worker_fps': [round(r['fps'], 1) for r in sorted(ok, key=lambda r: r['worker'])],
               'realtime_multiple': round(sum(r['fps'] for r in ok) / 60, 2),
               'errors': [r['error'] for r in res if 'error' in r]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (out_dir / 'bench.json').write_text(json.dumps(rows, indent=2))
    return rows
