"""Phillip (Slippi-AI) as an opponent: runs inside Slippi-AI's own Python environment.

Started by a puffbot worker as a child process (vendor/slippi-ai/.venv/bin/python
this_file.py ...). It must not import puffbot's own modules or torch; it only speaks a
tiny protocol on stdin/stdout:

    worker -> server   4-byte big-endian length + pickle of ('step', GameState) | ('quit',)
    server -> worker   4-byte big-endian length + pickle of
                       ('ready', info) | ('action', controller dict) | ('error', text)

The agent keeps its own recurrent state and reaction delay (medium-v2: 21 frames) and
resets itself on each game's first frame (frame -123), so the worker must send every
frame of every game, countdown included.

Loading medium-v2 needs two shims, both for training-only code it imports but never runs
at play time: an empty `wandb` module and a placeholder for run_lib.OpponentType.
"""
from __future__ import annotations

import argparse
import os
import pickle
import struct
import sys
import time
import types


def _send(out, msg):
    data = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
    out.write(struct.pack('>I', len(data)))
    out.write(data)
    out.flush()


def _recv(inp):
    head = inp.read(4)
    if len(head) < 4:
        return None
    (n,) = struct.unpack('>I', head)
    return pickle.loads(inp.read(n))


class Recorder:
    """Stands in for a melee.Controller: records what the agent presses."""

    def __init__(self, port: int):
        self.port = port
        self.reset()

    def reset(self):
        self.buttons: dict[str, bool] = {}
        self.main = (0.5, 0.5)
        self.c = (0.5, 0.5)
        self.shoulder = 0.0

    def press_button(self, b):
        self.buttons[b.name] = True

    def release_button(self, b):
        self.buttons[b.name] = False

    def tilt_analog(self, which, x, y):
        if which.name == 'BUTTON_MAIN':
            self.main = (float(x), float(y))
        else:
            self.c = (float(x), float(y))

    def press_shoulder(self, which, amount):
        self.shoulder = float(amount)

    def state(self) -> dict:
        return {'buttons': [b for b, on in self.buttons.items() if on], 'main': self.main,
                'c': self.c, 'shoulder': self.shoulder}


def load_state(path: str):
    sys.modules.setdefault('wandb', types.ModuleType('wandb'))
    from slippi_ai import saving

    class _Placeholder:
        def __new__(cls, *args):
            return args[0] if args else None

    class _Unpickler(saving.CustomUnpickler):
        def find_class(self, module, name):
            if module == 'slippi_ai.rl.run_lib' and name == 'OpponentType':
                return _Placeholder
            return super().find_class(module, name)

    with open(path, 'rb') as f:
        return _Unpickler(f).load()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--port', type=int, default=2)
    ap.add_argument('--opponent-port', type=int, default=1)
    a = ap.parse_args()
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr            # library prints must never corrupt the protocol
    try:
        t0 = time.time()
        sys.modules.setdefault('wandb', types.ModuleType('wandb'))
        from slippi_ai import eval_lib
        pad = Recorder(a.port)
        state = load_state(a.model)
        agent = eval_lib.build_agent(controller=pad, port=a.port, opponent_port=a.opponent_port,
                                     state=state, console_delay=0)
        delay = (state.get('config') or {}).get('policy', {}).get('delay')
        _send(out, ('ready', {'load_seconds': round(time.time() - t0, 1), 'delay': delay, 'pid': os.getpid()}))
    except Exception as exc:
        _send(out, ('error', f'load failed: {type(exc).__name__}: {exc}'))
        return
    while True:
        msg = _recv(inp)
        if msg is None or msg[0] == 'quit':
            return
        try:
            agent.step(msg[1])
            _send(out, ('action', pad.state()))
        except Exception as exc:
            _send(out, ('error', f'step failed: {type(exc).__name__}: {exc}'))


if __name__ == '__main__':
    main()
