"""Client for a Phillip opponent (Slippi-AI medium-v2), from inside a puffbot worker.

Slippi-AI needs TensorFlow, which lives in its own environment (vendor/slippi-ai/.venv),
so the model runs in a child process (phillip_server.py) and this client trades one
message per frame with it. Measured on this Mac: about 5.7 ms per frame on one CPU core,
so a worker playing Phillip runs near 110-120 fps instead of ~200.

Every wait has a deadline. A Phillip that stops answering raises PhillipError, which is an
EmulatorError, so the worker's existing recovery applies: the game is cut (its data
discarded, not taught as an ending) and the setup is relaunched with a fresh Phillip.
"""
from __future__ import annotations

import os
import pickle
import select
import struct
import subprocess
import time
from pathlib import Path

import melee

from .dolphin import EmulatorError

SERVER = Path(__file__).with_name('phillip_server.py')
LOAD_SECONDS = 180.0
STEP_SECONDS = 10.0
# Characters medium-v2 was trained on, minus the two our menus cannot select for a
# human-controlled port (Sheik needs a held-A transform, Ice Climbers are two bodies).
SUPPORTED = ('FOX', 'FALCO', 'MARTH', 'JIGGLYPUFF', 'CPTFALCON', 'PEACH', 'YOSHI', 'LUIGI',
             'PIKACHU', 'SAMUS')


class PhillipError(EmulatorError):
    pass


class Phillip:
    def __init__(self, python: str, model: str, log_path: Path, port: int = 2, opponent_port: int = 1,
                 server: Path = SERVER):
        env = dict(os.environ, TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1',
                   OMP_NUM_THREADS='1', TF_CPP_MIN_LOG_LEVEL='2', PYTHONDONTWRITEBYTECODE='1')
        log = open(log_path, 'ab')
        self.proc = subprocess.Popen(
            [python, '-u', str(server), '--model', model, '--port', str(port),
             '--opponent-port', str(opponent_port)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env, start_new_session=True)
        log.close()
        self.frames = 0
        self.step_seconds = 0.0
        msg = self._recv(LOAD_SECONDS)
        if msg[0] != 'ready':
            self.close()
            raise PhillipError(f'Phillip did not start: {msg[1]}')
        self.info = msg[1]

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def _read(self, n: int, deadline: float) -> bytes:
        fd = self.proc.stdout.fileno()
        buf = b''
        while len(buf) < n:
            left = deadline - time.monotonic()
            if left <= 0:
                raise PhillipError('Phillip stopped answering.')
            ready, _, _ = select.select([fd], [], [], left)
            if ready:
                chunk = os.read(fd, n - len(buf))
                if not chunk:
                    raise PhillipError(f'Phillip exited (code {self.proc.poll()}).')
                buf += chunk
        return buf

    def _recv(self, seconds: float):
        deadline = time.monotonic() + seconds
        (n,) = struct.unpack('>I', self._read(4, deadline))
        return pickle.loads(self._read(n, deadline))

    def _send(self, msg):
        data = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            self.proc.stdin.write(struct.pack('>I', len(data)) + data)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise PhillipError(f'Phillip exited (code {self.proc.poll()}).') from exc

    def step(self, gamestate) -> dict:
        """Controller state Phillip chooses for this frame (send every frame)."""
        t = time.perf_counter()
        self._send(('step', gamestate))
        msg = self._recv(STEP_SECONDS)
        if msg[0] != 'action':
            raise PhillipError(str(msg[1]))
        self.frames += 1
        self.step_seconds += time.perf_counter() - t
        return msg[1]

    def close(self):
        try:
            if self.alive:
                self._send(('quit',))
                self.proc.wait(timeout=2)
        except Exception:
            pass
        if self.alive:
            try:
                os.killpg(self.proc.pid, 9)
            except OSError:
                self.proc.kill()
        self.proc.wait(timeout=5)


def write(pad, action: dict):
    """Apply Phillip's controller state to our pad for this frame."""
    pad.release_all()
    for name in action.get('buttons', ()):
        pad.press_button(melee.Button[name])
    pad.tilt_analog(melee.Button.BUTTON_MAIN, *action.get('main', (0.5, 0.5)))
    pad.tilt_analog(melee.Button.BUTTON_C, *action.get('c', (0.5, 0.5)))
    if action.get('shoulder', 0.0) > 0:
        pad.press_shoulder(melee.Button.BUTTON_L, float(action['shoulder']))
