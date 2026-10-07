"""One Dolphin emulator: configure, launch, stream frames, navigate menus, tear down.

Rules learned the hard way in version one, all enforced here:

* Every wait has a deadline. libmelee's stream reader blocks forever on a silent
  Dolphin unless it is polled, and a blocked worker used to wedge the whole farm.
* Config files are written fresh on every launch. Dolphin rewrites its ini with its own
  key casing and configparser then refuses the duplicates.
* Nothing outlives its owner. Each Dolphin's pid is recorded; a new launch on the same
  slot kills whatever the last one left behind, because an orphan squatting on a Slippi
  port looks exactly like a Dolphin that never booted.
* A setup that will not settle is relaunched, never nursed. A fresh Dolphin reaches a
  match in about four seconds on the mainline build and only this worker waits.
"""
from __future__ import annotations

import configparser
import errno
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import melee
from melee.slippstream import SlippstreamClient

from .config import Config

POLL_SECONDS = 0.02
BOOT_DEADLINE = 45.0          # launch -> stream connected and pipes open
FRAME_DEADLINE = 20.0         # between any two frames once running
MENU_DEADLINE = 45.0          # launch -> first in-game frame
CSS_NUDGE_SECONDS = 8.0       # character select not settling -> drop the coin and retry

def gecko_codes() -> str:
    """libmelee's Extract Menu Info plus Instant Match with random stages disabled.

    Both blocks are read from the installed libmelee so the menu reader always matches
    the library parsing it. Instant Match replays the same setup the moment a game ends,
    so a worker spends almost no time in menus. Its first data word is the random-stage
    flag; clearing it keeps every game on the chosen stage."""
    text = (Path(melee.__file__).parent / 'GALE01r2.ini').read_text()

    def block(title):
        start = text.index(title)
        end = text.find('\n\n', start)
        return text[start:end if end >= 0 else len(text)].strip().splitlines()

    menu = [l for l in block('$Optional: Extract Menu Info [') if not l.startswith('#')]
    instant = [l for l in block('$Optional: Instant Match [') if not l.startswith('*')]
    code = [l.split('#')[0].strip() for l in instant[1:]]
    if code[2].split()[0] != '00000001':
        raise RuntimeError('Unexpected Instant Match code layout in libmelee; refusing to patch it.')
    code[2] = '00000000 ' + code[2].split()[1]
    return '\n'.join(['[Gecko_Enabled]', '$Optional: Extract Menu Info', '$Optional: Instant Match',
                      '[Gecko]', *menu, instant[0], *code, ''])


# ------------------------------------------------------------------ a human player
# Face buttons by *position* (Dolphin's SDL names: Button S/E/W/N = bottom/right/left/top).
# 'switch' puts GameCube A on the right, B on the bottom, X on top and Y on the left, the
# way a Switch Pro controller plays Melee; 'xbox' matches the letters printed on the pad.
FACE_LAYOUTS = {
    'switch': {'Buttons/A': 'Button E', 'Buttons/B': 'Button S', 'Buttons/X': 'Button N', 'Buttons/Y': 'Button W'},
    'xbox': {'Buttons/A': 'Button S', 'Buttons/B': 'Button E', 'Buttons/X': 'Button W', 'Buttons/Y': 'Button N'},
}
HUMAN_COMMON = {
    'Buttons/Z': ('Shoulder R', 'Shoulder L'),            # either bumper grabs
    'Buttons/Start': ('Start',),
    # Dolphin's SDL axes already point up at Y+ (confirmed by the user on 2026-09-26: Y- was
    # upside down).
    'Main Stick/Up': ('Left Y+',), 'Main Stick/Down': ('Left Y-',),
    'Main Stick/Left': ('Left X-',), 'Main Stick/Right': ('Left X+',),
    'C-Stick/Up': ('Right Y+',), 'C-Stick/Down': ('Right Y-',),
    'C-Stick/Left': ('Right X-',), 'C-Stick/Right': ('Right X+',),
    'Triggers/L': ('Trigger L',), 'Triggers/R': ('Trigger R',),              # full press = the click
    'Triggers/L-Analog': ('Trigger L',), 'Triggers/R-Analog': ('Trigger R',),
    'D-Pad/Up': ('Pad N',), 'D-Pad/Down': ('Pad S',), 'D-Pad/Left': ('Pad W',), 'D-Pad/Right': ('Pad E',),
}


def human_bindings(device: str, layout: str) -> dict[str, str]:
    """Port-1 bindings that listen to both the harness's pipe (it drives the menus, then
    lets go) and the player's controller: Dolphin's `|` takes the larger of the two, and
    an idle pipe is 0 on every input."""
    if layout not in FACE_LAYOUTS:
        raise ValueError(f'layout must be one of {sorted(FACE_LAYOUTS)}')
    pipe = {'Buttons/A': 'Button A', 'Buttons/B': 'Button B', 'Buttons/X': 'Button X', 'Buttons/Y': 'Button Y',
            'Buttons/Z': 'Button Z', 'Buttons/Start': 'Button START',
            'Main Stick/Up': 'Axis MAIN Y +', 'Main Stick/Down': 'Axis MAIN Y -',
            'Main Stick/Left': 'Axis MAIN X -', 'Main Stick/Right': 'Axis MAIN X +',
            'C-Stick/Up': 'Axis C Y +', 'C-Stick/Down': 'Axis C Y -', 'C-Stick/Left': 'Axis C X -',
            'C-Stick/Right': 'Axis C X +', 'Triggers/L': 'Button L', 'Triggers/R': 'Button R',
            'Triggers/L-Analog': 'Axis L +', 'Triggers/R-Analog': 'Axis R +',
            'D-Pad/Up': 'Button D_UP', 'D-Pad/Down': 'Button D_DOWN', 'D-Pad/Left': 'Button D_LEFT',
            'D-Pad/Right': 'Button D_RIGHT'}
    pad = {**{k: (v,) for k, v in FACE_LAYOUTS[layout].items()}, **HUMAN_COMMON}
    return {key: ' | '.join([f'`{pipe[key]}`'] + [f'`{device}:{name}`' for name in pad[key]]) for key in pipe}


def detect_controllers() -> list[str]:
    """Game controllers as Dolphin will name them ("SDL/0/Xbox One S Controller"), asked of
    the SDL library bundled inside Dolphin itself. That library is Intel-only, so the
    question runs in an Intel (Rosetta) Python; names can differ between SDL versions."""
    lib = Path(Config().dolphin).parent.parent / 'Frameworks' / 'libSDL2-2.0.0.dylib'
    code = ('import ctypes,time;s=ctypes.CDLL(%r);s.SDL_SetHint(b"SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS",b"1");'
            's.SDL_Init(0x2200);time.sleep(0.6);s.SDL_PumpEvents();'
            's.SDL_GameControllerNameForIndex.restype=ctypes.c_char_p;'
            '[print(s.SDL_GameControllerNameForIndex(i).decode()) for i in range(s.SDL_NumJoysticks()) if s.SDL_IsGameController(i)]') % str(lib)
    try:
        out = subprocess.run(['arch', '-x86_64', '/usr/bin/python3', '-c', code], capture_output=True,
                             text=True, timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [f'SDL/{i}/{name}' for i, name in enumerate(l for l in out.splitlines() if l.strip())]


class EmulatorError(RuntimeError):
    """Anything that means this Dolphin should be torn down and relaunched."""


@dataclass(frozen=True)
class Setup:
    """Who is in the match. Port 1 is always the learner's character."""
    p1: str
    p2: str
    cpu_level: int  # 0 = port 2 is driven by a policy through its pipe

    @property
    def is_cpu(self) -> bool:
        return self.cpu_level > 0


def kill_pid(pid: int):
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _is_dolphin(pid: int) -> bool:
    out = subprocess.run(['ps', '-o', 'command=', '-p', str(pid)], capture_output=True, text=True).stdout
    return 'Dolphin' in out


class Emulator:
    def __init__(self, config: Config, index: int, home: Path, visible: bool = False,
                 replay_dir: Path | None = None, log_path: Path | None = None,
                 human: dict | None = None):
        self.cfg = config
        self.human = human          # {'device': 'SDL/0/...', 'layout': 'switch'}: a person on port 1
        # Sound for a person: someone playing, or a watch window (never training's windows).
        self.audio = bool(human) or (visible and config.viewer_audio)
        self.index = index
        self.home = Path(home)
        self.port = config.base_port + index
        self.visible = visible
        self.replay_dir = replay_dir
        self.log_path = Path(log_path) if log_path else self.home.parent / 'dolphin.log'
        self.pidfile = self.home.parent / 'dolphin.pid'
        self.console: melee.Console | None = None
        self.pads: list[melee.Controller] = []
        self.proc: subprocess.Popen | None = None
        self.setup: Setup | None = None
        self.launches = 0
        self.last_frame_time = time.monotonic()

    # ------------------------------------------------------------------ lifecycle
    def _reap_previous(self):
        if self.pidfile.exists():
            try:
                pid = int(self.pidfile.read_text().strip())
                if _is_dolphin(pid):
                    kill_pid(pid)
            except (ValueError, OSError):
                pass
            self.pidfile.unlink(missing_ok=True)

    def _write_configs(self):
        for sub in ('Config', 'GameSettings', 'Pipes'):
            shutil.rmtree(self.home / sub, ignore_errors=True)
        (self.home / 'Config').mkdir(parents=True, exist_ok=True)
        ini = configparser.ConfigParser()
        sections = {
            'Analytics': {'enabled': 'False', 'permissionasked': 'True'},
            'Interface': {'confirmstop': 'False', 'showtoolbar': 'False', 'showstatusbar': 'False',
                          'usepanicdialogs': 'False', 'onscreendisplaymessages': 'False'},
            'Display': {'fullscreen': 'False', 'renderwindowwidth': '1280' if self.human else '400',
                        'renderwindowheight': '1056' if self.human else '330',
                        'renderwindowxpos': str(80 if self.human else 40 + 410 * (self.index % 4)),
                        'renderwindowypos': str(40 if self.human else 60 + 360 * (self.index // 4)),
                        'rendertomain': 'False'},
            # A person's controller must keep working when the browser has focus.
            'Input': {'backgroundinput': 'True'},
            'Core': {'cputhread': 'True', 'enablecheats': 'True'},
            'General': {'showlag': 'False'},
            'AutoUpdate': {'updatetrack': ''},
        }
        for name, values in sections.items():
            ini.add_section(name)
            for k, v in values.items():
                ini.set(name, k, v)
        if self.audio:
            ini.add_section('DSP')
            ini.set('DSP', 'backend', 'Cubeb')      # the Mac's default output device
            ini.set('DSP', 'volume', '100')
            ini.set('DSP', 'muted', 'False')
        with open(self.home / 'Config/Dolphin.ini', 'w') as f:
            ini.write(f)
        gfx = configparser.ConfigParser()
        gfx.add_section('Settings')
        gfx.set('Settings', 'internalresolution', '1')
        gfx.set('Settings', 'msaa', '1')
        gfx.add_section('Hardware')
        gfx.set('Hardware', 'vsync', 'False')
        with open(self.home / 'Config/GFX.ini', 'w') as f:
            gfx.write(f)

    def _bind_human(self):
        path = self.home / 'Config' / 'GCPadNew.ini'
        ini = configparser.ConfigParser()   # lower-cases keys, exactly as libmelee wrote them;
        ini.read(path)                      # Dolphin reads keys case-insensitively
        for key, expr in human_bindings(self.human['device'], self.human.get('layout', 'switch')).items():
            ini.set('GCPad1', key, expr)
        ini.set('GCPad1', 'Main Stick/Dead Zone', '8.')     # Xbox sticks drift a little at rest
        ini.set('GCPad1', 'C-Stick/Dead Zone', '15.')
        with open(path, 'w') as f:
            ini.write(f)

    def launch(self):
        self.stop()
        self._reap_previous()
        self.home.mkdir(parents=True, exist_ok=True)
        self._write_configs()
        gfx = 'OGL' if self.visible else 'Null'
        speed = self.cfg.visible_speed if self.visible else 0.0
        console = melee.Console(
            path=self.cfg.dolphin, dolphin_home_path=str(self.home), tmp_home_directory=False,
            blocking_input=True, polling_mode=True, polling_timeout=POLL_SECONDS,
            slippi_port=self.port, gfx_backend=gfx, disable_audio=not self.audio, emulation_speed=speed,
            save_replays=bool(self.replay_dir), replay_dir=str(self.replay_dir) if self.replay_dir else None,
            replay_monthly_folders=False, setup_gecko_codes=False, fullscreen=False, log_level=2)
        pads = [melee.Controller(console, port=p) for p in (1, 2)]
        if self.human:
            self._bind_human()
        games = self.home / 'GameSettings'
        games.mkdir(parents=True, exist_ok=True)
        (games / 'GALE01r2.ini').write_text(gecko_codes())

        cmd = [console.exe_path, '-e', self.cfg.iso, '-u', str(self.home)]
        if not self.visible:
            cmd.append('-b')
        log = open(self.log_path, 'ab')
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                                     start_new_session=True)
        log.close()
        console._process = self.proc
        self.pidfile.write_text(str(self.proc.pid))
        self.console, self.pads = console, pads
        self.launches += 1
        self.setup = None
        try:
            self._connect()
        except Exception:
            self.stop()
            raise

    def _alive(self):
        if self.proc is None or self.proc.poll() is not None:
            code = None if self.proc is None else self.proc.returncode
            raise EmulatorError(f'Dolphin exited (code {code}). See {self.log_path.name}.')

    def _connect(self):
        deadline = time.monotonic() + BOOT_DEADLINE
        console = self.console
        while True:
            self._alive()
            try:
                if console.connect():
                    break
            except (EOFError, OSError):
                pass  # the client's worker died; replaced below
            if time.monotonic() > deadline:
                raise EmulatorError(f'Slippi stream on port {self.port} did not connect within {BOOT_DEADLINE:.0f}s.')
            # A failed connect leaves the client's worker process spent; build a fresh one.
            console._slippstream = SlippstreamClient(console.slippi_address, console.slippi_port)
            time.sleep(0.3)
        for pad in self.pads:
            while True:
                self._alive()
                try:
                    fd = os.open(pad.pipe_path, os.O_WRONLY | os.O_NONBLOCK)
                    os.set_blocking(fd, True)
                    pad.pipe = os.fdopen(fd, 'w')
                    console.controllers.append(pad)
                    break
                except OSError as exc:
                    if exc.errno != errno.ENXIO:
                        raise
                    if time.monotonic() > deadline:
                        raise EmulatorError('Dolphin never opened its controller pipes.')
                    time.sleep(0.02)
        self.last_frame_time = time.monotonic()

    def stop(self):
        console, self.console = self.console, None
        for pad in self.pads:
            try:
                pad.disconnect()
            except Exception:
                pass
        self.pads = []
        if self.proc is not None:
            kill_pid(self.proc.pid)
            try:
                self.proc.wait(timeout=3)
            except Exception:
                pass
            self.proc = None
        if console is not None:
            try:
                stream = console._slippstream
                if getattr(stream, '_worker', None) is not None and stream._worker.is_alive():
                    stream._shutdown.set()
                    stream._worker.join(timeout=2)
                    if stream._worker.is_alive():
                        stream._worker.kill()
                        stream._worker.join(timeout=1)
            except Exception:
                pass
        self.pidfile.unlink(missing_ok=True)
        self.setup = None

    # --------------------------------------------------------------------- frames
    def frame(self) -> melee.GameState:
        """The next gamestate. Raises EmulatorError instead of ever blocking forever."""
        deadline = time.monotonic() + FRAME_DEADLINE
        console = self.console
        if console is None:
            raise EmulatorError('Emulator is not running.')
        while True:
            try:
                state = console.step()
            except (BrokenPipeError, EOFError, ConnectionError) as exc:
                raise EmulatorError(f'Stream broke: {exc!r}') from exc
            except Exception as exc:  # libmelee raises bare Exceptions on malformed events
                if type(exc).__name__ == 'EnetDisconnected':
                    raise EmulatorError('Slippi stream disconnected.') from exc
                raise
            if state is not None:
                self.last_frame_time = time.monotonic()
                return state
            self._alive()
            if time.monotonic() > deadline:
                raise EmulatorError(f'No frames from Dolphin for {FRAME_DEADLINE:.0f}s.')

    # ---------------------------------------------------------------------- menus
    def enter_match(self, setup: Setup, max_seconds: float = MENU_DEADLINE) -> melee.GameState:
        """Navigate from wherever the game is to the first playable frame of `setup`.

        Returns the first in-game frame with both players present. Characters and stage
        are verified; the CPU level is reported, not assumed, so a slider that did not
        take is recorded against the level actually played rather than mislabelled."""
        p1 = melee.Character[setup.p1]
        p2 = melee.Character[setup.p2]
        stage = melee.Stage[self.cfg.stage]
        helpers = [melee.MenuHelper(), melee.MenuHelper()]
        deadline = time.monotonic() + max_seconds
        stuck_since = None
        nudge_frames = 0
        tick = 0
        while True:
            g = self.frame()
            tick += 1
            if time.monotonic() > deadline:
                raise EmulatorError(f'Menus did not reach a match within {max_seconds:.0f}s '
                                    f'(stuck at {g.menu_state.name}).')
            if g.menu_state in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
                if 1 in g.players and 2 in g.players:
                    for pad in self.pads:
                        pad.release_all()
                    got = (g.players[1].character, g.players[2].character)
                    if got != (p1, p2):
                        raise EmulatorError(f'Wrong characters at match start: {got}.')
                    if g.stage != stage:
                        raise EmulatorError(f'Wrong stage at match start: {g.stage}.')
                    self.setup = setup
                    return g
                continue
            if nudge_frames > 0:
                # Drop the coin and pull the cursor down toward the controller-type box.
                # Pressing B alone re-enters the same wedged cursor position (v1 lesson).
                nudge_frames -= 1
                for pad in self.pads:
                    pad.release_all()
                    pad.press_button(melee.Button.BUTTON_B)
                    pad.tilt_analog(melee.Button.BUTTON_MAIN, .5, 0.)
                continue
            if g.menu_state == melee.Menu.CHARACTER_SELECT and 2 in g.players:
                q = g.players[2]
                settled = q.coin_down and q.character == p2 and int(q.cpu_level) == setup.cpu_level
                if settled:
                    stuck_since = None
                elif stuck_since is None:
                    stuck_since = time.monotonic()
                elif time.monotonic() - stuck_since > CSS_NUDGE_SECONDS:
                    stuck_since = None
                    nudge_frames = 20
                    helpers = [melee.MenuHelper(), melee.MenuHelper()]
                    continue
                p2_ready = settled or (q.coin_down and q.character == p2 and not setup.is_cpu)
            else:
                p2_ready = True
            for i, pad in enumerate(self.pads):
                helpers[i].menu_helper_simple(
                    g, pad, p1 if i == 0 else p2, stage,
                    cpu_level=0 if i == 0 else setup.cpu_level, costume=i,
                    autostart=(i == 0 and p2_ready))

    def quit_match(self, max_seconds: float = 10.0):
        """Leave an in-progress match for the character select screen (pause, L+R+A+Start)."""
        deadline = time.monotonic() + max_seconds
        tick = 0
        while True:
            g = self.frame()
            tick += 1
            if g.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
                for pad in self.pads:
                    pad.release_all()
                return g
            if time.monotonic() > deadline:
                raise EmulatorError('Could not leave the match.')
            pad = self.pads[0]
            pad.release_all()
            if tick % 2:
                for b in ('BUTTON_L', 'BUTTON_R', 'BUTTON_A', 'BUTTON_START'):
                    pad.press_button(melee.Button[b])
