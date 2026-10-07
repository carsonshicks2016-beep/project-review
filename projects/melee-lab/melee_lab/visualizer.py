"""Read-only desktop companion. Never imports the trainer or opens model weights.

Run `python -m melee_lab.visualizer --run-dir runs/<id>` to reopen it. One
companion per workspace follows new training launches. Closing it exits its
reader; it does not affect training. Native build and failures stay off the
training process and are recorded in .runtime/visualizer.log.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .config import ROOT
from . import layout


def read(path):
    try: return json.loads(Path(path).read_text())
    except (OSError, ValueError): return {}


def write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, allow_nan=False))
    os.replace(temporary, path)


def ancestry(directory, root):
    """Follow recorded inputs, using the frozen counter at each resume edge.

    latest.zip is mutable. Its *current* metadata must never stand in for the
    checkpoint that a descendant actually consumed.
    """
    nodes, seen = [], set()
    current, frozen, source = Path(directory), None, None
    for _ in range(64):
        if current in seen: break
        seen.add(current)
        request, status = read(current/'request.json'), read(current/'status.json')
        meta = frozen or {}
        nodes.append(dict(id=current.name, source=source, current=not nodes,
                          steps=status.get('steps') if not nodes else meta.get('steps'),
                          kind='Demonstration learning' if meta.get('imitation_samples') else
                               'Anchored PPO' if request.get('anchor') else 'PPO training',
                          samples=meta.get('imitation_samples'),
                          sha256=meta.get('sha256'), missing=not current.is_dir()))
        identity = request.get('checkpoint_identity') or {}
        source = identity.get('source') or request.get('checkpoint')
        if not source: break
        path = (root/source).resolve()
        if not path.is_relative_to(root) or path.parent == current: break
        # The child owns the immutable input metadata, even if source was deleted.
        frozen = dict(read(current/'input-policy.json'))
        frozen.update({k: identity[k] for k in ('steps', 'sha256') if k in identity})
        current = path.parent
    return list(reversed(nodes))


def snapshot(directory, root=ROOT, now=None):
    directory, root = Path(directory).resolve(), Path(root).resolve()
    now = time.time() if now is None else now
    state, request = read(directory/'status.json'), read(directory/'request.json')
    config = read(directory/'config.json')
    count = max(1, int(state.get('envs') or request.get('envs') or 1))
    status = state.get('status', 'starting')
    pid = state.get('pid')
    if pid and status in ('running', 'starting', 'paused'):
        try: os.kill(int(pid), 0)
        except ProcessLookupError: status = 'disconnected'
        except (PermissionError, ValueError, TypeError): pass
    slots = []
    for i in range(count):
        path = directory/f'env{i}'/'status.json'
        if count == 1 and not path.exists(): path = directory/'status.json'
        data = read(path)
        try: age = max(0, now-path.stat().st_mtime)
        except OSError: age = None
        slots.append(dict(index=i, frame=data.get('frame'), action=data.get('action'),
                          players=data.get('players', {}), connection=data.get('connection'),
                          age=age, stale=age is None or age > 15,
                          failed=data.get('failed'), cpu_level=data.get('cpu_level')))
    history = [h for h in state.get('history', []) if h.get('kind') in ('ppo', 'self-play')][-20:]
    steps = state.get('steps', 0)
    initial = state.get('initial_steps', (request.get('checkpoint_identity') or {}).get('steps', 0))
    boxes = layout.load(count, root/'window-layout.json')
    geometry = layout.visualizer_space(boxes)
    latest_meta = directory/'latest.json'
    try: saved_age = max(0, now-latest_meta.stat().st_mtime)
    except OSError: saved_age = None
    return dict(id=directory.name, status=status, phase=state.get('phase', 'booting'),
                timestamp=now, steps=steps, initial_steps=initial,
                session_steps=max(0, steps-initial), requested_steps=request.get('steps'),
                elapsed=state.get('elapsed'), cpu_level=state.get('cpu_level', config.get('cpu_level')),
                character=config.get('character', 'Agent'), action_set=config.get('action_set', 'legacy'),
                anchor=Path(request['anchor']).stem if request.get('anchor') else None,
                anchor_weight=state.get('anchor_weight'), saved_age=saved_age,
                slots=slots, lineage=ancestry(directory, root), recent=history,
                geometry=geometry, placement='Free desktop space' if geometry else 'Floating · no large free region')


def launch(directory):
    """Best-effort, non-blocking launch. The companion's flock deduplicates it."""
    if sys.platform != 'darwin': return
    directory = Path(directory).resolve()
    root = directory.parent.parent
    if read(root/'visualizer-settings.json').get('enabled', True) is False: return
    try:
        runtime = root/'.runtime'; runtime.mkdir(exist_ok=True)
        write(runtime/'visualizer-request.json', dict(directory=str(directory), nonce=time.time_ns()))
        with (runtime/'visualizer.log').open('a') as log:
            subprocess.Popen([sys.executable, '-m', 'melee_lab.visualizer', '--follow', '--root', str(root)],
                             cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    except OSError:
        pass  # An optional UI must never interrupt a training run.


def follow(root):
    runtime = root/'.runtime'; runtime.mkdir(exist_ok=True)
    with (runtime/'visualizer.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return
        source = Path(__file__).parent/'native/BrainWindow.swift'
        bundle = runtime/'Melee Brain.app'
        executable = bundle/'Contents/MacOS/MeleeBrain'
        executable.parent.mkdir(parents=True, exist_ok=True)
        import plistlib
        with (bundle/'Contents/Info.plist').open('wb') as handle:
            plistlib.dump(dict(CFBundleExecutable='MeleeBrain', CFBundleIdentifier='local.melee-lab.brain',
                              CFBundleName='Melee Brain', CFBundlePackageType='APPL',
                              LSUIElement=True, NSHighResolutionCapable=True), handle)
        if not executable.exists() or source.stat().st_mtime > executable.stat().st_mtime:
            subprocess.run(['xcrun', 'swiftc', '-O', str(source), '-o', str(executable),
                            '-framework', 'Cocoa', '-framework', 'WebKit'], check=True, timeout=120)
        payload = runtime/'visualizer-state.json'
        request = read(runtime/'visualizer-request.json')
        directory = Path(request['directory'])
        write(payload, snapshot(directory, root))
        child = subprocess.Popen([str(executable), str(payload), str(Path(__file__).parent/'web/brain.html')])
        try:
            while child.poll() is None:
                request = read(runtime/'visualizer-request.json')
                target = Path(request.get('directory', directory)).resolve()
                if target.parent == root/'runs' and target.is_dir(): directory = target
                try: write(payload, snapshot(directory, root))
                except (OSError, ValueError, TypeError) as exc: print(f'Snapshot unavailable: {exc}', flush=True)
                time.sleep(1)
        finally:
            if child.poll() is None: child.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--follow', action='store_true')
    args = parser.parse_args()
    if args.follow: follow(args.root.resolve())
    elif args.run_dir: launch(args.run_dir)
    else: parser.error('Choose --run-dir runs/<id>')
