"""Loopback-only dashboard service. Training and watching run in separate processes."""
from collections import deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
import json
import multiprocessing as mp
import queue
import re
import threading
import time
import uuid
import webbrowser
from .config import CARS, TRACKS, Config
from .environment import Environment
from .learning import atomic_json, evaluation_worker, load_checkpoint, train_worker, publish

WEB = Path(__file__).parent / 'web'


def watch_worker(commands, output, shutdown):
    import torch
    torch.set_num_threads(1)
    cfg = Config()
    env = Environment(cfg)
    policy = None
    source, paused, rate, generation = 'Reference driver', False, 1., 1
    reset_at = 0.
    episode = 0
    manual = None
    manual_time = 0.
    last_checkpoint, last_mtime, last_scan = None, 0, 0.
    def geometry():
        publish(output, dict(kind='geometry', geometry=env.geometry(), generation=generation))
    geometry()
    while not shutdown.is_set():
        start = time.monotonic()
        try:
            while True:
                msg = commands.get_nowait()
                op = msg['op']
                if op == 'configure':
                    next_cfg = Config.parse(msg['config'])
                    next_policy = None
                    checkpoint = msg.get('checkpoint')
                    if checkpoint:
                        next_policy, payload = load_checkpoint(checkpoint)
                        next_cfg = Config.parse(payload['config'])
                    next_env = Environment(next_cfg)
                    cfg, policy, env = next_cfg, next_policy, next_env
                    source = msg.get('label', 'Saved PPO') if policy else 'Reference driver'
                    manual = None
                    last_checkpoint = checkpoint if msg.get('live') else None
                    last_mtime = Path(checkpoint).stat().st_mtime_ns if checkpoint else 0
                    generation += 1
                    episode += 1
                    reset_at = 0
                    geometry()
                elif op == 'pause':
                    paused = bool(msg['paused'])
                elif op == 'speed':
                    rate = msg['rate']
                elif op == 'reset':
                    env.reset()
                    reset_at = 0
                    episode += 1
                elif op == 'manual':
                    manual = msg['action']
                    manual_time = time.monotonic()
                    source = 'Manual driving'
                    last_checkpoint = None
                elif op == 'geometry':
                    geometry()
        except queue.Empty:
            pass
        except Exception as exc:
            publish(output, dict(kind='error', error=str(exc)))
        try:
            if last_checkpoint and start-last_scan > 2:
                last_scan = start
                stamp = Path(last_checkpoint).stat().st_mtime_ns
                if stamp != last_mtime:
                    policy, payload = load_checkpoint(last_checkpoint)
                    last_mtime = stamp
                    source = f'Live PPO · update {payload["update"]}'
            if not paused:
                if reset_at:
                    if start >= reset_at:
                        env.reset()
                        episode += 1
                        reset_at = 0
                else:
                    if manual is not None:
                        action = manual if start-manual_time < .5 else [0., -.5, 0.]
                    else:
                        action = policy.act(env.observe()) if policy else env.baseline_action()
                    _, _, term, trunc, _ = env.step(action)
                    if term or trunc:
                        reset_at = start + 1.
            publish(output, dict(kind='frame', frame=env.telemetry(), generation=generation,
                                 source=source, paused=paused, rate=rate, episode=episode,
                                 car=cfg.car, track=cfg.track, mode=cfg.mode))
        except Exception as exc:
            publish(output, dict(kind='error', error=str(exc)))
            paused = True
        shutdown.wait(max(.001, (1/30)/rate-(time.monotonic()-start)))


class Laboratory:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ctx = mp.get_context('spawn')
        self.lock = threading.RLock()
        self.commands, self.frames = self.ctx.Queue(64), self.ctx.Queue(128)
        self.shutdown = self.ctx.Event()
        self.watcher = self.ctx.Process(target=watch_worker, args=(self.commands, self.frames, self.shutdown), daemon=True)
        self.watcher.start()
        self.trainer = self.evaluator = None
        self.training = dict(status='idle', update=0, steps=0)
        self.watching, self.geometry = {}, None
        self.events = deque(maxlen=60)
        self.event('Hybrid Lab ready. The reference driver is independent of PPO training.')
        # Interrupted runs remain resumable after a server restart.
        for p in self.root.glob('*/status.json'):
            try:
                data = json.loads(p.read_text())
                if data.get('status') in ('starting', 'training', 'paused', 'evaluating', 'stopping'):
                    atomic_json(p, dict(data, status='interrupted'))
            except (OSError, ValueError):
                continue

    def event(self, message):
        self.events.append(dict(time=time.time(), message=message))

    def refresh(self):
        with self.lock:
            while True:
                try:
                    msg = self.frames.get_nowait()
                except queue.Empty:
                    break
                if msg['kind'] == 'geometry':
                    self.geometry = dict(msg['geometry'], generation=msg['generation'])
                elif msg['kind'] == 'frame':
                    self.watching = msg
                else:
                    self.event('Viewer: '+msg['error'])
            if self.trainer:
                while True:
                    try:
                        msg = self.training_queue.get_nowait()
                    except queue.Empty:
                        break
                    if msg['status'] != self.training['status']:
                        self.event('Training '+msg['status'] + (': '+msg['error'] if msg.get('error') else '.'))
                    self.training.update(msg)
                if not self.trainer.is_alive() and self.training['status'] in ('starting', 'training', 'paused', 'evaluating', 'stopping'):
                    try:
                        self.training.update(json.loads((self.root/self.training['run']/'status.json').read_text()))
                    except (OSError, ValueError):
                        pass
                    if self.training['status'] in ('starting', 'training', 'paused', 'evaluating', 'stopping'):
                        self.training['status'] = 'failed'
                        self.training['error'] = f'Training worker exited ({self.trainer.exitcode}). Resume from the latest saved checkpoint.'
                        atomic_json(self.root/self.training['run']/'status.json', self.training)
            return dict(training=dict(self.training), watching=dict(self.watching), events=list(self.events),
                        viewer_alive=self.watcher.is_alive())

    def run_path(self, run):
        if not isinstance(run, str) or not re.fullmatch(r'[0-9]{8}-[0-9]{6}-[a-f0-9]{8}', run):
            raise ValueError('Invalid run identifier.')
        path = self.root / run
        if not path.is_dir():
            raise ValueError('Run does not exist.')
        return path

    def checkpoint_path(self, run, checkpoint='latest.pt'):
        if checkpoint not in ('latest.pt', 'best.pt'):
            raise ValueError('Choose latest.pt or best.pt.')
        path = self.run_path(run) / checkpoint
        if not path.is_file():
            raise ValueError('That checkpoint has not been saved yet.')
        return path

    def runs(self):
        rows = []
        for p in sorted(self.root.glob('*/config.json'), reverse=True):
            try:
                run = p.parent
                rows.append(dict(id=run.name, config=json.loads(p.read_text()),
                                 status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {},
                                 checkpoints=[n for n in ('latest.pt', 'best.pt') if (run/n).is_file()],
                                 evaluation=json.loads((run/'evaluation.json').read_text()) if (run/'evaluation.json').exists() else None))
            except (OSError, ValueError):
                continue
        return rows

    def start(self, data):
        if self.trainer and self.trainer.is_alive():
            raise ValueError('Stop the current run before starting another.')
        resume = None
        config = data.get('config', {})
        if data.get('resume_run'):
            resume = self.checkpoint_path(data['resume_run'], data.get('checkpoint', 'latest.pt'))
            _, payload = load_checkpoint(resume)
            config = dict(payload['config'], **config)
            config['updates'] = max(config['updates'], payload['update'] + data.get('additional_updates', 100))
            for key in ('mode', 'car', 'track', 'hills'):
                if config[key] != payload['config'][key]:
                    raise ValueError('Resume keeps the saved car, track, and mode.')
        cfg = Config.parse(config)
        run = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
        path = self.root / run
        path.mkdir()
        atomic_json(path/'config.json', cfg.dict())
        if resume:
            atomic_json(path/'lineage.json', dict(parent_run=data['resume_run'], checkpoint=resume.name,
                        note='Policy and optimizer resumed; simulation episodes restart.'))
        self.pause, self.stop, self.save = self.ctx.Event(), self.ctx.Event(), self.ctx.Event()
        self.training_queue = self.ctx.Queue(256)
        self.training = dict(status='starting', update=0, steps=0, run=run)
        atomic_json(path/'status.json', self.training)
        self.trainer = self.ctx.Process(target=train_worker, args=(cfg.dict(), str(path), self.training_queue,
                         self.pause, self.stop, self.save, str(resume) if resume else None), daemon=True)
        self.trainer.start()
        self.event('Started '+cfg.name+'.')
        return dict(run=run)

    def command(self, data):
        with self.lock:
            op = data.get('op')
            self.refresh()
            if op == 'start':
                return self.start(data)
            if op in ('pause', 'resume', 'stop', 'save'):
                if not self.trainer or not self.trainer.is_alive():
                    raise ValueError('There is no active training run.')
                if op == 'pause': self.pause.set()
                elif op == 'resume': self.pause.clear()
                elif op == 'stop': self.stop.set(); self.pause.clear()
                elif op == 'save': self.save.set()
                self.event({'pause':'Pause requested.', 'resume':'Resume requested.', 'stop':'Stopping and saving the current policy.', 'save':'Checkpoint save requested.'}[op])
            elif op == 'watch':
                cfg = Config.parse(data.get('config', {}))
                checkpoint = None
                if data.get('run'):
                    checkpoint = str(self.checkpoint_path(data['run'], data.get('checkpoint', 'latest.pt')))
                self.commands.put_nowait(dict(op='configure', config=cfg.dict(), checkpoint=checkpoint,
                              live=bool(data.get('live')), label='Live PPO' if data.get('live') else 'Saved PPO'))
                self.event('Loading '+('PPO policy.' if checkpoint else 'reference driver.'))
            elif op == 'viewer':
                action = data.get('action')
                if action == 'pause':
                    if type(data.get('paused')) is not bool:
                        raise ValueError('paused must be true or false.')
                    msg = dict(op='pause', paused=data['paused'])
                elif action == 'speed':
                    if data.get('rate') not in (.25, .5, 1, 2):
                        raise ValueError('Choose 0.25, 0.5, 1, or 2× playback speed.')
                    msg = dict(op='speed', rate=data['rate'])
                elif action == 'reset': msg = dict(op='reset')
                elif action == 'manual':
                    import math
                    a = data.get('controls')
                    if not isinstance(a, list) or len(a) != 3 or not all(type(v) in (float, int) and math.isfinite(v) and -1 <= v <= 1 for v in a):
                        raise ValueError('Invalid driving controls.')
                    msg = dict(op='manual', action=a)
                else: raise ValueError('Unknown viewer action.')
                self.commands.put_nowait(msg)
            elif op == 'evaluate':
                if self.evaluator and self.evaluator.is_alive():
                    raise ValueError('An evaluation is already running.')
                checkpoint = self.checkpoint_path(data.get('run'), data.get('checkpoint', 'latest.pt'))
                output = checkpoint.parent/'evaluation.json'
                atomic_json(output, dict(status='evaluating', checkpoint=checkpoint.name))
                self.evaluator = self.ctx.Process(target=evaluation_worker, args=(str(checkpoint), str(output)), daemon=True)
                self.evaluator.start()
                self.event('Evaluation started: three repeatable starts on the saved track.')
            else:
                raise ValueError('Unknown command.')
            return dict(ok=True)

    def close(self):
        self.shutdown.set()
        if self.trainer and self.trainer.is_alive():
            self.stop.set()
            self.pause.clear()
            self.trainer.join(15)
        for proc in (self.trainer, self.watcher, self.evaluator):
            if proc:
                proc.join(2)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(2)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def log_message(self, format, *args):
        pass

    @property
    def lab(self): return self.server.lab

    def json(self, data, status=200):
        body = json.dumps(data, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def allowed_host(self):
        return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')

    def do_GET(self):
        if not self.allowed_host():
            return self.json(dict(error='Local access only.'), 403)
        u = urlsplit(self.path)
        args = parse_qs(u.query)
        try:
            if u.path == '/api/state': return self.json(self.lab.refresh())
            if u.path == '/api/options': return self.json(dict(defaults=Config().dict(), cars=CARS, tracks=TRACKS))
            if u.path == '/api/track':
                self.lab.refresh()
                return self.json(self.lab.geometry)
            if u.path == '/api/runs': return self.json(self.lab.runs())
            if u.path == '/api/metrics':
                p = self.lab.run_path(args.get('run', [''])[0])/'metrics.jsonl'
                rows = []
                if p.exists():
                    with p.open() as stream:
                        for line in deque(stream, maxlen=1000):
                            try: rows.append(json.loads(line))
                            except ValueError: pass
                return self.json(rows)
            if u.path == '/api/download':
                run = self.lab.run_path(args.get('run', [''])[0])
                name = args.get('file', [''])[0]
                if name not in ('latest.pt', 'best.pt', 'metrics.jsonl', 'config.json', 'evaluation.json', 'error.log') or not (run/name).is_file():
                    raise ValueError('File is not available.')
                body = (run/name).read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Disposition', f'attachment; filename="{run.name}-{name}"')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if u.path.startswith('/api/'):
                return self.json(dict(error='Not found.'), 404)
            return super().do_GET()
        except (ValueError, FileNotFoundError) as exc:
            self.json(dict(error=str(exc)), 400)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        origin = self.headers.get('Origin')
        if not self.allowed_host() or (origin and origin != 'http://'+self.headers.get('Host', '')):
            return self.json(dict(error='Local, same-origin access only.'), 403)
        if self.path != '/api/command': return self.json(dict(error='Not found.'), 404)
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            return self.json(dict(error='Expected JSON.'), 415)
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 16000: raise ValueError('Invalid request size.')
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict): raise ValueError('Expected an object.')
            self.json(self.lab.command(data))
        except (ValueError, TypeError, KeyError, queue.Full) as exc:
            self.json(dict(error=str(exc) or 'Command queue is busy.'), 400)
        except Exception as exc:
            self.lab.event('Command failed: '+str(exc))
            self.json(dict(error=str(exc)), 500)


def serve(port=8766, root=None, open_browser=False):
    root = root or Path(__file__).parent/'runs'
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    try:
        server.lab = Laboratory(root)
        url = f'http://127.0.0.1:{server.server_port}'
        print(f'Hybrid Lab running at {url}', flush=True)
        if open_browser: webbrowser.open(url)
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        print('\nSaving and shutting down Hybrid Lab.', flush=True)
    finally:
        if hasattr(server, 'lab'): server.lab.close()
        server.server_close()
