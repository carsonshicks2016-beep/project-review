"""
Live PPO training view with interactive save / load.

Training runs in a background thread while the main thread shows the policy
driving live (at the current curriculum difficulty), a lap-progress chart, and
live stats. Keys:

    S      save a checkpoint  -> native file dialog to name + place it
    L      load a checkpoint  -> native file dialog to pick a run to resume
    Space  pause / resume training
    Esc    quit (autosaves to the run checkpoint)

Save/load happen at an iteration boundary inside the training thread (the file
path is chosen on the main thread via the dialog), so the network is never
mutated mid-update. This lets you branch runs, swap in different amounts of
training, and resume any of them.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import torch

from .config import PPOSpec
from .ppo import PPO
from .ppo_env import SupraEnv


class _Trainer(threading.Thread):
    def __init__(self, ppo: PPO, total_iters: int, checkpoint="ppo_race.pt",
                 log_every: int = 5, save_every: int = 20):
        super().__init__(daemon=True)
        self.ppo = ppo
        self.total = total_iters
        self.checkpoint = checkpoint
        self.log_every = max(1, log_every)
        self.save_every = max(1, save_every)
        self.stop = False
        self.paused = False
        self._lock = threading.Lock()
        self._save_req = None
        self._load_req = None
        self.message = ""
        self.iter = 0

    def request_save(self, path):
        with self._lock:
            self._save_req = path

    def request_load(self, path):
        with self._lock:
            self._load_req = path

    def run(self):
        ppo = self.ppo
        best_path = PPO.best_path_for(self.checkpoint)
        ppo._best_path = best_path           # so _graduate can preserve the peak
        ppo.init_best_metric(best_path)     # don't clobber an existing peak
        obs = ppo.vec.reset()
        t0 = time.time()
        while not self.stop and self.iter < self.total:
            with self._lock:
                sreq, self._save_req = self._save_req, None
                lreq, self._load_req = self._load_req, None
            if sreq:
                try:
                    ppo.save(sreq)
                    self.message = f"saved -> {sreq.split('/')[-1]}"
                except BaseException as e:
                    self.message = f"save failed: {e}"
            if lreq:
                try:
                    ppo.load_state(lreq)
                    obs = ppo.vec.reset()
                    self.message = f"loaded {lreq.split('/')[-1]} (diff {ppo.difficulty:.2f})"
                except BaseException as e:
                    self.message = f"load failed: {e}"
            if self.paused:
                time.sleep(0.05)
                continue

            ppo.set_schedule(self.iter, self.total)
            obs, batch = ppo.collect(obs)
            adv, ret = ppo.gae(batch[3], batch[4], batch[5], obs)
            st = ppo.update(batch, adv, ret)
            ppo.updates += 1
            promoted = ppo.maybe_promote()
            self.iter += 1
            ret_m = float(np.mean(ppo.ep_returns)) if ppo.ep_returns else 0.0
            lap_m = float(np.mean(ppo.ep_laps)) if ppo.ep_laps else 0.0
            ppo.history.append((ppo.updates, ret_m, lap_m, ppo.difficulty))
            # Print the SAME log line as headless training so the dashboard can
            # parse + chart a live (--live) run too — watching it in the window
            # no longer means flying blind in the browser.
            if self.iter % self.log_every == 0:
                if ppo.mode == "drift":
                    dr = float(np.mean(ppo.ep_drift)) if ppo.ep_drift else 0.0
                    metric = f"drift {dr:4.2f}"
                else:
                    metric = f"laps {lap_m:4.2f}"
                steps = self.iter * ppo.cfg.rollout * ppo.cfg.n_envs
                print(f"it {self.iter:4d}  ret {ret_m:7.1f}  {metric}  "
                      f"diff {ppo.difficulty:.2f}  pi {st['pi']:+.3f}  "
                      f"vf {st['vf']:.2f}  ent {st['ent']:.2f}  "
                      f"{steps / max(1e-6, time.time() - t0):.0f} sps"
                      + ("  [PROMOTED]" if promoted else ""), flush=True)
            # deterministic eval + keep-best, same as headless — so a run you
            # WATCH still banks its peak (_best.pt) and reports the honest metric.
            # (No auto early-stop here: you control when to stop a live run.)
            if self.iter % self.save_every == 0:
                try:
                    ppo.eval_and_save(self.checkpoint, best_path)
                except BaseException as e:
                    self.message = f"eval failed: {e}"


def run(car="supra", iterations=5000, checkpoint="ppo_race.pt", mode="race",
        resume=None, fixed_track=None, track_name=None,
        target_track=None, target_name=None, opponents=None, multiagent=False,
        ppo_cfg: PPOSpec | None = None, reward=None,
        env_cls_override=None, env_kwargs: dict | None = None,
        eval_callback=None, extra_metadata=None):
    try:
        import pygame
    except ImportError:
        raise SystemExit("pygame is required.")
    from .carart import draw_car
    from .app import Camera, draw_road
    from .background import TrackEnvironmentBackground

    def _dialog(save: bool, prompt: str, default_name: str = None):
        """Native macOS file picker via osascript (a SEPARATE process — tkinter
        crashes hard alongside an SDL window on macOS). Returns a path or None."""
        import subprocess
        if save:
            inner = f'choose file name with prompt "{prompt}"'
            if default_name:
                inner += f' default name "{default_name}"'
        else:
            inner = f'choose file with prompt "{prompt}"'
        try:
            r = subprocess.run(["osascript", "-e", f"POSIX path of ({inner})"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                return None                      # user cancelled
            path = r.stdout.strip()
            if save and path and not path.endswith(".pt"):
                path += ".pt"
            return path or None
        except BaseException as e:
            print(f"file dialog unavailable: {e}")
            return None

    cfg = ppo_cfg or PPOSpec()
    ppo = PPO(mode=mode, car=car, ppo=cfg, reward=reward, fixed_track=fixed_track,
              track_name=track_name, target_track=target_track,
              target_name=target_name, opponent_configs=opponents,
              multiagent=multiagent, env_cls_override=env_cls_override,
              env_kwargs=env_kwargs, eval_callback=eval_callback,
              extra_metadata=extra_metadata)
    if resume:
        try:
            ppo.load_state(resume)
            print(f"resumed {resume} (diff {ppo.difficulty:.2f})")
        except BaseException as e:
            print(f"resume failed: {e}")

    sdim = ppo.sdim

    def act_mean(env_obs):
        if env_obs.ndim == 2:
            acts = []
            for obs in env_obs:
                disp_sdim = disp.sensor_dim
                s = obs[:disp_sdim]
                if len(s) < sdim:
                    s = np.concatenate([s, np.zeros(sdim - len(s), dtype=np.float32)])
                s = ppo.norm.normalize(s)
                nobs = np.concatenate([s, obs[disp_sdim:]]).astype(np.float32)
                with torch.no_grad():
                    a = ppo.net.act_mean(torch.as_tensor(nobs).unsqueeze(0)).squeeze(0).numpy()
                acts.append(np.clip(a, -1, 1))
            return np.array(acts)
        else:
            disp_sdim = disp.sensor_dim
            s = env_obs[:disp_sdim]
            if len(s) < sdim:
                s = np.concatenate([s, np.zeros(sdim - len(s), dtype=np.float32)])

            s = ppo.norm.normalize(s)
            nobs = np.concatenate([s, env_obs[disp_sdim:]]).astype(np.float32)
            with torch.no_grad():
                a = ppo.net.act_mean(torch.as_tensor(nobs).unsqueeze(0)).squeeze(0).numpy()
            return np.clip(a, -1, 1)

    pygame.init()
    W, H = 1320, 800
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"Supra Drift — PPO {mode} training [{car}]")
    clock = pygame.time.Clock()
    f = pygame.font.SysFont("menlo,consolas,monospace", 15)
    fb = pygame.font.SysFont("menlo,consolas,monospace", 24, bold=True)

    trainer = _Trainer(ppo, iterations, checkpoint=checkpoint)
    trainer.start()

    PANEL = 300
    vw = W - PANEL
    # the live display car is cosmetic (real training happens in ppo.vec). Show
    # the SAME surface that's being trained: a specialist's fixed track, or the
    # generalist curriculum pool. Start at the line (no exploring starts) so the
    # view is stable and matches what you'd see with --watch.
    import copy
    disp_cfg = copy.copy(ppo.cfg)
    disp_cfg.random_start = False

    if multiagent:
        disp = ppo.env_cls(n_agents=4, car=car, ppo=disp_cfg,
                           reward=reward, fixed_track=fixed_track)
        def get_vehs(): return disp.vehs
    else:
        disp_kw = dict(car=car, ppo=disp_cfg, reward=reward,
                       fixed_track=fixed_track)
        disp_kw.update(env_kwargs or {})
        if opponents:
            disp = ppo.env_cls(opponent_configs=opponents, **disp_kw)
        else:
            disp = ppo.env_cls(mode=mode, **disp_kw)
        def get_vehs():
            v = [disp.veh]
            if hasattr(disp, "opponents"):
                v.extend([o.veh for o in disp.opponents])
            return v
    disp.difficulty = ppo.difficulty
    dobs = disp.reset()
    trail = []
    show_trail = False
    msg_until = 0.0

    # Immersive follow-cam (matches the drive/watch views) so the full trackside
    # environment + props render fast via the ±30-segment cull. The generalist
    # picks a fresh curriculum track each episode, so rebuild the environment
    # whenever the displayed track changes.
    cam = Camera(vw, H, scale=12.0, center=(vw / 2, H / 2))
    main_veh = get_vehs()[0]
    cam.cx, cam.cy = main_veh.x, main_veh.y
    bg_layer = TrackEnvironmentBackground(disp.trk, seed=7)
    _bg_trk = disp.trk

    def on_reset():
        nonlocal trail, bg_layer, _bg_trk
        trail = []
        if disp.trk is not _bg_trk:
            bg_layer = TrackEnvironmentBackground(disp.trk, seed=7)
            _bg_trk = disp.trk
        m_veh = get_vehs()[0]
        cam.cx, cam.cy = m_veh.x, m_veh.y

    from .sound import SpatialAudioMixer
    audio = SpatialAudioMixer()
    audio.start(get_vehs())

    try:
        running = True
        while running:
            clock.tick(120)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_m:
                        audio.toggle_mute()
                    elif ev.key == pygame.K_t:
                        show_trail = not show_trail
                    elif ev.key == pygame.K_SPACE:
                        trainer.paused = not trainer.paused
                    elif ev.key == pygame.K_s:
                        p = _dialog(True, "Save checkpoint",
                                    default_name=f"{car}_diff{ppo.difficulty:.2f}_u{ppo.updates}.pt")
                        if p:
                            trainer.request_save(p)
                    elif ev.key == pygame.K_l:
                        p = _dialog(False, "Load checkpoint to resume")
                        if p:
                            trainer.request_load(p)
                            on_reset()

            # step the live display car with the current policy
            dact = act_mean(dobs)
            if multiagent:
                disp_brakes = [max(0.0, -float(a[1])) for a in dact]
                dobs, _, term, trunc, _ = disp.step(dact)
                term = term.any() if isinstance(term, np.ndarray) else term
                trunc = trunc.any() if isinstance(trunc, np.ndarray) else trunc
            else:
                disp_brakes = [max(0.0, -float(dact[1]))]
                dobs, _, term, trunc, _ = disp.step(dact)

            main_veh = get_vehs()[0]
            if show_trail:
                trail.append((main_veh.x, main_veh.y))
                if len(trail) > 40:
                    trail = trail[-40:]
            else:
                trail = []
            if term or trunc:
                disp.difficulty = ppo.difficulty
                dobs = disp.reset()
                on_reset()

            # ---- render ----
            trk = disp.trk
            m_veh = get_vehs()[0]
            cam.follow(m_veh.x, m_veh.y, lerp=0.15)

            # Audio update
            lvx = m_veh.speed * np.cos(m_veh.yaw)
            lvy = m_veh.speed * np.sin(m_veh.yaw)
            if multiagent:
                throttles = [max(0.0, float(a[1])) for a in dact]
            else:
                throttles = [max(0.0, float(dact[1]))]
                # If there are frozen opponents, we need their throttles
                if hasattr(disp, "opponents"):
                    for opp in disp.opponents:
                        throttles.append(max(0.0, float(opp._last_act[1])) if hasattr(opp, "_last_act") else 0.0)

            audio.update((cam.cx, cam.cy), (lvx, lvy), m_veh.yaw, get_vehs(), throttles)

            bg_layer.draw_terrain(screen, cam)
            draw_road(pygame, screen, cam, trk, (34, 35, 40), (110, 70, 70), (70, 72, 80))
            if len(trail) > 1:
                pygame.draw.lines(screen, (90, 130, 200), False,
                                  [cam.to_screen(*p) for p in trail], 1)

            bg_layer.draw_objects(screen, cam, m_veh)

            for i, v in enumerate(get_vehs()):
                brk = disp_brakes[i] if i < len(disp_brakes) else 0.0
                draw_car(screen, cam.to_screen, cam.scale, v, disp.spec,
                         brake=brk, headlights=(bg_layer.time_of_day in ("dusk", "night")))

            if trainer.message:
                msg_until = time.time() + 4.0
                ppo_msg = trainer.message
                trainer.message = ""
            if time.time() < msg_until:
                screen.blit(f.render(ppo_msg, True, (235, 220, 120)), (20, H - 28))

            _panel(pygame, screen, f, fb, W, H, PANEL, ppo, trainer)
            pygame.display.flip()

        trainer.stop = True
        audio.stop()
        try:
            ppo.save(checkpoint)
            print(f"autosaved -> {checkpoint}")
        except Exception:
            pass
        trainer.join(timeout=2.0)
        pygame.quit()
    except BaseException as e:
        try:
            audio.stop()
        except:
            pass
        import traceback
        with open("crash_log.txt", "w") as f_crash:
            f_crash.write(traceback.format_exc())
        raise


def _panel(pygame, screen, f, fb, W, H, PANEL, ppo, trainer):
    x0 = W - PANEL
    pygame.draw.rect(screen, (28, 30, 36), (x0, 0, PANEL, H))
    pygame.draw.line(screen, (60, 64, 72), (x0, 0), (x0, H), 2)
    x = x0 + 18
    ret = float(np.mean(ppo.ep_returns)) if ppo.ep_returns else 0.0
    lap = float(np.mean(ppo.ep_laps)) if ppo.ep_laps else 0.0
    state = "PAUSED" if trainer.paused else "training"
    screen.blit(fb.render(f"iter {trainer.iter}", True, (235, 238, 245)), (x, 16))
    screen.blit(f.render(state, True, (150, 200, 150) if not trainer.paused
                         else (235, 190, 90)), (x, 48))
    y = 74
    rows = [
        (f"return    {ret:7.1f}", (200, 205, 212)),
        (f"laps      {lap:5.2f}", (120, 210, 140)),
        (f"difficulty {ppo.difficulty:.2f}", (235, 200, 90)),
        (f"updates   {ppo.updates}", (170, 175, 185)),
    ]
    for t, c in rows:
        screen.blit(f.render(t, True, c), (x, y)); y += 24

    # lap-progress chart
    y += 10
    screen.blit(f.render("laps / update", True, (140, 145, 155)), (x, y)); y += 18
    cw, ch = PANEL - 36, 150
    pygame.draw.rect(screen, (16, 18, 22), (x, y, cw, ch))
    hist = ppo.history
    if len(hist) >= 2:
        laps = np.array([h[2] for h in hist])
        diff = np.array([h[3] for h in hist])
        top = max(laps.max(), 1.0)
        n = len(laps)
        def pts(arr, tp):
            return [(int(x + i / (n - 1) * cw), int(y + ch - (v / tp) * ch))
                    for i, v in enumerate(arr)]
        pygame.draw.lines(screen, (120, 210, 140), False, pts(laps, top), 2)
        pygame.draw.lines(screen, (235, 200, 90), False, pts(diff, 1.0), 1)
        screen.blit(f.render(f"{top:.1f}", True, (110, 112, 120)), (x + cw - 30, y + 2))
    y += ch + 12
    screen.blit(f.render("laps", True, (120, 210, 140)), (x, y))
    screen.blit(f.render("difficulty", True, (235, 200, 90)), (x + 60, y)); y += 30

    for line in ["S      save (file dialog)", "L      load / resume",
                 "Space  pause", "Esc    quit (autosaves)"]:
        screen.blit(f.render(line, True, (135, 140, 150)), (x, y)); y += 20
