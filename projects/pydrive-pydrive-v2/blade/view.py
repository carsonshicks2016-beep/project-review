"""Simplified live 3D viewer for BLADE.

A clean pygame window — no MuJoCo debug panels — showing the fighter(s) with a
minimal HUD (loadout, curriculum stage, survival / health). It renders MuJoCo
offscreen and blits to the window, so plain `python3` works (no mjpython), and it
hot-reloads the checkpoint, following the curriculum into new stages as they're
reached.

    python3 -m blade.view --name knight
    python3 -m blade.view --name reaper --stage SPAR
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import mujoco
import pygame

from .env import BladeEnv
from .config import default_stages, STAGE_NAMES
from .render import tracking_camera, aim_camera
from foldspace.networks import ActorCritic

BG = (8, 10, 18)
INK = (228, 234, 255)
DIM = (120, 130, 170)
CYAN = (60, 224, 255)
ROSE = (255, 84, 132)
PANEL = (14, 18, 34)


def load_ckpt(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = ActorCritic(ck["obs_dim"], ck["act_dim"]).to(device)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, ck


class Viewer:
    def __init__(self, args):
        self.args = args
        self.path = f"blade/checkpoints/{args.name}.pt"
        self.device = "cpu"
        self.W, self.H = args.width, args.height
        pygame.init()
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption(f"BLADE · {args.name}")
        self.font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 15)
        self.big = pygame.font.SysFont("menlo,dejavusansmono,monospace", 21, bold=True)
        self.agent = None
        self.ck = None
        self.mtime = 0
        self.stage_idx = -1
        self.loadout = args.loadout
        self.flash = 0.0
        self.env = None
        self.renderer = None
        self.cam = tracking_camera()

    def reload(self, force=False):
        if not os.path.exists(self.path):
            return
        mt = os.path.getmtime(self.path)
        if mt == self.mtime and not force:
            return
        try:
            self.agent, self.ck = load_ckpt(self.path, self.device)
            self.mtime = mt
            self.loadout = self.args.loadout or self.ck.get("loadout", "sword")
            want_stage = self.ck.get("stage_idx", 3)
            if self.args.stage:
                want_stage = STAGE_NAMES.index(self.args.stage)
            if want_stage != self.stage_idx:
                self.stage_idx = want_stage
                self._build_world()
            self.flash = 1.4
        except Exception:
            pass

    def _build_world(self):
        stage = default_stages(self.loadout)[self.stage_idx]
        self.env = BladeEnv(stage, seed=int(time.time()) % 9999)
        self.obs = self.env.reset()
        self.renderer = mujoco.Renderer(self.env.model, self.H, self.W)
        self.survival = 0.0

    def run(self):
        # shot mode: render one frame headlessly for verification
        if self.args.shot:
            self.reload(force=True)
            for _ in range(self.args.frames):
                self._step()
            self._render()
            pygame.image.save(self.screen, self.args.shot)
            print("saved", self.args.shot)
            return
        print("waiting for checkpoint…")
        while self.agent is None:
            self.reload(force=True); time.sleep(0.4)
        clock = pygame.time.Clock()
        last_check = 0.0
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_q)):
                    running = False
            self._step()
            self._render()
            pygame.display.flip()
            if time.time() - last_check > 1.0:
                self.reload(); last_check = time.time()
            self.flash = max(0.0, self.flash - 1 / max(self.args.fps, 1))
            clock.tick(self.args.fps)
        pygame.quit()

    def _step(self):
        with torch.no_grad():
            act, _, _ = self.agent.act(torch.as_tensor(self.obs.reshape(2, -1)), deterministic=True)
        self.obs, _, done, info = self.env.step(act.numpy())
        self.hp = info["hp"]
        self.survival = 0.0 if done else self.survival + self.env.dt

    def _render(self):
        aim_camera(self.cam, self.env)
        self.renderer.update_scene(self.env.data, camera=self.cam)
        frame = self.renderer.render()                       # (H, W, 3)
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        self.screen.blit(surf, (0, 0))
        self._hud()

    def _hud(self):
        s = self.env.stage
        # top panel
        bar = pygame.Surface((self.W, 52), pygame.SRCALPHA); bar.fill((8, 10, 18, 170))
        self.screen.blit(bar, (0, 0))
        self._txt(self.big, "BLADE", 16, 8, INK)
        self._txt(self.font, self.loadout.replace("_", "+").upper(), 110, 13, CYAN)
        # stage chip
        chip = f"STAGE {self.stage_idx+1}/4  {s.name}"
        self._txt(self.font, chip, self.W // 2 - 70, 8, INK)
        self._txt(self.font, ("LIVE  iter " + str(self.ck.get("iter", 0))), self.W // 2 - 70, 28, DIM)
        # right: survival or HP
        if s.combat:
            self._hpbar(self.W - 230, 12, self.hp[0], CYAN, "A")
            self._hpbar(self.W - 230, 30, self.hp[1], ROSE, "B")
        else:
            self._txt(self.font, f"upright  {self.survival:4.1f}s", self.W - 180, 18, INK)
        if self.flash > 0:
            self._txt(self.font, "◆ RELOADED", self.W // 2 + 110, 18,
                      (int(60 + 195 * self.flash / 1.4), 240, 255))
        self._txt(self.font, f"hot-reloading {self.args.name}  ·  esc to quit",
                  16, self.H - 24, DIM)

    def _hpbar(self, x, y, hp, col, lbl):
        self._txt(self.font, lbl, x - 16, y - 2, col)
        pygame.draw.rect(self.screen, (40, 44, 60), (x, y, 200, 12))
        pygame.draw.rect(self.screen, col, (x, y, int(200 * max(hp, 0) / 100), 12))

    def _txt(self, font, s, x, y, col):
        self.screen.blit(font.render(s, True, col), (x, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--loadout", default=None, help="override; else read from checkpoint")
    ap.add_argument("--stage", default=None, choices=STAGE_NAMES, help="which stage's world to show")
    ap.add_argument("--fps", type=int, default=40)
    ap.add_argument("--width", type=int, default=1100)
    ap.add_argument("--height", type=int, default=680)
    ap.add_argument("--shot", default=None)
    ap.add_argument("--frames", type=int, default=50)
    args = ap.parse_args()
    Viewer(args).run()


if __name__ == "__main__":
    main()
