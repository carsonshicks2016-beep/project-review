"""BLADE arena — pit any two trained fighters against each other (any loadouts).

    python3 -m blade.arena --left knight --right reaper          # sword vs axe -> mp4
    python3 -m blade.arena --left knight --right knight          # mirror match
    python3 -m blade.arena --left knight --right reaper --live   # clean 3D window

Fighter A is driven by --left (using its own armament), B by --right; the model
is built with each fighter's trained loadout, so cross-weapon duels just work.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import mujoco

from .env import BladeEnv
from .config import default_stages
from .render import tracking_camera, aim_camera, save_mp4
from foldspace.networks import ActorCritic


def load_agent(name, device="cpu"):
    path = name if name.endswith(".pt") else f"blade/checkpoints/{name}.pt"
    ck = torch.load(path, map_location=device, weights_only=False)
    net = ActorCritic(ck["obs_dim"], ck["act_dim"]).to(device)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, ck


@torch.no_grad()
def _act(net, obs_row):
    a, _, _ = net.act(torch.as_tensor(obs_row[None]), deterministic=True)
    return a[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--height", type=int, default=620)
    args = ap.parse_args()

    left, lck = load_agent(args.left)
    right, rck = load_agent(args.right)
    la, lb = lck.get("loadout", "sword"), rck.get("loadout", "sword")
    print(f"A: {args.left} [{la}]   vs   B: {args.right} [{lb}]")

    stage = default_stages(la)[-1]          # SPAR world (terrain + combat)
    env = BladeEnv(stage, seed=args.seed, loadout_b=lb)
    obs = env.reset()
    nsteps = int(args.seconds / env.dt)
    policies = lambda o: np.stack([_act(left, o[0]), _act(right, o[1])])

    if args.live:
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame
        pygame.init()
        scr = pygame.display.set_mode((args.width, args.height))
        pygame.display.set_caption(f"BLADE arena · {args.left} vs {args.right}")
        font = pygame.font.SysFont("menlo,monospace", 15)
        r = mujoco.Renderer(env.model, args.height, args.width)
        cam = tracking_camera()
        clock = pygame.time.Clock()
        run = True
        while run:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q)):
                    run = False
            obs, _, done, info = env.step(policies(obs))
            aim_camera(cam, env); r.update_scene(env.data, camera=cam)
            scr.blit(pygame.surfarray.make_surface(r.render().swapaxes(0, 1)), (0, 0))
            for i, (col, hp) in enumerate([((60, 224, 255), info["hp"][0]), ((255, 84, 132), info["hp"][1])]):
                pygame.draw.rect(scr, (40, 44, 60), (args.width - 220, 14 + i * 18, 200, 12))
                pygame.draw.rect(scr, col, (args.width - 220, 14 + i * 18, int(200 * max(hp, 0) / 100), 12))
            scr.blit(font.render(f"{args.left}[{la}]  vs  {args.right}[{lb}]", True, (228, 234, 255)), (16, 14))
            pygame.display.flip()
            if done:
                obs = env.reset()
            clock.tick(int(1 / env.dt))
        pygame.quit()
        return

    cam = tracking_camera()
    r = mujoco.Renderer(env.model, args.height, args.width)
    frames, result = [], None
    for _ in range(nsteps):
        obs, _, done, info = env.step(policies(obs))
        aim_camera(cam, env); r.update_scene(env.data, camera=cam)
        frames.append(r.render())
        if done:
            result = info
            break
    out = args.out or f"blade/replays/{args.left}_vs_{args.right}.mp4"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_mp4(frames, out, fps=int(1 / env.dt))
    print(f"wrote {len(frames)} frames -> {out}")
    if result:
        print("winner:", ("A" if result["winner"] == 0 else "B" if result["winner"] == 1 else "draw"),
              "  hp:", tuple(round(h) for h in result["hp"]))


if __name__ == "__main__":
    main()
