"""Curriculum self-play PPO trainer for BLADE — laboratory-grade, loadout-aware.

Trains one named fighter through the curriculum (STAND → WALK → TERRAIN → SPAR),
auto-promoting between stages when a greedy evaluation shows mastery.  Pick the
armament with --loadout.

    python3 -m blade.train --name knight --loadout sword_shield
    python3 -m blade.train --name reaper --loadout axe --iters 30000
    python3 -m blade.train --name knight --resume          # continue, same stage

Checkpoints (with the active stage) land in blade/checkpoints/<name>.pt.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from .env import VecBlade
from .config import default_stages
from .curriculum import Curriculum, evaluate
from .dashboard import Dashboard
from .weapons import LOADOUT_NAMES
from foldspace.networks import ActorCritic
from duel.ppo import PPOConfig, compute_gae, ppo_update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--loadout", default="sword", choices=LOADOUT_NAMES)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--envs", type=int, default=24)
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--ent", type=float, default=0.003)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval-every", type=int, default=15)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    out = f"blade/checkpoints/{args.name}.pt"
    os.makedirs("blade/checkpoints", exist_ok=True)

    start_stage = 0
    saved_state = None
    if args.resume and os.path.exists(out):
        ck = torch.load(out, map_location=device, weights_only=False)
        start_stage = ck.get("stage_idx", 0)
        saved_state = ck["state_dict"]
        args.loadout = ck.get("loadout", args.loadout)

    stages = default_stages(args.loadout)
    curric = Curriculum(stages, start=start_stage)
    N, T = args.envs, args.rollout
    S = 2 * N

    env = VecBlade(curric.stage, N, seed=args.seed, workers=args.workers)
    eval_env = VecBlade(curric.stage, 6, seed=args.seed + 99, workers=args.workers)
    D, A = env.obs_dim, env.act_dim
    agent = ActorCritic(D, A).to(device)
    if saved_state is not None:
        agent.load_state_dict(saved_state)
    opt = torch.optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
    cfg = PPOConfig(ent_coef=args.ent, minibatch=2048)

    b_obs = np.zeros((T, S, D), np.float32); b_act = np.zeros((T, S, A), np.float32)
    b_logp = np.zeros((T, S), np.float32); b_val = np.zeros((T, S), np.float32)
    b_rew = np.zeros((T, S), np.float32); b_done = np.zeros((T, S), np.float32)

    dash = Dashboard(args.name, args.loadout, str(device), args.iters * T * S, stages)
    if start_stage:
        dash.banner(f"resumed at stage {curric.stage.name} ({start_stage+1}/{len(stages)})")

    obs = env.reset()
    last_ev = {"survive": 0, "reach": 0, "hits": 0, "ret": 0}
    gstep = 0
    t0 = time.time()

    def save():
        torch.save({"state_dict": agent.state_dict(), "obs_dim": D, "act_dim": A,
                    "loadout": args.loadout, "stage_idx": curric.i, "iter": it}, out)

    it = 0
    try:
        for it in range(1, args.iters + 1):
            curric.tick()
            for t in range(T):
                with torch.no_grad():
                    act, logp, val = agent.act(torch.as_tensor(obs.reshape(S, D), device=device))
                a = act.cpu().numpy()
                b_obs[t] = obs.reshape(S, D); b_act[t] = a
                b_logp[t] = logp.cpu().numpy(); b_val[t] = val.cpu().numpy()
                obs, rew, done, _ = env.step(a.reshape(N, 2, A))
                b_rew[t] = rew.reshape(S); b_done[t] = np.repeat(done, 2)
                gstep += S
            with torch.no_grad():
                last_val = agent.act(torch.as_tensor(obs.reshape(S, D), device=device))[2].cpu().numpy()
            adv, ret = compute_gae(b_rew, b_val, b_done, last_val, cfg.gamma, cfg.gae_lambda)
            flat = lambda x: x.reshape(T * S, *x.shape[2:])
            stats = ppo_update(agent, opt, {"obs": flat(b_obs), "act": flat(b_act), "logp": flat(b_logp),
                                            "adv": flat(adv), "ret": flat(ret), "val": flat(b_val)}, cfg, device)

            if it % args.eval_every == 0 or it == 1:
                last_ev = evaluate(agent, eval_env, device)
                sps = int(gstep / (time.time() - t0))
                metrics = {**last_ev, "ret": float(b_rew.mean()),
                           "ent": stats["ent"], "kl": stats["kl"]}
                dash.row(it, curric.i, curric.stage.name, gstep, sps, metrics,
                         curric.promote_progress(last_ev))
                save()
                if curric.should_promote(last_ev):
                    frm = curric.stage.name
                    val = curric.metric_value(last_ev)
                    curric.advance()
                    dash.promote(frm, curric.stage.name, it, frm and curric.stages[curric.i-1].promote_metric, val)
                    env = VecBlade(curric.stage, N, seed=args.seed + it, workers=args.workers)
                    eval_env = VecBlade(curric.stage, 6, seed=args.seed + 99, workers=args.workers)
                    obs = env.reset()
                    save()
    except KeyboardInterrupt:
        print()
        dash.banner("interrupted — saving", )
    save()
    dash.done(out, curric.stage.name)


if __name__ == "__main__":
    main()
