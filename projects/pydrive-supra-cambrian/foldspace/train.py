"""Train a Foldspace policy with PPO (single-agent, reusing duel.ppo).

    python3 -m foldspace.train --iters 400 --envs 128 --rollout 96 --out foldspace/checkpoints/fold.pt
"""
from __future__ import annotations

import argparse
import os
import time
from collections import deque

import numpy as np
import torch

from .env import VecFold, SHAPE_NAMES, N_SHAPE, T_MAX
from .networks import ActorCritic
from duel.ppo import PPOConfig, compute_gae, ppo_update


@torch.no_grad()
def deterministic_eval(agent, eval_env, device):
    """Fold every shape once with the greedy policy; return mean + per-shape score."""
    obs = eval_env.set_targets(np.arange(N_SHAPE))
    info = None
    for _ in range(T_MAX):
        act, _, _ = agent.act(torch.as_tensor(obs, device=device), deterministic=True)
        obs, _, _, info = eval_env.step(act.cpu().numpy(), auto_reset=False)
    return float(info["score"].mean()), info["score"].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--rollout", type=int, default=96)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.0025)
    ap.add_argument("--out", default="foldspace/checkpoints/fold.pt")
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    env = VecFold(args.envs, seed=args.seed)
    D, A, N, T = env.obs_dim, env.act_dim, args.envs, args.rollout
    agent = ActorCritic(D, A).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    cfg = PPOConfig(lr=args.lr, ent_coef=args.ent, minibatch=4096)

    b_obs = np.zeros((T, N, D), np.float32)
    b_act = np.zeros((T, N, A), np.float32)
    b_logp = np.zeros((T, N), np.float32)
    b_val = np.zeros((T, N), np.float32)
    b_rew = np.zeros((T, N), np.float32)
    b_done = np.zeros((T, N), np.float32)

    obs = env.reset()
    recent = deque(maxlen=800)                 # terminal scores
    per_shape = {n: deque(maxlen=120) for n in SHAPE_NAMES}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    eval_env = VecFold(N_SHAPE, seed=999)
    best_eval = -1.0
    gstep = 0
    t0 = time.time()

    for it in range(1, args.iters + 1):
        # anneal the entropy bonus: explore early, commit late
        cfg.ent_coef = args.ent * (0.1 + 0.9 * (1 - it / args.iters))
        for t in range(T):
            tid = env.tid.copy()
            with torch.no_grad():
                act, logp, val = agent.act(torch.as_tensor(obs, device=device))
            a = act.cpu().numpy()
            b_obs[t] = obs; b_act[t] = a; b_logp[t] = logp.cpu().numpy(); b_val[t] = val.cpu().numpy()
            obs, rew, done, info = env.step(a)
            b_rew[t] = rew; b_done[t] = done.astype(np.float32)
            for e in np.where(done)[0]:
                sc = float(info["score"][e])
                recent.append(sc)
                per_shape[SHAPE_NAMES[tid[e]]].append(sc)
            gstep += N

        with torch.no_grad():
            last_val = agent.act(torch.as_tensor(obs, device=device))[2].cpu().numpy()
        adv, ret = compute_gae(b_rew, b_val, b_done, last_val, cfg.gamma, cfg.gae_lambda)
        flat = lambda x: x.reshape(T * N, *x.shape[2:])
        batch = {"obs": flat(b_obs), "act": flat(b_act), "logp": flat(b_logp),
                 "adv": flat(adv), "ret": flat(ret), "val": flat(b_val)}
        stats = ppo_update(agent, opt, batch, cfg, device)

        # save the BEST policy by greedy eval, not the last (shapes peak at
        # different times; the final policy can drift on hard shapes like square)
        eval_tag = ""
        if it % args.save_every == 0 or it == args.iters:
            emean, _ = deterministic_eval(agent, eval_env, device)
            if emean > best_eval:
                best_eval = emean
                torch.save({"state_dict": agent.state_dict(), "obs_dim": D, "act_dim": A,
                            "meta": VecFold.metadata(), "iter": it, "eval": emean}, args.out)
                eval_tag = f" | eval {emean*100:4.1f}% *BEST*"
            else:
                eval_tag = f" | eval {emean*100:4.1f}% (best {best_eval*100:4.1f}%)"

        if it % 5 == 0 or it == 1 or eval_tag:
            sps = int(gstep / (time.time() - t0))
            sc = np.mean(recent) if recent else 0.0
            print(f"it {it:4d} | steps {gstep/1e6:5.2f}M | sps {sps:>6d} "
                  f"| score {sc*100:5.1f}% | ent {stats['ent']:6.2f} "
                  f"| vf {stats['vf']:.3f} | kl {stats['kl']:+.4f}{eval_tag}", flush=True)

    # final per-shape report
    print("per-shape final score:")
    for n in SHAPE_NAMES:
        v = np.mean(per_shape[n]) if per_shape[n] else 0.0
        print(f"   {n:9s} {v*100:5.1f}%")
    print(f"done in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
