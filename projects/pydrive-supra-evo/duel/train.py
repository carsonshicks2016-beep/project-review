"""Train a Duel policy with self-play PPO.

Each duel produces up to two agent-streams (one per fighter).  Most arenas pit
the current policy against itself (both streams trainable); a configurable
fraction pit it against a frozen snapshot from the opponent pool (only the
current side is trainable there).

Example:
    python3 -m duel.train --iters 400 --envs 256 --rollout 128 --out duel/checkpoints/duel.pt
"""
from __future__ import annotations

import argparse
import os
import time
from collections import deque

import numpy as np
import torch

from .env import VecDuel, DT
from .networks import ActorCritic, ACT_DIM
from .ppo import PPOConfig, compute_gae, ppo_update
from .selfplay import OpponentPool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.004)
    ap.add_argument("--pool-prob", type=float, default=0.3)
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=20, help="iters of pure mirror play before using the pool")
    ap.add_argument("--out", default="duel/checkpoints/duel.pt")
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    env = VecDuel(args.envs, seed=args.seed)
    D = env.obs_dim
    N = args.envs
    S = 2 * N
    T = args.rollout

    agent = ActorCritic(D).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    cfg = PPOConfig(lr=args.lr, ent_coef=args.ent)
    pool = OpponentPool(D, device=device)

    # rollout buffers, laid out (T, S)
    b_obs = np.zeros((T, S, D), np.float32)
    b_act = np.zeros((T, S, ACT_DIM), np.float32)
    b_logp = np.zeros((T, S), np.float32)
    b_val = np.zeros((T, S), np.float32)
    b_rew = np.zeros((T, S), np.float32)
    b_done = np.zeros((T, S), np.float32)
    b_mask = np.zeros((T, S), np.float32)

    obs = env.reset()                       # (N,2,D)
    rng = np.random.default_rng(args.seed)
    recent_len = deque(maxlen=400)
    recent_decisive = deque(maxlen=400)
    ep_len = np.zeros(N, np.int64)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    global_steps = 0
    t_start = time.time()
    for it in range(1, args.iters + 1):
        use_pool = (it > args.warmup) and len(pool) > 0 and args.pool_prob > 0
        if use_pool:
            pool_mask = rng.random(N) < args.pool_prob
            opp = pool.sample()
            p1_cols = np.where(pool_mask)[0] * 2 + 1
        else:
            pool_mask = np.zeros(N, bool)
            p1_cols = np.empty(0, np.int64)

        it_t0 = time.time()
        for t in range(T):
            os_streams = obs.reshape(S, D)
            with torch.no_grad():
                ot = torch.as_tensor(os_streams, device=device)
                action, logp, value = agent.act(ot)
            action = action.cpu().numpy()
            logp = logp.cpu().numpy()
            value = value.cpu().numpy()

            mask = np.ones(S, np.float32)
            if len(p1_cols):
                with torch.no_grad():
                    oo = torch.as_tensor(os_streams[p1_cols], device=device)
                    oa, _, _ = opp.act(oo)
                action[p1_cols] = oa.cpu().numpy()
                mask[p1_cols] = 0.0

            b_obs[t] = os_streams
            b_act[t] = action
            b_logp[t] = logp
            b_val[t] = value
            b_mask[t] = mask

            obs, rew, done, info = env.step(action.reshape(N, 2, ACT_DIM))
            b_rew[t] = rew.reshape(S)
            b_done[t] = np.repeat(done, 2).astype(np.float32)

            ep_len += 1
            if done.any():
                fin = np.where(done)[0]
                for e in fin:
                    recent_len.append(ep_len[e] * DT)
                    recent_decisive.append(1.0 if info["winner"][e] >= 0 else 0.0)
                    ep_len[e] = 0
            global_steps += S

        # bootstrap value of the post-rollout observation
        with torch.no_grad():
            last_val = agent.act(torch.as_tensor(obs.reshape(S, D), device=device))[2]
        last_val = last_val.cpu().numpy()

        adv, ret = compute_gae(b_rew, b_val, b_done, last_val, cfg.gamma, cfg.gae_lambda)

        flat = lambda x: x.reshape(T * S, *x.shape[2:])
        m = flat(b_mask).astype(bool)
        batch = {
            "obs": flat(b_obs)[m], "act": flat(b_act)[m], "logp": flat(b_logp)[m],
            "adv": flat(adv)[m], "ret": flat(ret)[m], "val": flat(b_val)[m],
        }
        stats = ppo_update(agent, opt, batch, cfg, device)

        if it % args.snapshot_every == 0:
            pool.add(agent)
        if it % args.save_every == 0 or it == args.iters:
            torch.save({"state_dict": agent.state_dict(), "obs_dim": D,
                        "meta": VecDuel.metadata(), "iter": it}, args.out)

        if it % 5 == 0 or it == 1:
            sps = int(global_steps / (time.time() - t_start))
            mlen = np.mean(recent_len) if recent_len else 0.0
            dec = np.mean(recent_decisive) if recent_decisive else 0.0
            itps = time.time() - it_t0
            print(f"it {it:4d} | steps {global_steps/1e6:5.2f}M | sps {sps:>7d} "
                  f"| {itps:4.1f}s/it | match {mlen:4.1f}s decisive {dec*100:4.0f}% "
                  f"| ret/step {b_rew.mean():+.4f} | ent {stats['ent']:5.2f} "
                  f"| vf {stats['vf']:.3f} | kl {stats['kl']:+.4f} "
                  f"| pool {len(pool)}", flush=True)

    print(f"done in {time.time()-t_start:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
