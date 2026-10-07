"""PPO-Clip with tanh actions, GAE, truncation bootstrap, and atomic checkpoints."""
import json
import math
import os
from pathlib import Path
import queue
import time
import traceback
from collections import deque
import numpy as np
import torch
from torch import nn
from .config import Config
from .environment import Environment

SCHEMA = 'hybrid-lab-v1'


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(60, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
        self.actor = nn.Linear(128, 3)
        self.critic = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.tensor([-.7, -.7, -1.]))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, math.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, .01)
        nn.init.orthogonal_(self.critic.weight, 1)
        with torch.no_grad():
            self.actor.bias.copy_(torch.tensor([0., .6, -1.5]))

    def distribution(self, obs):
        h = self.trunk(obs)
        return torch.distributions.Normal(self.actor(h), self.log_std.clamp(-3, .3).exp()), self.critic(h).squeeze(-1)

    def sample(self, obs):
        dist, value = self.distribution(obs)
        raw = dist.sample()
        return raw, dist.log_prob(raw).sum(-1), value

    def evaluate(self, obs, raw):
        dist, value = self.distribution(obs)
        # PPO ratio uses latent actions; tanh Jacobians cancel in old/new ratio.
        return dist.log_prob(raw).sum(-1), dist.entropy().sum(-1), value

    @torch.no_grad()
    def act(self, obs):
        dist, _ = self.distribution(torch.as_tensor(obs, dtype=torch.float32))
        return dist.mean.tanh().numpy()


def advantages(rewards, values, next_values, terminated, done, gamma, lam):
    """Truncations bootstrap final observations, but never carry GAE into resets."""
    adv = np.zeros_like(rewards)
    carry = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1-terminated[t]) - values[t]
        carry = delta + gamma * lam * (1-done[t]) * carry
        adv[t] = carry
    return adv, adv + values


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    os.replace(temp, path)


def save_checkpoint(path, policy, optimizer, cfg, update, steps, best, **extra):
    path = Path(path)
    payload = dict(schema=SCHEMA, policy=policy.state_dict(), optimizer=optimizer.state_dict(),
                   config=cfg.dict(), update=update, steps=steps, best=best,
                   torch_rng=torch.get_rng_state(), **extra)
    if not all(torch.isfinite(v).all() for v in payload['policy'].values()):
        raise FloatingPointError('Refusing to save non-finite policy.')
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    os.replace(temp, path)


def load_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('schema') != SCHEMA:
        raise ValueError('This checkpoint is not a Hybrid Lab v1 checkpoint. Legacy policies use a different action contract.')
    Config.parse(payload['config'])
    p = Policy()
    p.load_state_dict(payload['policy'], strict=True)
    if not all(torch.isfinite(v).all() for v in p.state_dict().values()):
        raise ValueError('Checkpoint contains non-finite parameters.')
    return p, payload


def evaluate_policy(policy, cfg, starts=3, seconds=20, cancel=None):
    env = Environment(cfg, cfg.seed + 100000)
    results = []
    for k in range(starts):
        obs = env.reset(index=k*len(env.track.center)//starts, speed=10)
        info = {}
        for _ in range(int(seconds*30)):
            if cancel is not None and cancel.is_set():
                return None
            obs, _, term, trunc, info = env.step(policy.act(obs))
            if term or trunc:
                break
        results.append(info)
    return dict(score=float(np.mean([i['return_'] for i in results])),
                progress=float(np.mean([i['progress'] for i in results])),
                drift=float(np.mean([i['drift'] for i in results])),
                offtrack=float(np.mean([i['offtrack'] for i in results])), starts=results,
                seconds=seconds, track=cfg.track, car=cfg.car, time=time.time())


def publish(out, message):
    try:
        out.put_nowait(message)
    except queue.Full:
        pass


def train_worker(config, run_dir, out, pause, stop, save, resume=None):
    torch.set_num_threads(1)
    cfg = Config.parse(config)
    path = Path(run_dir)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate, eps=1e-5)
    update = steps = 0
    best = -1e30
    status = 'starting'
    episodes = deque(maxlen=100)
    started = time.monotonic()
    def checkpoint(filename='latest.pt'):
        save_checkpoint(path/filename, policy, optimizer, cfg, update, steps, best,
                        shuffle_rng=rng.bit_generator.state)
    def report(**kw):
        data = dict(status=status, update=update, steps=steps, run=path.name, **kw)
        atomic_json(path/'status.json', data)
        publish(out, data)
    try:
        if resume:
            policy, payload = load_checkpoint(resume)
            optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate, eps=1e-5)
            optimizer.load_state_dict(payload['optimizer'])
            for group in optimizer.param_groups:
                group['lr'] = cfg.learning_rate
            update, steps, best = payload['update'], payload['steps'], payload['best']
            torch.set_rng_state(payload['torch_rng'])
            if 'shuffle_rng' in payload:
                rng.bit_generator.state = payload['shuffle_rng']
        envs = [Environment(cfg, cfg.seed+i) for i in range(cfg.environments)]
        obs = np.stack([e.reset(random_start=True) for e in envs])
        checkpoint()
        status = 'training'
        report()
        while update < cfg.updates and not stop.is_set():
            rows = []
            roll_started = time.monotonic()
            for t in range(cfg.rollout):
                while pause.is_set() and not stop.is_set():
                    if status != 'paused':
                        status = 'paused'
                        checkpoint()
                        report()
                    if save.is_set():
                        checkpoint()
                        save.clear()
                        report(saved_at=time.time())
                    stop.wait(.1)
                if stop.is_set():
                    break
                if status == 'paused':
                    status = 'training'
                    report()
                if save.is_set():
                    checkpoint()
                    save.clear()
                    report(saved_at=time.time())
                with torch.no_grad():
                    raw, logp, val = policy.sample(torch.from_numpy(obs))
                next_obs, rewards, terms, dones = [], [], [], []
                reset_flags = []
                for i, env in enumerate(envs):
                    env.style_scale = min(1., update/max(1, cfg.style_warmup)) if cfg.style_warmup else 1.
                    no, reward, term, trunc, info = env.step(raw[i].tanh().numpy())
                    next_obs.append(no)
                    rewards.append(reward)
                    terms.append(term)
                    dones.append(term or trunc)
                    reset_flags.append(term or trunc)
                    if term or trunc:
                        episodes.append(info)
                next_obs = np.stack(next_obs)
                with torch.no_grad():
                    _, nv = policy.distribution(torch.from_numpy(next_obs))
                rows.append((obs.copy(), raw.numpy(), logp.numpy(), val.numpy(), np.array(rewards),
                             nv.numpy(), np.array(terms), np.array(dones)))
                obs = next_obs
                for i, done in enumerate(reset_flags):
                    if done:
                        obs[i] = envs[i].reset(random_start=True)
                steps += cfg.environments
                if t % 16 == 0:
                    publish(out, dict(status='training', update=update, steps=steps,
                                     rollout_fraction=(t+1)/cfg.rollout, run=path.name))
            if stop.is_set():
                break
            ob, raw, oldlog, values, rew, nextval, terms, dones = [np.stack(x).astype(np.float32) for x in zip(*rows)]
            adv, ret = advantages(rew, values, nextval, terms, dones, cfg.gamma, cfg.gae_lambda)
            obs_t = torch.from_numpy(ob.reshape(-1, 60))
            raw_t = torch.from_numpy(raw.reshape(-1, 3))
            log_t = torch.from_numpy(oldlog.flatten())
            adv_t = torch.from_numpy(adv.flatten())
            adv_t = (adv_t-adv_t.mean()) / (adv_t.std(unbiased=False)+1e-8)
            ret_t = torch.from_numpy(ret.flatten())
            stats = []
            halt = False
            for epoch in range(cfg.epochs):
                order = rng.permutation(len(obs_t))
                for begin in range(0, len(order), cfg.batch_size):
                    idx = order[begin:begin+cfg.batch_size]
                    logp, entropy, value = policy.evaluate(obs_t[idx], raw_t[idx])
                    log_ratio = logp-log_t[idx]
                    ratio = log_ratio.exp()
                    kl = ((ratio-1)-log_ratio).mean()
                    if kl.item() > cfg.target_kl:
                        halt = True
                        break
                    loss_policy = -torch.minimum(ratio*adv_t[idx], ratio.clamp(1-cfg.clip, 1+cfg.clip)*adv_t[idx]).mean()
                    loss_value = .5*(value-ret_t[idx]).square().mean()
                    loss = loss_policy + .5*loss_value - cfg.entropy*entropy.mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite PPO loss; training stopped.')
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), .5, error_if_nonfinite=True)
                    optimizer.step()
                    with torch.no_grad():
                        policy.log_std.clamp_(-3, .3)
                    stats.append([loss_policy.item(), loss_value.item(), kl.item(), entropy.mean().item(),
                                  ((ratio-1).abs() > cfg.clip).float().mean().item()])
                if halt:
                    break
            update += 1
            mean = np.mean(stats, axis=0).tolist() if stats else [0.]*5
            row = dict(update=update, steps=steps, reward=float(np.mean([e['return_'] for e in episodes])) if episodes else None,
                       progress=float(np.mean([e['progress'] for e in episodes])) if episodes else None,
                       drift=float(np.mean([e['drift'] for e in episodes])) if episodes else None,
                       offtrack=float(np.mean([e['offtrack'] for e in episodes])) if episodes else None,
                       episodes=len(episodes), policy_loss=mean[0], value_loss=mean[1], kl=mean[2], entropy=mean[3],
                       clip_fraction=mean[4], kl_stopped=halt, style_scale=envs[0].style_scale,
                       fps=cfg.rollout*cfg.environments/max(.01, time.monotonic()-roll_started),
                       elapsed=time.monotonic()-started, time=time.time())
            if update % cfg.eval_every == 0 or update == cfg.updates:
                status = 'evaluating'
                report()
                result = evaluate_policy(policy, cfg, cancel=stop)
                if result is not None:
                    row['evaluation'] = result
                    atomic_json(path/'evaluation.json', result)
                    if result['score'] > best:
                        best = result['score']
                        checkpoint('best.pt')
                status = 'training'
            with (path/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False)+'\n')
            if update % cfg.save_every == 0:
                checkpoint()
            report(metrics=row)
        checkpoint()
        status = 'stopped' if stop.is_set() else 'completed'
        report(saved_at=time.time())
    except Exception as exc:
        status = 'failed'
        (path/'error.log').write_text(traceback.format_exc())
        report(error=str(exc))


def evaluation_worker(checkpoint, output):
    torch.set_num_threads(1)
    try:
        policy, payload = load_checkpoint(checkpoint)
        result = evaluate_policy(policy, Config.parse(payload['config']))
        atomic_json(output, dict(status='completed', checkpoint=Path(checkpoint).name, **result))
    except Exception as exc:
        atomic_json(output, dict(status='failed', error=str(exc)))
