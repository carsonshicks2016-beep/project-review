"""
PPO reinforcement-learning trainer -- the path to a genuinely fast lap.

A single actor-critic policy controls the whole Supra (steer, throttle, brake,
clutch, sequential up/down shifts) -- the same action space the GA uses, so a
trained policy is a drop-in ``TorchBrain`` you can watch live.

  python run.py --ppo 1500            # train (parallel, headless), save ppo_supra.pt
  python run.py --ppo 1500 --workers 8
  python run.py --watch-ppo           # watch the trained policy drive

What makes the lap *fast* (not just "complete"):
  * Reward = centre-line progress per frame, on a fixed TIME budget (see
    ``supra/ppo_env.py``).  More progress in the same time == more speed.
  * Time-budget truncation bootstraps the value; crashes/stalls are true
    terminals -> the policy learns the limit instead of the lap count.
  * Running observation normalisation + a look-ahead curvature preview in the
    observation (so it can plan braking and apexes) + domain randomisation over
    a curriculum of tracks -> it learns to *drive any track*, not memorise one.
  * Parallel rollout workers gather experience across many tracks at once.
"""

from __future__ import annotations
import math
import os
import shutil
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .track import stage_pool, STAGES
from .ppo_env import (SupraEnv, RunningNorm, make_vec_env, step_reward,
                      sample_kind)

OBS_VERSION = 2          # bump when the observation layout changes

# Bound the policy's action-noise (log std).  Without a CEILING the entropy
# bonus inflates std without limit until the squashed actions saturate into
# random bang-bang control (the failure mode that froze the old champion);
# clamping in forward() also zeroes the entropy gradient past the ceiling, which
# is what actually halts the runaway.  Without a FLOOR the policy can collapse
# to a brittle near-deterministic mode and stop exploring.
LOG_STD_MIN, LOG_STD_MAX = -2.5, 0.0     # std in [0.082, 1.0]


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------
def _orthogonal(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    """Actor-critic.  Optionally a HYBRID policy: a continuous Gaussian head
    (steer/throttle/brake/clutch) plus a categorical gear head {down,hold,up}.

    forward() -> (cont_mean, cont_std, gear_logits | None, value).
    The action vector is [cont(n_cont)] or [cont(n_cont), gear_class] when
    gear_head; log-prob/entropy sum across the two independent factors."""

    def __init__(self, n_in, n_out, hidden=(128, 128), gear_head=False, n_gear=3):
        super().__init__()
        self.gear_head = bool(gear_head)
        self.n_gear = n_gear
        # gear_head replaces the 2 boolean shift outputs with a categorical head
        self.n_cont = (n_out - 2) if self.gear_head else n_out
        layers, last = [], n_in
        for h in hidden:
            layers += [_orthogonal(nn.Linear(last, h), math.sqrt(2.0)), nn.Tanh()]
            last = h
        self.body = nn.Sequential(*layers)
        self.mean = _orthogonal(nn.Linear(last, self.n_cont), 0.01)  # tiny -> calm
        self.value = _orthogonal(nn.Linear(last, 1), 1.0)
        self.log_std = nn.Parameter(-0.5 * torch.ones(self.n_cont))
        # bias toward "drive forward, clutch engaged" (+ "no shift" in legacy mode)
        with torch.no_grad():
            self.mean.bias[:] = torch.tensor(
                [0.0, 1.0, -1.5, 2.0, -2.0, -2.0])[:self.n_cont]
        if self.gear_head:
            self.gear = _orthogonal(nn.Linear(last, n_gear), 0.01)
            with torch.no_grad():
                self.gear.bias[:] = torch.tensor([0.0, 0.5, 0.0])   # mild "hold" prior

    def forward(self, x):
        z = self.body(x)
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        gl = self.gear(z) if self.gear_head else None
        return self.mean(z), std, gl, self.value(z).squeeze(-1)

    def act(self, x, gear_action=True):
        mean, std, gl, value = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        ca = dist.sample()
        lp = dist.log_prob(ca).sum(-1)
        if self.gear_head and gear_action:
            cat = torch.distributions.Categorical(logits=gl)
            ga = cat.sample()
            lp = lp + cat.log_prob(ga)
            return torch.cat([ca, ga.unsqueeze(-1).to(ca.dtype)], -1), lp, value
        return ca, lp, value      # continuous-only (gear-assist) or non-gear-head

    def evaluate(self, x, a, gear_action=True):
        """-> (log_prob, entropy, value, gear_logits|None) under the current net.
        gear_action=False (gear-assist phase): the gear is env-controlled, so the
        policy log-prob/entropy cover the CONTINUOUS dims only; gl is still
        returned for the behaviour-cloning loss."""
        mean, std, gl, value = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        if self.gear_head and gear_action:
            ca, ga = a[..., :self.n_cont], a[..., self.n_cont].long()
            cat = torch.distributions.Categorical(logits=gl)
            lp = dist.log_prob(ca).sum(-1) + cat.log_prob(ga)
            ent = dist.entropy().sum(-1) + cat.entropy()
            return lp, ent, value, gl
        ca = a[..., :self.n_cont]
        return dist.log_prob(ca).sum(-1), dist.entropy().sum(-1), value, gl


class RecurrentActorCritic(nn.Module):
    """Encoder MLP -> LSTM -> (policy mean, value).  Same action contract as
    ActorCritic, but carries hidden state so it can anticipate."""

    def __init__(self, n_in, n_out, enc=(128,), lstm_hidden=128):
        super().__init__()
        layers, last = [], n_in
        for h in enc:
            layers += [_orthogonal(nn.Linear(last, h), math.sqrt(2.0)), nn.Tanh()]
            last = h
        self.enc = nn.Sequential(*layers)
        self.lstm = nn.LSTM(last, lstm_hidden)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param, 1.0)
            elif "bias" in name:
                nn.init.zeros_(param)
        self.mean = _orthogonal(nn.Linear(lstm_hidden, n_out), 0.01)
        self.value = _orthogonal(nn.Linear(lstm_hidden, 1), 1.0)
        self.log_std = nn.Parameter(-0.5 * torch.ones(n_out))
        self.lstm_hidden = lstm_hidden
        with torch.no_grad():
            self.mean.bias[:] = torch.tensor(
                [0.0, 1.0, -1.5, 2.0, -2.0, -2.0])[:n_out]

    def initial_state(self, batch):
        h = torch.zeros(1, batch, self.lstm_hidden)
        return (h, h.clone())

    def step(self, x, state):
        """One timestep. x: [B, n_in], state: (h, c) -> mean,std,value,new_state."""
        z = self.enc(x).unsqueeze(0)             # [1, B, enc]
        out, state = self.lstm(z, state)          # [1, B, H]
        z = out.squeeze(0)
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        return self.mean(z), std, self.value(z).squeeze(-1), state

    def act(self, x, state):
        mean, std, value, state = self.step(x, state)
        dist = torch.distributions.Normal(mean, std)
        a = dist.sample()
        return a, dist.log_prob(a).sum(-1), value, state

    def seq(self, x, state, dones):
        """Whole sequence with hidden resets at episode boundaries.
        x: [T, B, n_in], dones: [T, B] -> mean[T,B,A], std, value[T,B]."""
        T, B, _ = x.shape
        z = self.enc(x)                           # [T, B, enc]
        h, c = state
        outs = []
        for t in range(T):
            o, (h, c) = self.lstm(z[t:t + 1], (h, c))
            outs.append(o)
            keep = (1.0 - dones[t]).view(1, B, 1)   # zero the state after a terminal
            h = h * keep
            c = c * keep
        out = torch.cat(outs, 0)                  # [T, B, H]
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        return self.mean(out), std, self.value(out).squeeze(-1)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _save(path, net, norm, n_obs, n_act, hidden, iteration, stage, generation=0,
          **extra):
    d = {
        "state": net.state_dict(),
        "norm": norm.state(),
        "n_obs": n_obs, "n_act": n_act, "hidden": list(hidden),
        "obs_version": OBS_VERSION,
        "iteration": iteration, "stage": stage, "generation": generation,
    }
    d.update(extra)
    torch.save(d, path)


def _try_resume(net, norm, path, n_obs, n_act, hidden, recurrent=False,
                gear_head=False):
    """Load a checkpoint iff it matches the current obs/network layout.

    If it's from an older layout (e.g. before the look-ahead observation, or a
    feed-forward file when we want recurrent) we refuse to half-load it -- back
    it up and start fresh, loudly."""
    if not os.path.exists(path):
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"(couldn't read {path}: {e}; starting fresh)")
        return None
    incompatible = (ckpt.get("n_obs") != n_obs or ckpt.get("n_act") != n_act
                    or tuple(ckpt.get("hidden", (96, 96))) != tuple(hidden)
                    or ckpt.get("obs_version") != OBS_VERSION
                    or bool(ckpt.get("recurrent", False)) != bool(recurrent)
                    or bool(ckpt.get("gear_head", False)) != bool(gear_head))
    if incompatible:
        bak = f"{path}.legacy-{int(time.time())}"
        try:
            shutil.copy(path, bak)
            note = f" (backed up to {os.path.basename(bak)})"
        except Exception:
            note = ""
        print(f"** {os.path.basename(path)} is from an older layout{note}; "
              f"starting FRESH -- the new superhuman setup must be retrained.")
        return None
    try:
        net.load_state_dict(ckpt["state"])
        # repair a blown-up action-noise from an older (pre-clamp) checkpoint
        if hasattr(net, "log_std"):
            with torch.no_grad():
                net.log_std.clamp_(LOG_STD_MIN, LOG_STD_MAX)
        if "norm" in ckpt:
            norm.load(ckpt["norm"])
        print(f"resumed PPO policy from {path} "
              f"(iter {ckpt.get('iteration', 0)}, stage {ckpt.get('stage', '?')})")
        return ckpt
    except Exception as e:
        print(f"(couldn't resume {path}: {e}; starting fresh)")
        return None


# ---------------------------------------------------------------------------
# Headless trainer (parallel) -- the main path to a superhuman lap
# ---------------------------------------------------------------------------
def train_ppo(cfg: Config, iterations=1500, seed=0, save_path="ppo_supra.pt",
              resume=True, n_workers=None, objective="race"):
    p = cfg.ppo
    if p.recurrent:
        return _train_recurrent(cfg, iterations, seed, save_path, resume,
                                n_workers, objective)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cur = cfg.curriculum
    n_obs, n_act = cfg.sensors.n_inputs, cfg.sensors.n_outputs
    # Hybrid gear action is a RACE feature (drift auto-shifts).
    gear_head = bool(p.gear_head) and (objective == "race")
    cont_dim = (n_act - 2) if gear_head else n_act        # continuous action dims
    act_full = (cont_dim + 1) if gear_head else n_act     # full action incl. gear
    # BC teacher (rpm-based shift schedule) -- indices into the RAW observation
    car = cfg.car
    RPM_IDX = cfg.sensors.n_rays + 4
    GEAR_IDX = cfg.sensors.n_rays + 5
    n_gears = len(car.gear_ratios)

    net = ActorCritic(n_obs, n_act, hidden=p.hidden, gear_head=gear_head)
    norm = RunningNorm(n_obs, clip=p.obs_clip)
    stage = cur.start_stage if cur.enabled else len(STAGES) - 1
    start_it = 0
    if resume:
        ckpt = _try_resume(net, norm, save_path, n_obs, n_act, p.hidden,
                           gear_head=gear_head)
        if ckpt:
            start_it = int(ckpt.get("iteration", 0))
            if cur.enabled and "stage" in ckpt:
                stage = int(ckpt["stage"])
    opt = torch.optim.Adam(net.parameters(), lr=p.lr)

    vec = make_vec_env(cfg, p.n_envs, n_workers if n_workers is not None
                       else p.n_workers, base_seed=int(rng.integers(1, 1_000_000)),
                       stage=stage, objective=objective)
    n_proc = len(getattr(vec, "counts", [vec.n_envs]))
    print(f"PPO [{objective}]: {vec.n_envs} envs across {n_proc} worker process(es)  "
          f"obs={n_obs} act={n_act} hidden={p.hidden}  stage={stage}")

    # A separate BEST checkpoint: in RL the *latest* policy is not always the
    # best (stage-promotion dips, instability), and `save_path` is overwritten
    # every 25 iters.  "Best" is stage-aware -- a higher curriculum stage always
    # wins, and within a stage we keep the highest rolling-average distance --
    # because raw distance isn't comparable across stages (easy laps are longer).
    best_path = os.path.splitext(save_path)[0] + ".best.pt"
    best_stage, best_score = -1, -1e18
    if os.path.exists(best_path):
        try:
            _b = torch.load(best_path, map_location="cpu", weights_only=False)
            best_stage = int(_b.get("stage", -1))
            best_score = float(_b.get("best_score", -1e18))
            print(f"best-so-far: {os.path.basename(best_path)} "
                  f"(stage {best_stage}, avg {best_score:.0f} m)")
        except Exception:
            pass

    obs_raw = vec.reset()
    norm.update(obs_raw)
    ep_ret = np.zeros(vec.n_envs)
    recent_ret = deque(maxlen=200)
    recent_best = deque(maxlen=200)
    laps_done = deque(maxlen=cur.promote_window)

    total = start_it + iterations
    saved_it = start_it
    prev_assist = None
    t0 = time.time()
    try:
        for it in range(start_it, total):
            saved_it = it + 1
            # linear LR + entropy anneal over THIS run's iterations
            frac = 1.0 - (it - start_it) / max(1, iterations)
            for grp in opt.param_groups:
                grp["lr"] = p.lr * (p.lr_final_frac + (1 - p.lr_final_frac) * frac)
            ent_now = p.ent_coef * (p.ent_final_frac + (1 - p.ent_final_frac) * frac)

            # gear curriculum: full env-shift assist, then a GRADUAL per-iteration
            # handover (each iter is cleanly env-shift or policy-shift, with the
            # policy-shift probability ramping 0->1), then full policy control.
            # The ramp keeps the field anchored to the healthy assist regime while
            # the policy learns gears -- a hard switch collapses driving.
            if not gear_head:
                assist = False
            elif it < p.gear_assist_until:
                assist = True
            elif it < p.gear_assist_until + p.gear_handover_iters:
                prog = (it - p.gear_assist_until) / max(1, p.gear_handover_iters)
                assist = (rng.random() > prog)     # policy-shift prob ramps up
            else:
                assist = False
            if assist != prev_assist:
                vec.set_assist(assist)
            if it == p.gear_assist_until and gear_head:
                print(f"  >> gear handover ramp begins (iter {it})")
            if it == p.gear_assist_until + p.gear_handover_iters and gear_head:
                print(f"  >> gear handover complete (iter {it}): policy controls gears")
            prev_assist = assist
            cur_act_dim = cont_dim if assist else act_full

            O, A, LP, R, V, D = [], [], [], [], [], []
            O_raw = []                          # raw obs -> rpm/gear for BC teacher
            for _ in range(p.horizon):
                obs_n = norm.normalize(obs_raw).astype(np.float32)
                with torch.no_grad():
                    a, lp, v = net.act(torch.from_numpy(obs_n), gear_action=not assist)
                a_np = a.numpy().astype(np.float32)

                (next_raw, rews, terms, truncs, finals,
                 laps, travs) = vec.step(a_np)

                # bootstrap the value of TRUNCATED (time-budget) episodes
                boot = np.zeros(vec.n_envs, dtype=np.float32)
                bmask = truncs & (~terms)
                if bmask.any():
                    fobs = norm.normalize(finals[bmask]).astype(np.float32)
                    with torch.no_grad():
                        vf = net.forward(torch.from_numpy(fobs))[3].numpy()
                    boot[bmask] = p.gamma * vf
                dones = terms | truncs

                O.append(obs_n)
                O_raw.append(obs_raw)
                A.append(a_np)
                LP.append(lp.numpy())
                R.append((rews + boot).astype(np.float32))   # training reward
                V.append(v.numpy())
                D.append(dones.astype(np.float32))

                ep_ret += rews                               # logging (raw)
                for i in range(vec.n_envs):
                    if dones[i]:
                        recent_ret.append(float(ep_ret[i]))
                        recent_best.append(float(travs[i]))
                        laps_done.append(float(laps[i]))
                        ep_ret[i] = 0.0

                obs_raw = next_raw
                norm.update(obs_raw)

            # curriculum: unlock the next tier once laps complete reliably
            if (cur.enabled and stage < len(STAGES) - 1
                    and len(laps_done) >= cur.promote_window
                    and np.mean(laps_done) >= cur.promote_threshold):
                stage += 1
                vec.set_stage(stage)
                laps_done.clear()
                print(f"  >> curriculum: promoted to stage {stage} "
                      f"(tracks: {stage_pool(stage)})")

            with torch.no_grad():
                last_v = net.forward(
                    torch.from_numpy(norm.normalize(obs_raw).astype(np.float32)))[3].numpy()

            # ---- GAE(lambda) ----
            O = np.array(O); A = np.array(A); LP = np.array(LP)
            R = np.array(R); V = np.array(V); D = np.array(D)
            adv = np.zeros_like(R)
            gae = np.zeros(vec.n_envs, dtype=np.float32)
            for t in reversed(range(p.horizon)):
                nv = last_v if t == p.horizon - 1 else V[t + 1]
                nonterm = 1.0 - D[t]
                delta = R[t] + p.gamma * nv * nonterm - V[t]
                gae = delta + p.gamma * p.lam * nonterm * gae
                adv[t] = gae
            ret = adv + V

            bO = torch.from_numpy(O.reshape(-1, n_obs))
            bA = torch.from_numpy(A.reshape(-1, cur_act_dim))
            bLP = torch.from_numpy(LP.reshape(-1))
            bAdv = torch.from_numpy(adv.reshape(-1))
            bRet = torch.from_numpy(ret.reshape(-1))
            bAdv = (bAdv - bAdv.mean()) / (bAdv.std() + 1e-8)

            # gear behaviour-cloning teacher: rpm-based shift schedule on raw obs
            bc_beta = 0.0
            bGear = None
            if gear_head:
                bc_beta = max(0.0, 1.0 - it / max(1, p.gear_bc_until))
                if bc_beta > 0.0:
                    raw_flat = np.asarray(O_raw, dtype=np.float32).reshape(-1, n_obs)
                    rpm = raw_flat[:, RPM_IDX] * car.redline_rpm
                    gear_i = np.round(raw_flat[:, GEAR_IDX] * (n_gears - 1))
                    up = (rpm > car.shift_up_rpm) & (gear_i < n_gears - 1)
                    down = (rpm < car.shift_down_rpm) & (gear_i > 0)
                    teacher = np.where(up, 2, np.where(down, 0, 1)).astype(np.int64)
                    bGear = torch.from_numpy(teacher)

            N = bO.shape[0]
            idx = np.arange(N)
            approx_kl = 0.0
            for _ in range(p.epochs):
                rng.shuffle(idx)
                kls = []
                for s in range(0, N, p.minibatch):
                    mb = idx[s:s + p.minibatch]
                    lp, ent, val, gl = net.evaluate(bO[mb], bA[mb], gear_action=not assist)
                    logratio = lp - bLP[mb]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        kls.append(((ratio - 1) - logratio).mean().item())
                    s1 = ratio * bAdv[mb]
                    s2 = torch.clamp(ratio, 1 - p.clip, 1 + p.clip) * bAdv[mb]
                    pol = -torch.min(s1, s2).mean()
                    vloss = ((val - bRet[mb]) ** 2).mean()
                    loss = pol + p.vf_coef * vloss - ent_now * ent.mean()
                    if bGear is not None:        # gear BC warm-start (annealed)
                        loss = loss + bc_beta * p.gear_bc_coef * \
                            nn.functional.cross_entropy(gl, bGear[mb])
                    if not torch.isfinite(loss):
                        opt.zero_grad(set_to_none=True)
                        continue                  # skip a NaN/inf minibatch
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), p.max_grad_norm)
                    opt.step()
                    with torch.no_grad():         # keep action-noise bounded
                        net.log_std.clamp_(LOG_STD_MIN, LOG_STD_MAX)
                approx_kl = float(np.mean(kls)) if kls else 0.0
                if p.target_kl and approx_kl > p.target_kl:
                    break

            # update the BEST checkpoint (stage-aware; needs a stable window)
            if len(recent_best) >= 30:
                score = float(np.mean(recent_best))
                if stage > best_stage or (stage == best_stage and score > best_score):
                    best_stage, best_score = stage, score
                    _save(best_path, net, norm, n_obs, n_act, p.hidden, it + 1,
                          stage, best_score=best_score, objective=objective,
                          gear_head=gear_head)

            # heartbeat every iter (deques may be empty until the first episode
            # ends -- a clean policy can drive a whole rollout without finishing)
            best_str = f"{best_score:5.0f}m@s{best_stage}" if best_stage >= 0 else "  --"
            ips = (it - start_it + 1) / max(time.time() - t0, 1e-9)
            std_mean = net.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().mean().item()
            ret_str = f"{np.mean(recent_ret):8.1f}" if recent_ret else "      --"
            trav_str = f"{np.mean(recent_best):8.1f}" if recent_best else "      --"
            compl = 100 * np.mean(laps_done) if laps_done else 0.0
            print(f"iter {it:4d}  stage {stage}  "
                  f"ep_return {ret_str}  "
                  f"travelled {trav_str} m  "
                  f"compl% {compl:4.0f}  "
                  f"std {std_mean:.3f}  "
                  f"kl {approx_kl:.3f}  best {best_str}  {ips:.2f} it/s", flush=True)
            if (it + 1) % 25 == 0 or it == total - 1:
                _save(save_path, net, norm, n_obs, n_act, p.hidden, it + 1, stage,
                      objective=objective, gear_head=gear_head)
    except KeyboardInterrupt:
        print("\n** interrupted — saving latest checkpoint before exit **")
    finally:
        try:
            _save(save_path, net, norm, n_obs, n_act, p.hidden, saved_it, stage,
                  objective=objective, gear_head=gear_head)
            print(f"\nsaved PPO policy [{objective}] -> {save_path} "
                  f"(iter {saved_it}, stage {stage})")
        except Exception as e:
            print(f"(could not save on exit: {e})")
        vec.close()
    return net


def _train_recurrent(cfg: Config, iterations, seed, save_path, resume,
                     n_workers, objective):
    """Recurrent (LSTM) PPO.  Hidden state persists across rollouts and is reset
    at episode boundaries; the update does BPTT over each env's full sequence,
    minibatched over environments."""
    p = cfg.ppo
    if bool(p.gear_head) and objective == "race":
        raise NotImplementedError(
            "the hybrid gear head is feed-forward only for now; train the race "
            "gear lineage without --recurrent (or set ppo.gear_head=False to run "
            "a recurrent race policy with the legacy boolean shifts).")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cur = cfg.curriculum
    n_obs, n_act = cfg.sensors.n_inputs, cfg.sensors.n_outputs
    enc = (p.hidden[0],)

    net = RecurrentActorCritic(n_obs, n_act, enc=enc, lstm_hidden=p.lstm_hidden)
    norm = RunningNorm(n_obs, clip=p.obs_clip)
    stage = cur.start_stage if cur.enabled else len(STAGES) - 1
    start_it = 0
    if resume:
        ckpt = _try_resume(net, norm, save_path, n_obs, n_act, enc, recurrent=True)
        if ckpt:
            start_it = int(ckpt.get("iteration", 0))
            if cur.enabled and "stage" in ckpt:
                stage = int(ckpt["stage"])
    opt = torch.optim.Adam(net.parameters(), lr=p.lr)

    def _save_rec(path, it_, st_, **extra):
        _save(path, net, norm, n_obs, n_act, enc, it_, st_,
              recurrent=True, lstm_hidden=p.lstm_hidden, objective=objective, **extra)

    vec = make_vec_env(cfg, p.n_envs, n_workers if n_workers is not None
                       else p.n_workers, base_seed=int(rng.integers(1, 1_000_000)),
                       stage=stage, objective=objective)
    N = vec.n_envs
    n_proc = len(getattr(vec, "counts", [N]))
    print(f"PPO [{objective}, LSTM-{p.lstm_hidden}]: {N} envs across {n_proc} "
          f"worker process(es)  obs={n_obs} act={n_act}  stage={stage}")

    best_path = os.path.splitext(save_path)[0] + ".best.pt"
    best_stage, best_score = -1, -1e18
    if os.path.exists(best_path):
        try:
            _b = torch.load(best_path, map_location="cpu", weights_only=False)
            if bool(_b.get("recurrent", False)):
                best_stage = int(_b.get("stage", -1))
                best_score = float(_b.get("best_score", -1e18))
        except Exception:
            pass

    obs_raw = vec.reset()
    norm.update(obs_raw)
    lstm_state = net.initial_state(N)
    ep_ret = np.zeros(N)
    recent_ret, recent_best = deque(maxlen=200), deque(maxlen=200)
    laps_done = deque(maxlen=cur.promote_window)
    mb_envs = max(1, min(p.recurrent_minibatch_envs, N))

    total = start_it + iterations
    saved_it = start_it
    t0 = time.time()
    try:
        for it in range(start_it, total):
            saved_it = it + 1
            frac = 1.0 - (it - start_it) / max(1, iterations)
            for grp in opt.param_groups:
                grp["lr"] = p.lr * (p.lr_final_frac + (1 - p.lr_final_frac) * frac)
            ent_now = p.ent_coef * (p.ent_final_frac + (1 - p.ent_final_frac) * frac)

            init_h = lstm_state[0].detach().clone()
            init_c = lstm_state[1].detach().clone()
            O, A, LP, R, V, D = [], [], [], [], [], []
            for _ in range(p.horizon):
                obs_n = norm.normalize(obs_raw).astype(np.float32)
                with torch.no_grad():
                    a, lp, v, lstm_state = net.act(torch.from_numpy(obs_n), lstm_state)
                a_np = a.numpy().astype(np.float32)
                next_raw, rews, terms, truncs, finals, laps, travs = vec.step(a_np)

                boot = np.zeros(N, dtype=np.float32)
                bmask = truncs & (~terms)
                if bmask.any():
                    with torch.no_grad():
                        fn = norm.normalize(finals).astype(np.float32)
                        vf = net.step(torch.from_numpy(fn), lstm_state)[2].numpy()
                    boot[bmask] = p.gamma * vf[bmask]
                dones = terms | truncs

                O.append(obs_n); A.append(a_np); LP.append(lp.numpy())
                R.append((rews + boot).astype(np.float32))
                V.append(v.numpy()); D.append(dones.astype(np.float32))

                # reset hidden state for envs whose episode just ended
                keep = torch.from_numpy((1.0 - dones).astype(np.float32)).view(1, N, 1)
                lstm_state = (lstm_state[0] * keep, lstm_state[1] * keep)

                ep_ret += rews
                for i in range(N):
                    if dones[i]:
                        recent_ret.append(float(ep_ret[i]))
                        recent_best.append(float(travs[i]))
                        laps_done.append(float(laps[i]))
                        ep_ret[i] = 0.0
                obs_raw = next_raw
                norm.update(obs_raw)

            if (cur.enabled and stage < len(STAGES) - 1
                    and len(laps_done) >= cur.promote_window
                    and np.mean(laps_done) >= cur.promote_threshold):
                stage += 1
                vec.set_stage(stage)
                laps_done.clear()
                print(f"  >> curriculum: promoted to stage {stage} "
                      f"(tracks: {stage_pool(stage)})")

            with torch.no_grad():
                last_v = net.step(torch.from_numpy(
                    norm.normalize(obs_raw).astype(np.float32)), lstm_state)[2].numpy()

            # ---- GAE(lambda) over [T, N] ----
            O = np.array(O); A = np.array(A); LP = np.array(LP)
            R = np.array(R); V = np.array(V); D = np.array(D)
            adv = np.zeros_like(R)
            gae = np.zeros(N, dtype=np.float32)
            for t in reversed(range(p.horizon)):
                nv = last_v if t == p.horizon - 1 else V[t + 1]
                nonterm = 1.0 - D[t]
                delta = R[t] + p.gamma * nv * nonterm - V[t]
                gae = delta + p.gamma * p.lam * nonterm * gae
                adv[t] = gae
            ret = adv + V

            tO = torch.from_numpy(O)            # [T,N,obs]
            tA = torch.from_numpy(A)            # [T,N,act]
            tLP = torch.from_numpy(LP)          # [T,N]
            tD = torch.from_numpy(D)            # [T,N]
            tAdv = torch.from_numpy(adv)
            tRet = torch.from_numpy(ret)
            tAdv = (tAdv - tAdv.mean()) / (tAdv.std() + 1e-8)

            approx_kl = 0.0
            env_idx = np.arange(N)
            for _ in range(p.epochs):
                rng.shuffle(env_idx)
                kls = []
                for s in range(0, N, mb_envs):
                    mb = env_idx[s:s + mb_envs]
                    st0 = (init_h[:, mb].contiguous(), init_c[:, mb].contiguous())
                    mean, std, val = net.seq(tO[:, mb], st0, tD[:, mb])
                    dist = torch.distributions.Normal(mean, std)
                    lp = dist.log_prob(tA[:, mb]).sum(-1).reshape(-1)
                    ent = dist.entropy().sum(-1).reshape(-1)
                    old = tLP[:, mb].reshape(-1)
                    adv_f = tAdv[:, mb].reshape(-1)
                    ret_f = tRet[:, mb].reshape(-1)
                    logratio = lp - old
                    ratio = logratio.exp()
                    with torch.no_grad():
                        kls.append(((ratio - 1) - logratio).mean().item())
                    s1 = ratio * adv_f
                    s2 = torch.clamp(ratio, 1 - p.clip, 1 + p.clip) * adv_f
                    loss = (-torch.min(s1, s2).mean()
                            + p.vf_coef * ((val.reshape(-1) - ret_f) ** 2).mean()
                            - ent_now * ent.mean())
                    if not torch.isfinite(loss):
                        opt.zero_grad(set_to_none=True)
                        continue                  # skip a NaN/inf minibatch
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), p.max_grad_norm)
                    opt.step()
                    with torch.no_grad():         # keep action-noise bounded
                        net.log_std.clamp_(LOG_STD_MIN, LOG_STD_MAX)
                approx_kl = float(np.mean(kls)) if kls else 0.0
                if p.target_kl and approx_kl > p.target_kl:
                    break

            if len(recent_best) >= 30:
                score = float(np.mean(recent_best))
                if stage > best_stage or (stage == best_stage and score > best_score):
                    best_stage, best_score = stage, score
                    _save_rec(best_path, it + 1, stage, best_score=best_score)

            best_str = f"{best_score:5.0f}m@s{best_stage}" if best_stage >= 0 else "  --"
            ips = (it - start_it + 1) / max(time.time() - t0, 1e-9)
            std_mean = net.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().mean().item()
            ret_str = f"{np.mean(recent_ret):8.1f}" if recent_ret else "      --"
            trav_str = f"{np.mean(recent_best):8.1f}" if recent_best else "      --"
            compl = 100 * np.mean(laps_done) if laps_done else 0.0
            print(f"iter {it:4d}  stage {stage}  "
                  f"ep_return {ret_str}  "
                  f"travelled {trav_str} m  "
                  f"compl% {compl:4.0f}  "
                  f"std {std_mean:.3f}  "
                  f"kl {approx_kl:.3f}  best {best_str}  {ips:.2f} it/s", flush=True)
            if (it + 1) % 25 == 0 or it == total - 1:
                _save_rec(save_path, it + 1, stage)
    except KeyboardInterrupt:
        print("\n** interrupted — saving latest checkpoint before exit **")
    finally:
        try:
            _save_rec(save_path, saved_it, stage)
            print(f"\nsaved recurrent PPO policy [{objective}] -> {save_path} "
                  f"(iter {saved_it}, stage {stage})")
        except Exception as e:
            print(f"(could not save on exit: {e})")
        vec.close()
    return net


# ---------------------------------------------------------------------------
# Inference wrapper: a trained PPO policy as a Brain in the live app
# ---------------------------------------------------------------------------
class TorchBrain:
    """Drop-in for MLPGenome: .forward(obs) -> 6 controls (normalised input).

    Supports both the feed-forward and the recurrent (LSTM) policy; for the
    recurrent one it carries per-instance hidden state across frames and zeroes
    it on .reset() (called when the car respawns)."""

    def __init__(self, path, stochastic=False):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.recurrent = bool(ckpt.get("recurrent", False))
        self.gear_head = bool(ckpt.get("gear_head", False))
        hidden = tuple(ckpt.get("hidden", (96, 96)))
        if self.recurrent:
            self.net = RecurrentActorCritic(
                ckpt["n_obs"], ckpt["n_act"], enc=hidden,
                lstm_hidden=int(ckpt.get("lstm_hidden", 128)))
        else:
            self.net = ActorCritic(ckpt["n_obs"], ckpt["n_act"], hidden=hidden,
                                   gear_head=self.gear_head)
        self.net.load_state_dict(ckpt["state"])
        self.net.eval()
        self.norm = RunningNorm(ckpt["n_obs"])
        if "norm" in ckpt:
            self.norm.load(ckpt["norm"])
        self.stochastic = stochastic
        self.path = path
        self.state = self.net.initial_state(1) if self.recurrent else None

    def reset(self):
        """Clear LSTM memory (called when the car respawns)."""
        if self.recurrent:
            self.state = self.net.initial_state(1)

    def reload(self, path=None):
        """Hot-reload weights from disk (for watching PPO train live)."""
        try:
            ckpt = torch.load(path or self.path, map_location="cpu",
                              weights_only=False)
            self.net.load_state_dict(ckpt["state"])
            self.net.eval()
            if "norm" in ckpt:
                self.norm.load(ckpt["norm"])
            return True
        except Exception:
            return False        # checkpoint mid-write; try again next tick

    def forward(self, x):
        xn = self.norm.normalize(x).astype(np.float32)
        with torch.no_grad():
            if self.recurrent:
                mean, std, _, self.state = self.net.step(
                    torch.from_numpy(xn).unsqueeze(0), self.state)
                raw = (torch.distributions.Normal(mean, std).sample()[0].numpy()
                       if self.stochastic else mean[0].numpy())
            elif self.gear_head:
                mean, std, gl, _ = self.net.forward(torch.from_numpy(xn))
                if self.stochastic:
                    cont = torch.distributions.Normal(mean, std).sample().numpy()
                    gear = int(torch.distributions.Categorical(logits=gl).sample())
                else:
                    cont = mean.numpy()
                    gear = int(torch.argmax(gl))
                raw = np.concatenate([cont, [float(gear)]])
            elif self.stochastic:
                raw = self.net.act(torch.from_numpy(xn))[0].numpy()
            else:
                raw = self.net.forward(torch.from_numpy(xn))[0].numpy()
        return SupraEnv.act_to_controls(raw, self.gear_head)


# ---------------------------------------------------------------------------
# Live "train while you watch" mode (secondary; --ppo is the superhuman path)
# ---------------------------------------------------------------------------
def _gae(rews, vals, dones, gamma, lam):
    adv = np.zeros(len(rews), np.float32)
    gae = 0.0
    for t in reversed(range(len(rews))):
        nextv = vals[t + 1] if t + 1 < len(vals) else 0.0
        nonterm = 1.0 - dones[t]
        delta = rews[t] + gamma * nextv * nonterm - vals[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
    return adv, adv + np.asarray(vals, np.float32)


class LivePPOBrain:
    """Samples from a shared net and records each transition, so a field of
    rendered cars doubles as the PPO experience source.  Observations are
    normalised with the shared running normaliser (updated online)."""

    def __init__(self, net, norm):
        self.net = net
        self.norm = norm
        self.pending = None              # (obs_norm, raw_action, logp, value)
        self.prev_raw = np.zeros(net.mean.out_features, np.float32)

    def forward(self, x):
        x = np.asarray(x, np.float32)
        self.norm.update(x)
        xn = self.norm.normalize(x).astype(np.float32)
        with torch.no_grad():
            mean, std, _gl, val = self.net.forward(torch.from_numpy(xn))
            dist = torch.distributions.Normal(mean, std)
            a = dist.sample()
            lp = float(dist.log_prob(a).sum(-1))
        raw = a.numpy()
        self.pending = (xn, raw, lp, float(val))
        return SupraEnv.act_to_controls(raw)


def watch_ppo_live(cfg: Config, seed=0, save_path="ppo_supra.pt",
                   batch=4000, epochs=6, minibatch=2048, max_frames=None,
                   resume=True):
    """Single window: train PPO while watching the field learn in real time."""
    import pygame
    from .app import App

    p = cfg.ppo
    # The single-window live trainer is legacy (boolean shifts) only.  If the
    # save file holds a hybrid gear-head policy, refuse rather than clobber it.
    if os.path.exists(save_path):
        try:
            _peek = torch.load(save_path, map_location="cpu", weights_only=False)
            if bool(_peek.get("gear_head", False)):
                print(f"{save_path} is a hybrid gear-head policy; --ppo-live "
                      "doesn't support it yet. Train headless with --ppo and watch "
                      "with `--watch-ppo --reload` instead.")
                return
        except Exception:
            pass
    app = App(cfg, seed=seed)
    n_obs, n_act = cfg.sensors.n_inputs, cfg.sensors.n_outputs
    net = ActorCritic(n_obs, n_act, hidden=p.hidden)
    norm = RunningNorm(n_obs, clip=p.obs_clip)
    iters, start_gen = 0, 0
    stage = cfg.curriculum.start_stage if cfg.curriculum.enabled else len(STAGES) - 1
    if resume:
        ckpt = _try_resume(net, norm, save_path, n_obs, n_act, p.hidden)
        if ckpt:
            iters = int(ckpt.get("iteration", 0))
            start_gen = int(ckpt.get("generation", 0))
            app.sim.generation = start_gen
            if cfg.curriculum.enabled and "stage" in ckpt:
                stage = int(ckpt["stage"])
    opt = torch.optim.Adam(net.parameters(), lr=p.lr)
    brains = [LivePPOBrain(net, norm) for _ in range(cfg.evo.population)]

    cur = cfg.curriculum
    rng = np.random.default_rng(seed)
    if cur.enabled:
        app.sim.new_track(kind=sample_kind(rng, stage))
    app.sim.set_population(brains, app.evo.colors())
    app.evolve_enabled = False

    buf = [[] for _ in brains]           # per-car (obs,act,logp,val,rew,done)
    recent = deque(maxlen=400)
    laps_done = deque(maxlen=cur.promote_window)
    app.sound.start()
    running, frame = True, 0
    while running:
        dt = cfg.sim.dt
        running = app._events()
        frame += 1
        if max_frames is not None and frame > max_frames:
            running = False
        if not app.paused:
            prev_dist = [c.distance for c in app.sim.cars]
            prev_alive = [c.alive for c in app.sim.cars]
            gen_done = app.sim.step(dt)
            if app.auto_follow:
                app.sim.focus_leader()
            for i, c in enumerate(app.sim.cars):
                b = brains[i]
                if prev_alive[i] and b.pending is not None:
                    obs, act, lp, val = b.pending
                    jerk = float(np.mean((act - b.prev_raw) ** 2))
                    b.prev_raw = act
                    crashed = (not c.alive) and not getattr(c, "finished", False)
                    progress = c.distance - prev_dist[i]
                    r = step_reward(progress, c.vehicle.speed, jerk, dt, p, crashed)
                    buf[i].append((obs, act, lp, val, r, float(crashed)))
                    b.pending = None
                app._emit_effects(c)
            app._update_smoke(dt)
            if gen_done:
                tlen = app.sim.track.length
                cap = cur.promote_fraction_cap
                for c in app.sim.cars:
                    recent.append(c.best_distance)
                    laps_done.append(min(c.best_distance / max(tlen, 1.0), cap))
                if (cur.enabled and stage < len(STAGES) - 1
                        and len(laps_done) >= cur.promote_window
                        and np.mean(laps_done) >= cur.promote_threshold):
                    stage += 1
                    laps_done.clear()
                    print(f">> promoted to stage {stage}: {stage_pool(stage)}")
                if cur.enabled:
                    app.sim.new_track(kind=sample_kind(rng, stage))
                app.sim.reset_episode()
                app.smoke.clear(); app.skid.clear()

            if sum(len(b) for b in buf) >= batch:
                iters += 1
                _ppo_update(net, opt, buf, epochs, minibatch, p.gamma, p.lam,
                            p.clip, p.ent_coef, p.vf_coef, p.max_grad_norm, n_obs, n_act)
                buf = [[] for _ in brains]
                _save(save_path, net, norm, n_obs, n_act, p.hidden, iters,
                      stage, generation=app.sim.generation)

        fc = app.sim.focus_car
        if fc is not None:
            if app.sim.focus != app._prev_focus:
                app.cam[0], app.cam[1] = fc.vehicle.x, fc.vehicle.y
                app._prev_focus = app.sim.focus
            else:
                app.cam[0] += (fc.vehicle.x - app.cam[0]) * 0.12
                app.cam[1] += (fc.vehicle.y - app.cam[1]) * 0.12
        app.sound.update(app.sim.cars, app.sim.focus, app.cam,
                         app.view_w / 2 * app.mpp, muted=app.muted)

        app.screen.fill(cfg.render.grass_color)
        app._draw_track(); app._draw_smoke()
        for c in app.sim.cars:
            app._draw_car(c, focus=(c is fc))
        app._draw_flames()
        mr = (np.mean(recent) if recent else 0.0)
        lap_rate = (np.mean(laps_done) * 100) if laps_done else 0.0
        app.screen.blit(app.font_b.render(
            f"● LIVE PPO   upd {iters}   stage {stage} ({app.sim.track.kind})   "
            f"avg {mr:5.0f}m   lap% {lap_rate:3.0f}   std {net.log_std.exp().mean().item():.2f}",
            True, (90, 210, 120)), (12, 12))
        app.screen.blit(app.font.render(
            "[SPACE] pause  [T] fast  [A] follow  [+/-] zoom  [ESC] quit",
            True, (140, 146, 158)), (12, app.view_h - 24))
        app.dash.render(app.screen, app.sim, app.evo)
        pygame.display.flip()
        app.clock.tick(cfg.render.fps)

    _save(save_path, net, norm, n_obs, n_act, p.hidden, iters, stage,
          generation=app.sim.generation)
    print(f"\nsaved PPO policy on exit -> {save_path} "
          f"(iter {iters}, gen {app.sim.generation}, stage {stage})")
    app.sound.stop()
    pygame.quit()


def _ppo_update(net, opt, buf, epochs, minibatch, gamma, lam, clip,
                ent_coef, vf_coef, max_grad_norm, n_obs, n_act):
    O, A, LP, ADV, RET = [], [], [], [], []
    for traj in buf:
        if not traj:
            continue
        obs = np.array([t[0] for t in traj], np.float32)
        act = np.array([t[1] for t in traj], np.float32)
        lp = np.array([t[2] for t in traj], np.float32)
        val = np.array([t[3] for t in traj], np.float32)
        rew = np.array([t[4] for t in traj], np.float32)
        dn = np.array([t[5] for t in traj], np.float32)
        adv, ret = _gae(rew, val, dn, gamma, lam)
        O.append(obs); A.append(act); LP.append(lp); ADV.append(adv); RET.append(ret)
    if not O:
        return
    bO = torch.from_numpy(np.concatenate(O))
    bA = torch.from_numpy(np.concatenate(A))
    bLP = torch.from_numpy(np.concatenate(LP))
    bAdv = torch.from_numpy(np.concatenate(ADV))
    bRet = torch.from_numpy(np.concatenate(RET))
    bAdv = (bAdv - bAdv.mean()) / (bAdv.std() + 1e-8)
    N = bO.shape[0]
    idx = np.arange(N)
    for _ in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, N, minibatch):
            mb = idx[s:s + minibatch]
            lp, ent, val, _ = net.evaluate(bO[mb], bA[mb])
            ratio = torch.exp(lp - bLP[mb])
            s1 = ratio * bAdv[mb]
            s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * bAdv[mb]
            loss = (-torch.min(s1, s2).mean()
                    + vf_coef * ((val - bRet[mb]) ** 2).mean()
                    - ent_coef * ent.mean())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            opt.step()
