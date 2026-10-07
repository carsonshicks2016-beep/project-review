"""Evader jamming policy training and runtime helpers."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .comms import AUTH_WORDS, VOCABULARY, WORD_TO_TOKEN
from .imitation import require_torch
from .observations import OBS_SIZE, build_observation
from .sim import HeistSim

SPOOF_LEN = 5
TARGET_COUNT = 5


class JammerPolicy(require_torch()[1].Module):
    """Predict jam trigger, spoof token payload, and target victim mask."""

    def __init__(self, obs_dim: int = OBS_SIZE, vocab_size: int = len(VOCABULARY), hidden: int = 128):
        torch, nn = require_torch()
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.vocab_size = int(vocab_size)
        self.hidden = int(hidden)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.trigger = nn.Linear(hidden, 2)
        self.spoof = nn.Linear(hidden, SPOOF_LEN * vocab_size)
        self.targets = nn.Linear(hidden, TARGET_COUNT)

    def forward(self, obs):
        h = self.encoder(obs)
        return {
            "trigger_logits": self.trigger(h),
            "spoof_logits": self.spoof(h).reshape(obs.shape[0], SPOOF_LEN, self.vocab_size),
            "target_logits": self.targets(h),
        }

    def relaxed_spoof_tokens(self, obs, tau: float = 0.9, hard: bool = True):
        import torch.nn.functional as functional

        logits = self.forward(obs)["spoof_logits"]
        flat = logits.reshape(-1, self.vocab_size)
        relaxed = functional.gumbel_softmax(flat, tau=tau, hard=hard)
        return relaxed.reshape(obs.shape[0], SPOOF_LEN, self.vocab_size)


@dataclass
class JammerTrainingResult:
    checkpoint: str
    samples: int
    epochs: int
    final_loss: float
    trigger_accuracy: float
    positive_rate: float
    stats: list[dict[str, Any]]
    mean_counterfactual_reward: float = 0.0
    training_mode: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "samples": self.samples,
            "epochs": self.epochs,
            "final_loss": self.final_loss,
            "trigger_accuracy": self.trigger_accuracy,
            "positive_rate": self.positive_rate,
            "stats": self.stats,
            "mean_counterfactual_reward": self.mean_counterfactual_reward,
            "training_mode": self.training_mode,
        }


def collect_jammer_dataset(seed: int = 11, steps: int = 3600) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect evader observations and heuristic jamming targets from scripted chases."""
    sim = HeistSim(seed=seed, reset_on_capture=True)
    sim.jam_cooldown = 0.0
    obs_rows: list[np.ndarray] = []
    trigger_rows: list[int] = []
    spoof_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    for _ in range(int(steps)):
        obs_rows.append(build_observation(sim, "evader_0"))
        trigger, spoof, targets = heuristic_jam_label(sim)
        trigger_rows.append(trigger)
        spoof_rows.append(spoof)
        target_rows.append(targets)
        sim.step()
        if sim.episode_done and sim.reset_on_capture:
            sim.jam_cooldown = 0.0
    return (
        np.stack(obs_rows).astype(np.float32),
        np.asarray(trigger_rows, dtype=np.int64),
        np.stack(spoof_rows).astype(np.int64),
        np.stack(target_rows).astype(np.float32),
    )


def heuristic_jam_label(sim: HeistSim) -> tuple[int, np.ndarray, np.ndarray]:
    ev = sim.evader.vehicle
    dists = np.asarray([np.hypot(p.vehicle.x - ev.x, p.vehicle.y - ev.y) for p in sim.pursuers], dtype=np.float32)
    pressure = int(np.sum(dists < 125.0))
    close = float(np.min(dists))
    ready = sim.channel.jamming_budget > 0
    trigger = int(ready and (pressure >= 2 or close < 110.0))

    spoof, target_mask = jam_candidate_payload(sim, dists)
    return trigger, spoof, target_mask


def jam_candidate_payload(sim: HeistSim, dists: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return the current best spoof payload and target mask without deciding trigger."""
    if dists is None:
        ev = sim.evader.vehicle
        dists = np.asarray([np.hypot(p.vehicle.x - ev.x, p.vehicle.y - ev.y) for p in sim.pursuers], dtype=np.float32)
    target_mask = np.zeros(TARGET_COUNT, dtype=np.float32)
    victims = np.argsort(dists)[:2]
    target_mask[victims] = 1.0

    spoof = _spoof_payload(sim, int(victims[0]))
    return spoof, target_mask


def collect_counterfactual_jammer_dataset(
    seed: int = 11,
    steps: int = 1200,
    sample_every: int = 12,
    horizon_steps: int = 72,
    positive_threshold: float = 0.16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect jammer labels weighted by paired jam/no-jam trajectory deviation.

    Each sampled state is cloned into two deterministic branches. One branch
    receives the candidate spoof payload, the other has jamming disabled. The
    returned reward is a normalized measure of how much the spoof branch changed
    pursuer trajectories and damaged scanner confidence.
    """
    sim = HeistSim(seed=seed, reset_on_capture=True)
    _disable_future_jamming(sim)
    obs_rows: list[np.ndarray] = []
    trigger_rows: list[int] = []
    spoof_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    weight_rows: list[float] = []
    reward_rows: list[float] = []
    sample_every = max(1, int(sample_every))
    for step in range(int(steps)):
        if step % sample_every == 0:
            obs_rows.append(build_observation(sim, "evader_0"))
            spoof, targets = jam_candidate_payload(sim)
            score = counterfactual_jam_reward(
                sim,
                spoof_tokens=spoof,
                target_mask=targets,
                horizon_steps=horizon_steps,
            )
            trigger_rows.append(int(score >= positive_threshold))
            spoof_rows.append(spoof)
            target_rows.append(targets)
            weight_rows.append(1.0 + 3.0 * float(score))
            reward_rows.append(float(score))
        _disable_future_jamming(sim)
        sim.step()
        if sim.episode_done and sim.reset_on_capture:
            _disable_future_jamming(sim)

    if not obs_rows:
        raise RuntimeError("counterfactual jammer dataset collection produced no samples")
    return (
        np.stack(obs_rows).astype(np.float32),
        np.asarray(trigger_rows, dtype=np.int64),
        np.stack(spoof_rows).astype(np.int64),
        np.stack(target_rows).astype(np.float32),
        np.asarray(weight_rows, dtype=np.float32),
        np.asarray(reward_rows, dtype=np.float32),
    )


def counterfactual_jam_reward(
    sim: HeistSim,
    *,
    spoof_tokens: np.ndarray,
    target_mask: np.ndarray,
    horizon_steps: int = 72,
) -> float:
    """Estimate the reward contribution of one spoof by paired rollout."""
    jammed = copy.deepcopy(sim)
    baseline = copy.deepcopy(sim)
    _disable_future_jamming(baseline)
    jammed.jam_cooldown = 0.0
    jammed.channel.jamming_budget = max(1, int(jammed.channel.jamming_budget))

    jam_positions: list[np.ndarray] = []
    baseline_positions: list[np.ndarray] = []
    horizon = max(1, int(horizon_steps))
    jam_action = {
        "evader_0": {
            "jam": 1,
            "spoof_tokens": np.asarray(spoof_tokens, dtype=np.int64),
            "target_mask": np.asarray(target_mask, dtype=np.int64),
        }
    }
    for t in range(horizon):
        if t == 0:
            jammed.step(actions=jam_action)
            _disable_future_jamming(jammed)
        else:
            jammed.step()
        _disable_future_jamming(baseline)
        baseline.step()
        jam_positions.append(_pursuer_xy(jammed))
        baseline_positions.append(_pursuer_xy(baseline))

    deviation = np.linalg.norm(
        np.asarray(jam_positions, dtype=np.float32) - np.asarray(baseline_positions, dtype=np.float32),
        axis=2,
    )
    mean_deviation = float(deviation.mean()) if deviation.size else 0.0
    confidence_damage = max(0.0, float(baseline.channel.confidence) - float(jammed.channel.confidence))
    deception_lift = max(0.0, float(jammed.deception_score) - float(baseline.deception_score))
    return float(np.clip(
        0.68 * min(1.0, mean_deviation / 60.0)
        + 0.22 * confidence_damage
        + 0.10 * min(1.0, deception_lift),
        0.0,
        1.0,
    ))


def train_jammer_policy(
    out: str | Path,
    *,
    seed: int = 11,
    steps: int = 3600,
    epochs: int = 12,
    batch_size: int = 128,
    lr: float = 3e-4,
    hidden: int = 128,
    device: str = "cpu",
) -> JammerTrainingResult:
    torch, nn = require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs, triggers, spoof_tokens, target_masks = collect_jammer_dataset(seed=seed, steps=steps)
    model = JammerPolicy(hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    trigger_loss = nn.CrossEntropyLoss()
    spoof_loss = nn.CrossEntropyLoss()
    target_loss = nn.BCEWithLogitsLoss()
    x = torch.as_tensor(obs, dtype=torch.float32, device=device)
    y_trigger = torch.as_tensor(triggers, dtype=torch.long, device=device)
    y_spoof = torch.as_tensor(spoof_tokens, dtype=torch.long, device=device)
    y_targets = torch.as_tensor(target_masks, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    final_loss = 0.0
    trigger_accuracy = 0.0
    stats: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(obs))
        losses = []
        correct = 0
        total = 0
        for start in range(0, len(obs), batch_size):
            idx = torch.as_tensor(order[start:start + batch_size], dtype=torch.long, device=device)
            outp = model(x[idx])
            loss = (
                trigger_loss(outp["trigger_logits"], y_trigger[idx])
                + 0.55 * spoof_loss(outp["spoof_logits"].reshape(-1, len(VOCABULARY)), y_spoof[idx].reshape(-1))
                + 0.35 * target_loss(outp["target_logits"], y_targets[idx])
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            final_loss = float(loss.detach().cpu())
            losses.append(final_loss)
            pred = outp["trigger_logits"].argmax(dim=-1)
            correct += int((pred == y_trigger[idx]).sum().detach().cpu())
            total += int(y_trigger[idx].numel())
        trigger_accuracy = correct / max(1, total)
        stats.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else final_loss,
            "trigger_accuracy": float(trigger_accuracy),
            "positive_rate": float(np.mean(triggers)),
            "samples": int(len(obs)),
        })

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind": "evader_jammer_policy",
        "obs_dim": OBS_SIZE,
        "vocab_size": len(VOCABULARY),
        "hidden": hidden,
        "seed": seed,
        "steps": steps,
        "epochs": epochs,
        "state_dict": model.state_dict(),
        "final_loss": final_loss,
        "trigger_accuracy": trigger_accuracy,
        "positive_rate": float(np.mean(triggers)),
        "stats": stats,
    }, out)
    return JammerTrainingResult(
        checkpoint=str(out),
        samples=len(obs),
        epochs=epochs,
        final_loss=final_loss,
        trigger_accuracy=trigger_accuracy,
        positive_rate=float(np.mean(triggers)),
        stats=stats,
    )


def train_counterfactual_jammer_policy(
    out: str | Path,
    *,
    seed: int = 11,
    steps: int = 1200,
    sample_every: int = 12,
    horizon_steps: int = 72,
    positive_threshold: float = 0.16,
    epochs: int = 12,
    batch_size: int = 128,
    lr: float = 3e-4,
    hidden: int = 128,
    device: str = "cpu",
) -> JammerTrainingResult:
    """Train the jammer from paired counterfactual spoof-effect rollouts."""
    torch, nn = require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs, triggers, spoof_tokens, target_masks, sample_weights, rewards = collect_counterfactual_jammer_dataset(
        seed=seed,
        steps=steps,
        sample_every=sample_every,
        horizon_steps=horizon_steps,
        positive_threshold=positive_threshold,
    )
    model = JammerPolicy(hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    trigger_loss = nn.CrossEntropyLoss(reduction="none")
    spoof_loss = nn.CrossEntropyLoss(reduction="none")
    target_loss = nn.BCEWithLogitsLoss(reduction="none")
    x = torch.as_tensor(obs, dtype=torch.float32, device=device)
    y_trigger = torch.as_tensor(triggers, dtype=torch.long, device=device)
    y_spoof = torch.as_tensor(spoof_tokens, dtype=torch.long, device=device)
    y_targets = torch.as_tensor(target_masks, dtype=torch.float32, device=device)
    weights = torch.as_tensor(sample_weights, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    final_loss = 0.0
    trigger_accuracy = 0.0
    stats: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(obs))
        losses = []
        correct = 0
        total = 0
        weighted_reward = 0.0
        weight_total = 0.0
        for start in range(0, len(obs), batch_size):
            idx = torch.as_tensor(order[start:start + batch_size], dtype=torch.long, device=device)
            outp = model(x[idx])
            w = weights[idx]
            trig = trigger_loss(outp["trigger_logits"], y_trigger[idx])
            spoof = spoof_loss(
                outp["spoof_logits"].reshape(-1, len(VOCABULARY)),
                y_spoof[idx].reshape(-1),
            ).reshape(len(idx), SPOOF_LEN).mean(dim=1)
            target = target_loss(outp["target_logits"], y_targets[idx]).mean(dim=1)
            loss = ((trig + 0.55 * spoof + 0.35 * target) * w).sum() / torch.clamp(w.sum(), min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            final_loss = float(loss.detach().cpu())
            losses.append(final_loss)
            pred = outp["trigger_logits"].argmax(dim=-1)
            correct += int((pred == y_trigger[idx]).sum().detach().cpu())
            total += int(y_trigger[idx].numel())
            weighted_reward += float((w.detach().cpu().numpy() * rewards[order[start:start + batch_size]]).sum())
            weight_total += float(w.detach().cpu().numpy().sum())
        trigger_accuracy = correct / max(1, total)
        stats.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else final_loss,
            "trigger_accuracy": float(trigger_accuracy),
            "positive_rate": float(np.mean(triggers)),
            "mean_counterfactual_reward": float(np.mean(rewards)),
            "weighted_counterfactual_reward": weighted_reward / max(1.0, weight_total),
            "samples": int(len(obs)),
        })

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind": "evader_jammer_policy",
        "training_mode": "counterfactual",
        "obs_dim": OBS_SIZE,
        "vocab_size": len(VOCABULARY),
        "hidden": hidden,
        "seed": seed,
        "steps": steps,
        "sample_every": sample_every,
        "horizon_steps": horizon_steps,
        "positive_threshold": positive_threshold,
        "epochs": epochs,
        "state_dict": model.state_dict(),
        "final_loss": final_loss,
        "trigger_accuracy": trigger_accuracy,
        "positive_rate": float(np.mean(triggers)),
        "mean_counterfactual_reward": float(np.mean(rewards)),
        "stats": stats,
    }, out)
    return JammerTrainingResult(
        checkpoint=str(out),
        samples=len(obs),
        epochs=epochs,
        final_loss=final_loss,
        trigger_accuracy=trigger_accuracy,
        positive_rate=float(np.mean(triggers)),
        stats=stats,
        mean_counterfactual_reward=float(np.mean(rewards)),
        training_mode="counterfactual",
    )


def load_jammer_policy(path: str | Path):
    torch, _ = require_torch()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("kind") != "evader_jammer_policy":
        raise ValueError(f"unsupported jammer checkpoint kind: {ckpt.get('kind')!r}")
    model = JammerPolicy(
        obs_dim=int(ckpt["obs_dim"]),
        vocab_size=int(ckpt["vocab_size"]),
        hidden=int(ckpt["hidden"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


class EvaderJammerController:
    """Runtime that supplies learned jamming actions while scripted driving remains active."""

    def __init__(self, checkpoint: str | Path):
        self.model, self.metadata = load_jammer_policy(checkpoint)
        self.torch, _ = require_torch()

    def action(self, sim: HeistSim) -> dict[str, dict[str, np.ndarray | int]]:
        obs = build_observation(sim, "evader_0")
        with self.torch.no_grad():
            x = self.torch.as_tensor(obs[None, :], dtype=self.torch.float32)
            outp = self.model(x)
            trigger = int(outp["trigger_logits"].argmax(dim=-1).item())
            spoof = outp["spoof_logits"].argmax(dim=-1)[0].detach().cpu().numpy().astype(np.int64)
            target_scores = outp["target_logits"][0].detach().cpu().numpy().astype(np.float32)
        mask = np.zeros(TARGET_COUNT, dtype=np.int64)
        mask[np.argsort(target_scores)[-2:]] = 1
        return {
            "evader_0": {
                "jam": trigger,
                "spoof_tokens": spoof,
                "target_mask": mask,
            }
        }


def _spoof_payload(sim: HeistSim, victim_idx: int) -> np.ndarray:
    victim = sim.pursuers[victim_idx].vehicle
    ev = sim.evader.vehicle
    auth = AUTH_WORDS[sim.channel.cipher_epoch]
    east_west = "east" if victim.x >= ev.x else "west"
    north_south = "south" if victim.y >= ev.y else "north"
    words = [auth[0], "gate", east_west, north_south, "switch"]
    return np.asarray([WORD_TO_TOKEN[word] for word in words], dtype=np.int64)


def _pursuer_xy(sim: HeistSim) -> np.ndarray:
    return np.asarray([[p.vehicle.x, p.vehicle.y] for p in sim.pursuers], dtype=np.float32)


def _disable_future_jamming(sim: HeistSim) -> None:
    sim.jam_cooldown = 1.0e9
    sim.channel.jamming_budget = 0
