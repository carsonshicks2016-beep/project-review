"""Evader scanner decoder training and runtime utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .comms import VOCABULARY, WORD_TO_TOKEN
from .imitation import require_torch
from .observations import MAX_CITY_EXTENT
from .sim import HeistSim

PAD_TOKEN = len(VOCABULARY)
DEFAULT_WINDOW = 24
DEFAULT_HORIZON_STEPS = 72


class ScannerDecoder(require_torch()[1].Module):
    """GRU decoder from radio-token history to future pursuer coordinates."""

    def __init__(self, vocab_size: int, window: int, embed: int = 32, hidden: int = 128):
        torch, nn = require_torch()
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.window = int(window)
        self.embed = int(embed)
        self.hidden = int(hidden)
        self.embedding = nn.Embedding(vocab_size, embed, padding_idx=PAD_TOKEN)
        self.gru = nn.GRU(embed, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 10),
        )

    def forward(self, tokens):
        embedded = self.embedding(tokens)
        seq, _ = self.gru(embedded)
        out = self.head(seq[:, -1])
        return out.reshape(-1, 5, 2)


@dataclass
class ScannerTrainingResult:
    checkpoint: str
    samples: int
    epochs: int
    final_loss: float
    horizon_steps: int
    window: int
    stats: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "samples": self.samples,
            "epochs": self.epochs,
            "final_loss": self.final_loss,
            "horizon_steps": self.horizon_steps,
            "window": self.window,
            "stats": self.stats,
        }


def token_window(sim: HeistSim, window: int = DEFAULT_WINDOW) -> np.ndarray:
    ids = [WORD_TO_TOKEN.get(event.word, PAD_TOKEN) for event in sim.channel.recent(window)]
    if len(ids) < window:
        ids = [PAD_TOKEN] * (window - len(ids)) + ids
    return np.asarray(ids[-window:], dtype=np.int64)


def pursuer_xy(sim: HeistSim) -> np.ndarray:
    return np.asarray([[p.vehicle.x, p.vehicle.y] for p in sim.pursuers], dtype=np.float32)


def collect_scanner_dataset(
    seed: int = 11,
    steps: int = 2400,
    horizon_steps: int = DEFAULT_HORIZON_STEPS,
    window: int = DEFAULT_WINDOW,
    sample_every: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect token histories and future pursuer positions from scripted rollouts."""
    sim = HeistSim(seed=seed, reset_on_capture=True)
    total = int(steps + horizon_steps + 1)
    token_rows: list[np.ndarray] = []
    xy_rows: list[np.ndarray] = []
    for t in range(total):
        token_rows.append(token_window(sim, window))
        xy_rows.append(pursuer_xy(sim) / MAX_CITY_EXTENT)
        sim.step()
    x = []
    y = []
    for t in range(0, int(steps), int(sample_every)):
        x.append(token_rows[t])
        y.append(xy_rows[t + horizon_steps])
    return np.stack(x).astype(np.int64), np.stack(y).astype(np.float32)


def train_scanner_decoder(
    out: str | Path,
    *,
    seed: int = 11,
    steps: int = 2400,
    horizon_steps: int = DEFAULT_HORIZON_STEPS,
    window: int = DEFAULT_WINDOW,
    sample_every: int = 4,
    epochs: int = 8,
    batch_size: int = 256,
    lr: float = 3e-4,
    embed: int = 32,
    hidden: int = 128,
    device: str = "cpu",
) -> ScannerTrainingResult:
    torch, nn = require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    tokens, targets = collect_scanner_dataset(
        seed=seed,
        steps=steps,
        horizon_steps=horizon_steps,
        window=window,
        sample_every=sample_every,
    )
    model = ScannerDecoder(len(VOCABULARY) + 1, window=window, embed=embed, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    x = torch.as_tensor(tokens, dtype=torch.long, device=device)
    y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    final_loss = 0.0
    stats: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(tokens))
        epoch_losses = []
        for start in range(0, len(tokens), batch_size):
            idx = torch.as_tensor(order[start:start + batch_size], dtype=torch.long, device=device)
            pred = model(x[idx])
            loss = loss_fn(pred, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            value = float(loss.detach().cpu())
            final_loss = value
            epoch_losses.append(value)
        stats.append({
            "epoch": epoch,
            "loss": float(np.mean(epoch_losses)) if epoch_losses else final_loss,
            "samples": int(len(tokens)),
        })

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind": "evader_scanner_decoder",
        "vocab_size": len(VOCABULARY) + 1,
        "window": window,
        "horizon_steps": horizon_steps,
        "embed": embed,
        "hidden": hidden,
        "seed": seed,
        "steps": steps,
        "sample_every": sample_every,
        "max_city_extent": MAX_CITY_EXTENT,
        "state_dict": model.state_dict(),
        "final_loss": final_loss,
        "stats": stats,
    }, out)
    return ScannerTrainingResult(
        checkpoint=str(out),
        samples=len(tokens),
        epochs=epochs,
        final_loss=final_loss,
        horizon_steps=horizon_steps,
        window=window,
        stats=stats,
    )


def load_scanner_decoder(path: str | Path):
    torch, _ = require_torch()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("kind") != "evader_scanner_decoder":
        raise ValueError(f"unsupported scanner checkpoint kind: {ckpt.get('kind')!r}")
    model = ScannerDecoder(
        vocab_size=int(ckpt["vocab_size"]),
        window=int(ckpt["window"]),
        embed=int(ckpt["embed"]),
        hidden=int(ckpt["hidden"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


class ScannerDecoderRuntime:
    """Applies a trained scanner to a live sim and updates spectator predictions."""

    def __init__(self, checkpoint: str | Path):
        self.model, self.metadata = load_scanner_decoder(checkpoint)
        self.torch, _ = require_torch()
        self.window = int(self.metadata["window"])
        self.horizon_steps = int(self.metadata["horizon_steps"])
        self.step_index = 0
        self.pending: list[tuple[int, np.ndarray]] = []
        self.confidence = 0.35
        self.seen_radio_events: set[tuple[float, str, str, bool, str]] = set()

    def predict_xy(self, sim: HeistSim) -> np.ndarray:
        tokens = token_window(sim, self.window)
        with self.torch.no_grad():
            x = self.torch.as_tensor(tokens[None, :], dtype=self.torch.long)
            pred = self.model(x)[0].detach().cpu().numpy().astype(np.float32)
        return pred * MAX_CITY_EXTENT

    def apply(self, sim: HeistSim) -> dict[str, tuple[float, float]]:
        pred_xy = self.predict_xy(sim)
        predictions = {f"P{i + 1}": (float(x), float(y)) for i, (x, y) in enumerate(pred_xy)}
        sim.decoder_predictions = predictions
        self.pending.append((self.step_index + self.horizon_steps, pred_xy.copy()))
        self._react_to_radio_disruptions(sim)
        self._settle_confidence(sim)
        sim.channel.confidence = float(self.confidence)
        self.step_index += 1
        return predictions

    def _react_to_radio_disruptions(self, sim: HeistSim):
        spoofed = 0
        cipher_rotations = 0
        for event in sim.channel.events:
            key = (round(float(event.time), 6), event.speaker, event.word, bool(event.spoofed), event.meaning)
            if key in self.seen_radio_events:
                continue
            self.seen_radio_events.add(key)
            spoofed += int(event.spoofed)
            cipher_rotations += int(event.speaker == "dispatch" and event.word == "cipher")
        if len(self.seen_radio_events) > 1024:
            self.seen_radio_events = set(list(self.seen_radio_events)[-512:])
        if spoofed:
            self.confidence = float(np.clip(self.confidence * 0.56 - 0.02, 0.04, 0.98))
        if cipher_rotations:
            self.confidence = float(np.clip(self.confidence * 0.48, 0.04, 0.98))

    def _settle_confidence(self, sim: HeistSim):
        actual = pursuer_xy(sim)
        live = []
        realized = []
        for due, pred in self.pending:
            if due <= self.step_index:
                realized.append(float(np.linalg.norm(pred - actual, axis=1).mean()))
            else:
                live.append((due, pred))
        self.pending = live[-240:]
        if not realized:
            return
        err = float(np.mean(realized))
        instant = float(np.clip(np.exp(-err / 120.0), 0.04, 0.98))
        self.confidence = float(np.clip(self.confidence * 0.84 + instant * 0.16, 0.04, 0.98))
