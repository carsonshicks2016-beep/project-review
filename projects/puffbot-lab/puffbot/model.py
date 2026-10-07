"""Policy and value networks, and checkpoint I/O."""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import actions, contract, observation

MASKED = -1e9


def _mlp(inp, hidden, out_dim, layers=3):
    mods, d = [], inp
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        d = hidden
    head = nn.Linear(d, out_dim)
    return nn.Sequential(*mods), head


class Policy(nn.Module):
    """Separate policy and value torsos over shared input embeddings."""

    def __init__(self, hidden: int = 512):
        super().__init__()
        self.hidden = hidden
        self.action_emb = nn.Embedding(observation.ACTION_VOCAB, 32)
        self.char_emb = nn.Embedding(observation.CHAR_VOCAB, 8)
        inp = observation.FLOATS + 2 * 32 + 2 * 8
        self.pi_body, self.pi_head = _mlp(inp, hidden, actions.COUNT)
        self.v_body, self.v_head = _mlp(inp, hidden, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # A near-uniform starting policy over the legal macros, and a quiet value head.
        nn.init.orthogonal_(self.pi_head.weight, gain=0.01)
        nn.init.orthogonal_(self.v_head.weight, gain=1.0)

    def features(self, floats, ids):
        return torch.cat([floats, self.action_emb(ids[..., 0]), self.action_emb(ids[..., 1]),
                          self.char_emb(ids[..., 2]), self.char_emb(ids[..., 3])], dim=-1)

    def forward(self, floats, ids, mask):
        x = self.features(floats, ids)
        logits = self.pi_head(self.pi_body(x)).masked_fill(~mask, MASKED)
        value = self.v_head(self.v_body(x)).squeeze(-1)
        return logits, value

    def logits(self, floats, ids, mask):
        x = self.features(floats, ids)
        return self.pi_head(self.pi_body(x)).masked_fill(~mask, MASKED)


class Actor:
    """CPU inference for one worker. Single-threaded on purpose: eight workers each
    grabbing every core is how a small model makes a machine thrash."""

    def __init__(self, hidden: int):
        self.net = Policy(hidden).eval()
        self.version = -1

    def load_state(self, state: dict, version: int):
        self.net.load_state_dict(state)
        self.version = version

    @torch.no_grad()
    def act(self, floats: np.ndarray, ids: np.ndarray, masks: np.ndarray, rng: np.random.Generator,
            greedy: bool = False):
        """Batched: arrays with a leading batch axis. Returns (actions, log-probs)."""
        logits = self.net.logits(torch.from_numpy(floats), torch.from_numpy(ids), torch.from_numpy(masks))
        logp = torch.log_softmax(logits, dim=-1).numpy().astype(np.float64)
        out_a = np.empty(len(logp), np.int64)
        out_l = np.empty(len(logp), np.float32)
        for i, row in enumerate(logp):
            if greedy:
                a = int(np.argmax(row))
            else:
                p = np.exp(row - row.max())
                p /= p.sum()
                a = int(rng.choice(len(p), p=p))
            out_a[i] = a
            out_l[i] = row[a]
        return out_a, out_l


# ---------------------------------------------------------------- checkpoint files
def save_policy(path: Path, net: Policy, version: int, frames: int, extra: dict | None = None,
                act_every: int | None = None):
    """Atomic write: workers poll this file and must never read a half-written one."""
    path = Path(path)
    payload = {'state': {k: v.detach().cpu() for k, v in net.state_dict().items()},
               'version': int(version), 'frames': int(frames), 'hidden': net.hidden,
               'contract': contract.current(act_every), **(extra or {})}
    buf = io.BytesIO()
    torch.save(payload, buf)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_bytes(buf.getvalue())
    os.replace(tmp, path)


def load_policy(path: Path, act_every: int | None = None) -> dict:
    """Load a policy file; raises IncompatibleCheckpoint, and puts any softer
    mismatches (decision timing, executor version) in payload['notes']."""
    from . import migrate
    payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    payload['notes'] = contract.check(payload.get('contract'), act_every, what=Path(path).name, migrate=True)
    if migrate.is_v1(payload.get('contract')):
        payload['state'] = migrate.v1_to_current(payload['state'])
        payload['migrated_from'] = payload.get('contract')
        payload['contract'] = {**contract.current(contract.upgrade(payload.get('contract'))['act_every']),
                               'executor': contract.upgrade(payload.get('contract'))['executor']}
    return payload
