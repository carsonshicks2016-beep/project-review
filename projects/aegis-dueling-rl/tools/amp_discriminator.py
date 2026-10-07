"""
Aegis M4 — AMP discriminator (research scaffold).

Trains a discriminator D(s, s') to tell EXPERT motion transitions (from the reference clips)
apart from POLICY transitions (from the agent), then exports it to ONNX so Unity can run it
in-engine (Sentis) and add a STYLE reward:  r_style = -log(1 - D)  (AMP, Peng et al. 2021).

This is the experimental milestone — ML-Agents has no built-in AMP. The loop is:
  1) record motion_expert.jsonl from the reference clone   (MotionRecorder, label="expert")
  2) record motion_policy.jsonl from the agent periodically (MotionRecorder, label="policy")
  3) python tools/amp_discriminator.py --expert motion_expert.jsonl --policy motion_policy.jsonl
  4) it writes amp_discriminator.onnx → load in Unity, add r_style in DuelAgent
  5) repeat 2-4 as the policy improves (the discriminator must keep up)

Requires: torch. (pip install torch)
"""
import argparse, json
import torch
import torch.nn as nn


def load_jsonl(path):
    s, s2 = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            s.append(o["s"]); s2.append(o["s2"])
    x = torch.tensor([a + b for a, b in zip(s, s2)], dtype=torch.float32)
    return x


class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)  # logits; apply sigmoid for probability


def train(expert, policy, epochs, lr, grad_pen, out):
    in_dim = expert.shape[1]
    D = Discriminator(in_dim)
    opt = torch.optim.Adam(D.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        # balanced minibatch
        ie = torch.randint(0, expert.shape[0], (256,))
        ip = torch.randint(0, policy.shape[0], (256,))
        xe, xp = expert[ie], policy[ip]

        le = D(xe)                      # expert -> 1
        lp = D(xp)                      # policy -> 0
        loss = bce(le, torch.ones_like(le)) + bce(lp, torch.zeros_like(lp))

        # gradient penalty on expert (AMP stabilizer)
        if grad_pen > 0:
            xe.requires_grad_(True)
            g = torch.autograd.grad(D(xe).sum(), xe, create_graph=True)[0]
            loss = loss + grad_pen * (g.pow(2).sum(dim=1)).mean()

        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0:
            acc_e = (torch.sigmoid(le) > 0.5).float().mean().item()
            acc_p = (torch.sigmoid(lp) < 0.5).float().mean().item()
            print(f"ep {ep:4d}  loss {loss.item():.4f}  acc_expert {acc_e:.2f}  acc_policy {acc_p:.2f}")

    dummy = torch.zeros(1, in_dim)
    torch.onnx.export(D, dummy, out, input_names=["features"], output_names=["logit"],
                      dynamic_axes={"features": {0: "batch"}})
    print(f"exported {out}  (in_dim={in_dim})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", default="motion_expert.jsonl")
    ap.add_argument("--policy", default="motion_policy.jsonl")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-pen", type=float, default=10.0)
    ap.add_argument("--out", default="amp_discriminator.onnx")
    a = ap.parse_args()
    train(load_jsonl(a.expert), load_jsonl(a.policy), a.epochs, a.lr, a.grad_pen, a.out)
