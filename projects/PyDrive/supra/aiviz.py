"""
AI-brain visualisation for the watch view (cosmetic — never touches training).

PolicyAgent wraps a trained actor-critic so the renderer can show what the brain
is doing: the action it picks, the critic's value estimate (confidence), how
unsure it is (policy std), its last hidden-layer activations (the "neurons"),
and a short forward roll-out of where it THINKS it's going (predicted path).
"""
from __future__ import annotations

import copy
from collections import deque

import numpy as np
import torch


class PolicyAgent:
    def __init__(self, net, norm, meta, mode="drift"):
        self.net = net
        self.norm = norm
        self.sdim = int(meta["sdim"])
        self.mode = mode
        self.mode_vec = (np.array([1.0, 1.0], dtype=np.float32) if mode == "hybrid"
                         else np.array([1.0, 0.0], dtype=np.float32) if mode == "race"
                         else np.array([0.0, 1.0], dtype=np.float32))
        self.act_dim = net.mean.out_features
        from .sensors import SensorSuite
        self.sensors = SensorSuite()
        self.vhist = deque(maxlen=400)      # recent values -> adaptive confidence
        self.last = {"value": 0.0, "std": np.full(self.act_dim, 0.3),
                     "act": np.zeros(self.act_dim), "h": np.zeros(8)}
        self.path = []                      # last completed predicted path
        self._roll = None                   # in-progress amortised rollout

    # -- observation packing ---------------------------------------------- #
    def _nobs(self, obs_vector):
        # Pad with zeros if environment provided fewer sensors than the network expects
        if len(obs_vector) < self.sdim:
            obs_vector = np.concatenate([obs_vector, np.zeros(self.sdim - len(obs_vector))])
        # Slice to network size if environment provided more sensors
        elif len(obs_vector) > self.sdim:
            obs_vector = obs_vector[:self.sdim]
            
        full = np.concatenate([obs_vector, self.mode_vec]).astype(np.float32)
        s = self.norm.normalize(full[:self.sdim])
        return np.concatenate([s, full[self.sdim:]]).astype(np.float32)

    def _to_controls(self, a):
        steer = float(np.clip(a[0], -1, 1))
        lo = float(np.clip(a[1], -1, 1))
        hb = float(np.clip((a[2] + 1) / 2, 0, 1)) if a.shape[0] > 2 else 0.0
        return (steer, max(lo, 0.0), max(-lo, 0.0), hb)

    # -- control + telemetry (called at the policy's 30 Hz) --------------- #
    def act(self, veh, obs):
        nobs = self._nobs(obs.vector)
        with torch.no_grad():
            t = torch.as_tensor(nobs).unsqueeze(0)
            h = self.net.trunk(t)
            mean = self.net.mean(h).squeeze(0).numpy()
            value = float(self.net.value(h).squeeze(-1).item())
            std = self.net.log_std.clamp(
                self.net.LOG_STD_MIN, self.net.LOG_STD_MAX).exp().numpy()
        self.vhist.append(value)
        self.last = {"value": value, "std": std.copy(), "act": mean.copy(),
                     "h": h.squeeze(0).numpy().copy()}
        return self._to_controls(mean)

    # -- derived signals for the viz -------------------------------------- #
    def confidence(self):
        """0..1 from the critic's value, normalised to its recent range."""
        if len(self.vhist) < 8:
            return 0.5
        v = np.fromiter(self.vhist, float)
        lo, hi = np.percentile(v, 5), np.percentile(v, 95)
        if hi - lo < 1e-6:
            return 0.5
        return float(np.clip((self.last["value"] - lo) / (hi - lo), 0.0, 1.0))

    def uncertainty(self):
        """0..1 from the policy std (log_std clamp range -> [0.11, 1.0])."""
        s = float(np.mean(self.last["std"]))
        return float(np.clip((s - 0.11) / (1.0 - 0.11), 0.0, 1.0))

    # -- forward roll-out: where it thinks it's going --------------------- #
    def predict_path(self, veh, trk, seconds=1.1):
        from .config import SimSpec
        from .physics import Controls
        from .app import AutoBox
        v = copy.deepcopy(veh)
        dt = v.sim.dt if getattr(v, "sim", None) else SimSpec().dt
        box = AutoBox(v.spec)
        cper = max(1, round(1.0 / (30 * dt)))
        n = int(seconds / dt)
        path = [(v.x, v.y)]
        ctl = (0.0, 0.0, 0.0, 0.0)
        for i in range(n):
            if i % cper == 0:
                o = self.sensors.observe(v, trk)
                with torch.no_grad():
                    a = self.net.act_mean(
                        torch.as_tensor(self._nobs(o.vector)).unsqueeze(0)
                    ).squeeze(0).numpy()
                ctl = self._to_controls(a)
            s_in, t_in, b_in, hb = ctl
            clutch, up, down = box.update(v, t_in, dt)
            fr = trk.frame(v.x, v.y)
            v.surface_grip = v.spec.offtrack_grip if fr["off_track"] else 1.0
            v.set_road(fr["grade"], fr["bank"], fr["heading"], fr["z"],
                       fr["vcurv"])
            v.step(Controls(steer=s_in, throttle=t_in, brake=b_in, clutch=clutch,
                            handbrake=hb, shift_up=up, shift_down=down))
            if i % 3 == 0:
                path.append((v.x, v.y))
            if abs(fr["lateral"]) > trk.half + 14:   # predicted spin-off: stop
                break
        return path

    # -- amortised roll-out: spread the work over frames so no single frame
    # spikes (the full rollout was ~30 ms -> visible stutter). Each call advances
    # an in-progress rollout by `budget` steps; the last COMPLETED path is what's
    # drawn, so it stays stable between refreshes (~8x/sec). World-anchored, so it
    # correctly shows the line the car drives along.
    def step_prediction(self, veh, trk, budget=16, seconds=0.95):
        from .config import SimSpec
        from .physics import Controls
        from .app import AutoBox
        R = self._roll
        if R is None:
            v = copy.deepcopy(veh)
            dt = v.sim.dt if getattr(v, "sim", None) else SimSpec().dt
            R = self._roll = {"v": v, "box": AutoBox(v.spec), "dt": dt,
                              "cper": max(1, round(1.0 / (30 * dt))),
                              "n": int(seconds / dt), "i": 0,
                              "path": [(v.x, v.y)], "ctl": (0.0, 0.0, 0.0, 0.0)}
        v, dt = R["v"], R["dt"]
        for _ in range(budget):
            if R["i"] >= R["n"]:
                break
            if R["i"] % R["cper"] == 0:
                o = self.sensors.observe(v, trk)
                with torch.no_grad():
                    a = self.net.act_mean(
                        torch.as_tensor(self._nobs(o.vector)).unsqueeze(0)
                    ).squeeze(0).numpy()
                R["ctl"] = self._to_controls(a)
            s_in, t_in, b_in, hb = R["ctl"]
            clutch, up, down = R["box"].update(v, t_in, dt)
            fr = trk.frame(v.x, v.y)
            v.surface_grip = v.spec.offtrack_grip if fr["off_track"] else 1.0
            v.set_road(fr["grade"], fr["bank"], fr["heading"], fr["z"],
                       fr["vcurv"])
            v.step(Controls(steer=s_in, throttle=t_in, brake=b_in, clutch=clutch,
                            handbrake=hb, shift_up=up, shift_down=down))
            if R["i"] % 3 == 0:
                R["path"].append((v.x, v.y))
            R["i"] += 1
            if abs(fr["lateral"]) > trk.half + 14:
                R["i"] = R["n"]                # predicted spin-off: finish early
                break
        if R["i"] >= R["n"]:
            self.path = R["path"]
            self._roll = None                  # done -> publish, restart next call
        return self.path


# --------------------------------------------------------------------------- #
# drawing helpers
# --------------------------------------------------------------------------- #
def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def draw_path(pygame, screen, to_screen, path, conf):
    """Dotted ghost-line of the predicted trajectory, fading with distance."""
    col = _lerp((90, 200, 255), (255, 210, 90), conf)
    n = len(path)
    for i, (x, y) in enumerate(path):
        f = i / max(1, n - 1)
        r = max(1, int(4 * (1 - f) + 1))
        p = to_screen(x, y)
        dot = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
        a = int(200 * (1 - f) + 30)
        pygame.draw.circle(dot, (*col, a), (r + 1, r + 1), r)
        screen.blit(dot, (p[0] - r - 1, p[1] - r - 1))
