#!/usr/bin/env python3
"""Regression gates for Fable Five's autonomous-training safeguards."""
from __future__ import annotations

import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import supra.fable5 as f5
from supra.ppo import ActorCritic, PPO, validate_checkpoint_payload
from supra.track import named_track


def gate(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f" — {detail}" if detail else ""))
    if not ok:
        raise SystemExit(f"gate failed: {name} {detail}")


def _hash_state(state: dict[str, torch.Tensor]) -> str:
    net = ActorCritic(4, 3, (8,), action_log_std_max=(0.0, 0.0, -1.0))
    net.load_state_dict(state)
    holder = type("HashHolder", (), {"net": net})()
    return PPO.policy_hash(holder)


def checkpoint_ordering() -> None:
    print("== 1. monotonic checkpoint ordering ==")
    ungated = {"stage": "foundation", "metric": 100.0,
               "recommendation": "keep training foundation"}
    gated = dict(ungated, recommendation="advance to flow")
    lap_slow = {"stage": "frontier", "metric": 1.0, "lap_time": 500.0}
    lap_fast = dict(lap_slow, lap_time=450.0)
    gate("gate proof outranks ungated trend", f5._checkpoint_score(gated)
         > f5._checkpoint_score(ungated))
    gate("any valid lap outranks a lapless fragment",
         f5._checkpoint_score(lap_slow) > f5._checkpoint_score(ungated))
    gate("faster valid lap ranks higher",
         f5._checkpoint_score(lap_fast) > f5._checkpoint_score(lap_slow))


def eval_transaction(tmp: Path) -> None:
    print("== 2. evaluated-policy checkpoint transaction ==")

    class Callback:
        evaluated_hash = None
        evaluated_state = None

        def __call__(self, ppo):
            self.evaluated_hash = ppo.policy_hash()
            self.evaluated_state = copy.deepcopy(ppo.net.state_dict())
            return {"stage": "fast", "metric": 10.0,
                    "checkpoint_score": 10.0, "log": "[eval-fable] probe"}

        def after_eval(self, ppo, result):
            with torch.no_grad():
                next(ppo.net.parameters()).add_(0.25)
            return {"policy_changed": True, "decision": "rollback"}

    class Holder:
        def __init__(self):
            self.net = ActorCritic(4, 3, (8,))
            self.eval_callback = Callback()
            self.best_metric = -1e9
            self._lr_safety_scale = 1.0

        def policy_hash(self):
            return PPO.policy_hash(self)

        def save(self, path):
            torch.save({"state_dict": copy.deepcopy(self.net.state_dict()),
                        "policy_sha256": self.policy_hash()}, path)

    ppo = Holder()
    latest, best = tmp / "txn.pt", tmp / "txn_best.pt"
    improved = PPO.eval_and_save(ppo, str(latest), str(best))
    best_payload = torch.load(best, map_location="cpu", weights_only=False)
    latest_payload = torch.load(latest, map_location="cpu", weights_only=False)
    gate("baseline evaluation banks a best", improved)
    gate("best is exactly the policy that was evaluated",
         best_payload["policy_sha256"] == ppo.eval_callback.evaluated_hash
         and all(torch.equal(best_payload["state_dict"][k], v)
                 for k, v in ppo.eval_callback.evaluated_state.items()))
    gate("latest persists the post-eval recovery separately",
         latest_payload["policy_sha256"] != ppo.eval_callback.evaluated_hash)


def normalizer_and_metadata(tmp: Path) -> None:
    print("== 3. raw normalizer transaction + checkpoint metadata ==")
    trk = f5.attach_envelope(named_track(f5.RING_TRACK))
    sp = f5.stage_defaults("foundation")
    cfg = f5._configure_ppo(sp, workers=1, pop=2)
    cfg.rollout = 4
    cfg.epochs = 0
    cfg.minibatches = 1
    cfg.target_kl = None
    evaluator = f5.FableEvaluator(sp, trk, tmp / "norm_manifest.json")
    ppo = PPO(mode="race", car=f5.FABLE_CAR, ppo=cfg, reward=sp.reward,
              fixed_track=trk, track_name=f5.RING_TRACK,
              env_cls_override=f5.FableEnv,
              env_kwargs={"fable_spec": sp}, eval_callback=evaluator,
              extra_metadata=evaluator.metadata)
    try:
        obs = ppo.vec.reset()
        mean0, var0, count0 = (ppo.norm.mean.copy(), ppo.norm.var.copy(),
                               float(ppo.norm.count))
        last, batch = ppo.collect(obs)
        raw = ppo._pending_norm_obs.copy()
        adv, ret = ppo.gae(batch[3], batch[4], batch[5], last)
        gate("collect and GAE keep one immutable normalizer snapshot",
             np.array_equal(ppo.norm.mean, mean0)
             and np.array_equal(ppo.norm.var, var0)
             and ppo.norm.count == count0)
        ppo.update(batch, adv, ret)
        expected = (mean0 * count0 + raw.sum(axis=0)) / (count0 + len(raw))
        gate("accepted update commits RAW sensor samples",
             np.allclose(ppo.norm.mean, expected, atol=1e-10)
             and abs(ppo.norm.count - (count0 + len(raw))) < 1e-9)

        frozen = (ppo.norm.mean.copy(), ppo.norm.var.copy(), ppo.norm.count)
        ppo.norm_update_mode = "legacy-frozen-v1"
        ppo._pending_norm_obs = np.full((8, ppo.sdim), 999.0)
        ppo.update(batch, adv, ret)
        gate("legacy coordinate system stays frozen",
             np.array_equal(ppo.norm.mean, frozen[0])
             and np.array_equal(ppo.norm.var, frozen[1])
             and ppo.norm.count == frozen[2])

        ppo.norm_update_mode = "raw-v2"
        cp = tmp / "metadata.pt"
        ppo.save(str(cp))
        meta = torch.load(cp, map_location="cpu", weights_only=False)
        validate_checkpoint_payload(meta, require_hash=True)
        gate("checkpoint binds policy, protocol, config, and safety state",
             meta.get("policy_sha256")
             and meta.get("fable_eval_protocol") == f5.EVAL_PROTOCOL_VERSION
             and meta.get("fable_code_fingerprint") == f5.CODE_FINGERPRINT
             and isinstance(meta.get("ppo_config"), dict)
             and "kl_rejections" in meta and "runtime_versions" in meta)

        corrupt = dict(meta, norm_var=np.full(ppo.sdim, np.nan))
        try:
            validate_checkpoint_payload(corrupt, require_hash=True)
            rejected = False
        except ValueError:
            rejected = True
        gate("non-finite normalizer is rejected", rejected)
    finally:
        ppo.vec.close()


def kl_transaction() -> None:
    print("== 4. transactional KL rejection ==")
    torch.manual_seed(7)
    np.random.seed(7)
    obs_dim, act_dim, t, nenv = 6, 3, 8, 4
    net = ActorCritic(obs_dim, act_dim, (16,))
    fake = type("PPOProbe", (), {})()
    fake.cfg = type("Cfg", (), {"minibatches": 2, "epochs": 3,
                                 "clip": 0.2, "vf_coef": 0.5,
                                 "max_grad_norm": 0.5,
                                 "target_kl": 1e-12, "vf_clip": 5.0})()
    fake.net = net
    fake.opt = torch.optim.Adam(net.parameters(), lr=0.03)
    fake._ent_coef = 0.003
    fake.obs_dim, fake.act_dim = obs_dim, act_dim
    fake._t = lambda x: torch.as_tensor(x, dtype=torch.float32)
    fake._pending_norm_obs = None
    fake._kl_rejections = 0
    fake._lr_safety_scale = 1.0
    b_obs = np.random.randn(t, nenv, obs_dim).astype(np.float32)
    b_act = np.random.randn(t, nenv, act_dim).astype(np.float32)
    with torch.no_grad():
        old_logp, _, values, _, _, _ = net.evaluate(
            torch.as_tensor(b_obs.reshape(-1, obs_dim)),
            torch.as_tensor(b_act.reshape(-1, act_dim)))
    b_logp = old_logp.reshape(t, nenv).numpy().astype(np.float32)
    b_val = values.reshape(t, nenv).numpy().astype(np.float32)
    adv = np.random.randn(t, nenv).astype(np.float32)
    ret = (b_val + np.random.randn(t, nenv)).astype(np.float32)
    batch = (b_obs, b_act, b_logp, b_val,
             np.zeros((t, nenv), np.float32), np.zeros((t, nenv), np.float32))
    before = copy.deepcopy(net.state_dict())
    st = PPO.update(fake, batch, adv, ret)
    gate("destructive overshoot is rejected",
         st["kl_reject"] and not st["accepted"] and fake._kl_rejections == 1,
         f"kl={st['kl']:.3e}")
    gate("policy and Adam transaction roll back",
         all(torch.equal(before[k], net.state_dict()[k]) for k in before)
         and not fake.opt.state_dict()["state"])
    gate("rejection backs off the safety LR",
         abs(fake._lr_safety_scale - 0.8) < 1e-12
         and abs(fake.opt.param_groups[0]["lr"] - 0.024) < 1e-12)


def tuning_and_footprint() -> None:
    print("== 5. stage tuning + whole-body certification ==")
    cfgs = [f5._configure_ppo(f5.stage_defaults(st), workers=64, pop=64)
            for st in f5.STAGES]
    gear_caps = [cfg.action_log_std_max[2] for cfg in cfgs]
    gate("gear exploration tightens monotonically",
         all(b < a for a, b in zip(gear_caps, gear_caps[1:])), str(gear_caps))
    gate("64 envs are sharded within host capacity",
         all(cfg.n_workers < cfg.n_envs and cfg.n_envs % cfg.n_workers == 0
             for cfg in cfgs), f"workers={cfgs[0].n_workers}")
    gate("eval cadence scales with population",
         f5._eval_every_for(f5.stage_defaults("fast"), 64)
         < f5._eval_every_for(f5.stage_defaults("fast"), 16))

    net = ActorCritic(4, 3, (8,), action_log_std_max=(0.0, 0.0, -1.2))
    with torch.no_grad():
        net.log_std.fill_(5.0)
        net.clamp_log_std_()
    gate("raw gear log-std cannot escape its cap",
         float(net.log_std[2].detach()) <= -1.2 + 1e-7)

    trk = f5.attach_envelope(named_track(f5.RING_TRACK))
    sp = f5.stage_defaults("foundation")
    cfg = f5._configure_ppo(sp, workers=1, pop=1)
    env = f5.FableEnv(mode="race", car=f5.FABLE_CAR, ppo=cfg,
                      fixed_track=trk, fable_spec=sp, rng_seed=9,
                      diagnostics=True)
    idx = len(trk.center) // 5
    env.reset_at(idx, speed=0.0)
    pos = trk.center[idx] + trk.normal[idx] * (trk.half_width[idx] - 0.05)
    yaw = float(np.arctan2(trk.tangent[idx, 1], trk.tangent[idx, 0]))
    env.veh.reset(float(pos[0]), float(pos[1]), yaw, speed=0.0)
    fr = trk.frame(env.veh.x, env.veh.y)
    env.prev_frac = fr["progress"]
    corners_off = any(trk.frame(float(x), float(y))["off_track"]
                      for x, y in env.veh.get_obb())
    _, _, _, _, info = env.step(np.zeros(3))
    gate("centre can be legal while body crosses boundary",
         not fr["off_track"] and corners_off)
    gate("eval invalidates a footprint-off run",
         info["fable_invalid"] and not info["fable_footprint_valid"]
         and info["fable_footprint_offtrack_seconds"] > 0.0)


def baseline_and_protocol(tmp: Path) -> None:
    print("== 6. baseline gating + protocol recertification ==")
    original = f5.FableEvaluator.evaluate

    def instant_gate(self, ppo):
        return {"stage": self.spec.stage, "metric": 40.0,
                "clean_sectors": 16, "sector_count": 16, "clean_chain": 16,
                "terminal_rate": 0.0, "pace_ratio": 0.8,
                "max_progress_m": self.track.length, "progress_frac": 1.0,
                "distance_laps": 1.0, "offtrack_seconds": 0.0,
                "lap_time": None, "recommendation": "advance to flow",
                "laps": 1.0, "drift": 0.0, "fail_sectors": [],
                "worst_sectors": [], "termination_counts": {},
                "lap_style": "flying", "latest_trace": [],
                "log": "[eval-fable] instant baseline",
                "best_tag": "[best-fable/foundation]"}

    old_cwd = Path.cwd()
    try:
        os.chdir(tmp)
        f5.FableEvaluator.evaluate = instant_gate
        result = f5._train_one_stage(
            "foundation", 3, out=str(tmp / "baseline.pt"), workers=1, pop=2,
            promote_best=False, stop_on_gate=True)
    finally:
        f5.FableEvaluator.evaluate = original
        os.chdir(old_cwd)
    gate("incoming policy gates at update zero", result.get("iters_used") == 0)
    best = tmp / "baseline_best.pt"
    gate("baseline is banked under the current protocol",
         f5._checkpoint_protocol_current(str(best), "foundation"))
    d = torch.load(best, map_location="cpu", weights_only=False)
    mismatched = dict(d, fable_stage="flow")
    mismatch_path = tmp / "mismatch.pt"
    torch.save(mismatched, mismatch_path)
    stale = dict(d, fable_code_fingerprint="stale")
    stale_path = tmp / "stale.pt"
    torch.save(stale, stale_path)
    gate("stage mismatch cannot gate-skip",
         not f5._checkpoint_protocol_current(str(mismatch_path), "flow"))
    gate("code-fingerprint drift forces recertification",
         not f5._checkpoint_protocol_current(str(stale_path), "foundation"))


def supervisor_and_lock(tmp: Path) -> None:
    print("== 7. watchdog completion, process-group cleanup, and lock ==")
    supervisor = ROOT / "tools" / "supervise_fable5.py"
    logpath = tmp / "probe.logpath"
    probe = subprocess.run(
        [sys.executable, str(supervisor), "--name", "_safeguard_complete",
         "--max-restarts", "0", "--rerun-completed",
         "--logpath-file", str(logpath), "--",
         sys.executable, "-c", "print('[fable5] AUTO ladder done')"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=20)
    state = json.loads((ROOT / "runtime/fable5/_safeguard_complete/"
                        "supervisor_state.json").read_text())
    gate("watchdog recognizes a finite successful run",
         probe.returncode == 0 and state.get("status") == "complete")

    grandchild_pid = tmp / "grandchild.pid"
    child_code = (
        "import os,pathlib,signal,subprocess,time;"
        "signal.signal(signal.SIGINT, signal.SIG_IGN);"
        "p=subprocess.Popen(['/bin/sleep','300']);"
        f"pathlib.Path({str(grandchild_pid)!r}).write_text(str(p.pid));"
        "time.sleep(300)"
    )
    proc = subprocess.Popen(
        [sys.executable, str(supervisor), "--name", "_safeguard_stop",
         "--max-restarts", "0", "--stop-grace-seconds", "0.5",
         "--logpath-file", str(tmp / "stop.logpath"), "--",
         sys.executable, "-c", child_code], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 10.0
    while not grandchild_pid.exists() and time.time() < deadline:
        time.sleep(0.05)
    gate("synthetic worker tree launched", grandchild_pid.exists())
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=10)
    gpid = int(grandchild_pid.read_text())
    dead = False
    for _ in range(40):
        try:
            os.kill(gpid, 0)
        except ProcessLookupError:
            dead = True
            break
        time.sleep(0.05)
    gate("forced stop cleans the entire trainer process group",
         proc.returncode == 0 and dead)

    lock_path = ROOT / ".fable5_training.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.seek(0); lock.truncate(); lock.write("safeguard validator"); lock.flush()
        duplicate = subprocess.run(
            [sys.executable, "run.py", "--fable", "1", "--fable-stage",
             "foundation", "--workers", "1", "--pop", "1",
             "--out", "_duplicate_probe.pt"], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
    gate("second trainer is rejected without retry ambiguity",
         duplicate.returncode == 73 and "already owns" in duplicate.stdout)


def main() -> None:
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="fable5_safeguards_") as td:
        tmp = Path(td)
        checkpoint_ordering()
        eval_transaction(tmp)
        normalizer_and_metadata(tmp)
        kl_transaction()
        tuning_and_footprint()
        baseline_and_protocol(tmp)
        supervisor_and_lock(tmp)
    print(f"ALL AUTONOMOUS SAFEGUARD GATES PASSED in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
