"""Train a PPO specialist for one Super Mario World level.

    python -m smwrl.train --level YoshiIsland1 --steps 5_000_000

Progress is written to checkpoints/<level>/ and TensorBoard logs to runs/.
Training is resumable: pass --resume to continue from the latest checkpoint.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from smwrl.env import make_vec_env
from smwrl.curriculum_store import load_verified_curriculum
from smwrl.levels import LEVELS
from smwrl.policy import load_policy_runtime, publish_policy
from smwrl.wrappers import CheckpointPool, EpisodeConfig, ObsConfig, RewardConfig

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"
RUNS = ROOT / "runs"


class LivePublish(BaseCallback):
    """Publish the in-progress policy so `smwrl.live` can watch training.

    Both files are written to a temp path and os.replace()d into place: the
    spectator polls these while they are being rewritten, and a partially
    written zip would crash it.
    """

    def __init__(self, outdir: Path, report: "ProgressReport", every: int,
                 obs_config: ObsConfig, positions: list | None = None,
                 frame_stack: int = 4):
        super().__init__()
        self.outdir = outdir
        self.report = report
        self.every = every
        self._next = every
        self.obs_config = obs_config
        self.frame_stack = frame_stack
        self.positions = positions or []

    def _on_training_start(self) -> None:
        # On resume num_timesteps may already be in the tens of millions.  The
        # old `_next = every` then saved a ~23 MB model on every callback until
        # it caught up one interval at a time (hundreds of redundant writes).
        self._next = (self.num_timesteps // self.every + 1) * self.every

    def _publish(self, model=None, timesteps: int | None = None) -> None:
        selected_model = self.model if model is None else model
        selected_steps = self.num_timesteps if timesteps is None else timesteps
        publish_policy(selected_model, self.outdir / "latest.zip", self.obs_config,
                       self.frame_stack, self.report.level)

        status = {
            "level": self.report.level,
            "timesteps": int(selected_steps),
            "mean_x": float(np.mean(self.report._recent_x)) if self.report._recent_x else 0.0,
            "best_x": int(self.report.best_x),
            "clear_rate": (
                float(np.mean(self.report._recent_clear)) if self.report._recent_clear else 0.0
            ),
            "solo_clear_rate": (
                float(np.mean(self.report._solo_clear)) if self.report._solo_clear else 0.0
            ),
            "solo_mean_x": (
                float(np.mean(self.report._solo_x)) if self.report._solo_x else 0.0
            ),
            "curriculum_stage": self.report.focus,
            "episodes": int(self.report.episodes),
            "steps_per_sec": float(self.report.steps_per_sec),
            "updated": time.time(),
        }
        # Persist the curriculum stage as a position so it survives the
        # curriculum being rebuilt by a later exploration pass.
        f = self.report.focus
        if f is not None and 0 <= f < len(self.positions) and self.positions[f]:
            room, x = self.positions[f]
            tmp2 = self.outdir / "curriculum_stage.tmp.json"
            tmp2.write_text(json.dumps({"room": room, "x": x, "stage": f}))
            os.replace(tmp2, self.outdir / "curriculum_stage.json")

        stmp = self.outdir / "status.tmp.json"
        stmp.write_text(json.dumps(status, indent=2))
        os.replace(stmp, self.outdir / "status.json")

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            self._publish()
        return True


class StopWhenGoodEnough(BaseCallback):
    """Halt once the level is reliably cleared from a fresh start.

    A step budget is the only stop otherwise, so a run that succeeds early
    keeps burning compute for hours proving the same thing. Judged on the
    unaided clear rate only -- curriculum episodes start next to the goal and
    would trip this immediately.
    """

    def __init__(self, report: "ProgressReport", target: float, min_episodes: int = 100):
        super().__init__()
        self.report = report
        self.target = target
        self.min_episodes = min_episodes

    def _on_step(self) -> bool:
        solo = self.report._solo_clear
        if self.target <= 0 or len(solo) < self.min_episodes:
            return True
        rate = float(np.mean(solo))
        if rate >= self.target:
            print(f"\n[{self.report.level}] unaided clear rate {rate:.0%} >= target "
                  f"{self.target:.0%} over {len(solo)} fresh starts -- stopping early.",
                  flush=True)
            return False
        return True


class ProgressReport(BaseCallback):
    """Console + TensorBoard reporting in the terms that actually matter here:
    how far into the level the agent gets, and how often it finishes."""

    def __init__(self, level: str, report_every: int = 20_000):
        super().__init__()
        self.level = level
        self.report_every = report_every
        self._next = report_every
        self._t0 = time.time()
        # On --resume num_timesteps starts at the checkpoint's value, so rate
        # has to be measured against where *this* run began, not zero.
        self._start_steps: int | None = None
        self.best_x = 0
        self.clears = 0
        self.episodes = 0
        self._recent_x: list[int] = []
        self._recent_clear: list[float] = []
        self._solo_x: list[int] = []       # fresh starts only
        self._solo_clear: list[float] = []
        self.focus: int | None = None      # current backward-curriculum stage

    def _on_training_start(self) -> None:
        self._start_steps = self.num_timesteps
        self._next = (self.num_timesteps // self.report_every + 1) * self.report_every
        self._t0 = time.time()

    @property
    def steps_per_sec(self) -> float:
        start = self.num_timesteps if self._start_steps is None else self._start_steps
        done = self.num_timesteps - start
        return done / max(1e-6, time.time() - self._t0)

    def _on_step(self) -> bool:
        if self._start_steps is None:
            self._start_steps = self.num_timesteps
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            self.episodes += 1
            x = int(info.get("max_x", 0))
            cleared = bool(info.get("level_cleared", False))
            self.best_x = max(self.best_x, x)
            self.clears += int(cleared)
            self._recent_x.append(x)
            self._recent_clear.append(float(cleared))
            self._recent_x = self._recent_x[-100:]
            self._recent_clear = self._recent_clear[-100:]
            if "curriculum_focus" in info:
                self.focus = int(info["curriculum_focus"])
            if not info.get("from_checkpoint", False):
                self._solo_x.append(x)
                self._solo_clear.append(float(cleared))
                self._solo_x = self._solo_x[-100:]
                self._solo_clear = self._solo_clear[-100:]

        if self.num_timesteps >= self._next:
            self._next += self.report_every
            fps = self.steps_per_sec
            mean_x = float(np.mean(self._recent_x)) if self._recent_x else 0.0
            clear_rate = float(np.mean(self._recent_clear)) if self._recent_clear else 0.0
            solo_x = float(np.mean(self._solo_x)) if self._solo_x else 0.0
            solo_clear = float(np.mean(self._solo_clear)) if self._solo_clear else 0.0
            self.logger.record("smw/mean_max_x", mean_x)
            self.logger.record("smw/best_x", self.best_x)
            self.logger.record("smw/clear_rate", clear_rate)
            self.logger.record("smw/solo_mean_x", solo_x)
            self.logger.record("smw/solo_clear_rate", solo_clear)
            print(
                f"[{self.level}] {self.num_timesteps:>9,} steps | "
                f"{fps:6.0f} steps/s | mean_x {mean_x:7.0f} | best_x {self.best_x:6d} | "
                f"clear {clear_rate:5.1%} | SOLO x {solo_x:7.0f} clear {solo_clear:5.1%} | "
                f"stage {self.focus if self.focus is not None else '-':>3} | "
                f"eps {self.episodes:>6,}",
                flush=True,
            )
        return True


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    # Measured on an M2 Pro: MPS runs the CNN ~5x faster than CPU here, because
    # the emulator subprocesses already saturate the CPU cores.
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--level", default="YoshiIsland1", choices=sorted(LEVELS))
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--n-envs", type=int, default=12)
    p.add_argument("--device", default="auto")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-episode-steps", type=int, default=1200)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing latest.zip with a new model; ignored with --resume")
    p.add_argument("--save-every", type=int, default=100_000,
                   help="publish the live policy + status this often (steps)")
    p.add_argument("--curriculum", action="store_true",
                   help="seed some episodes from checkpoints/<level>/curriculum.pkl")
    p.add_argument("--curriculum-ratio", type=float, default=0.35,
                   help="fraction of episodes starting from a harvested state")
    p.add_argument("--curriculum-anneal", type=int, default=0,
                   help="legacy fixed-schedule annealing; 0 (default) uses the "
                        "success-gated backward curriculum, which only moves the "
                        "start earlier once the agent can finish from where it is")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stop-at-solo-clear", type=float, default=0.85,
                   help="stop once this unaided clear rate is sustained (0 disables)")
    p.add_argument("--tag", default="",
                   help="suffix for the checkpoint dir, for A/B runs")
    p.add_argument("--grayscale", action="store_true",
                   help="legacy observation: grayscale and keep the HUD")
    p.add_argument("--keep-hud", action="store_true",
                   help="do not crop the score/timer bar out of the observation")
    p.add_argument("--curriculum-window", type=int, default=3,
                   help="how many curriculum states around the current stage to sample")
    args = p.parse_args()

    if args.steps <= 0 or args.n_envs <= 0 or args.n_steps <= 0 or args.batch_size <= 0:
        p.error("--steps, --n-envs, --n-steps, and --batch-size must be positive")
    if args.save_every <= 0 or args.max_episode_steps <= 0:
        p.error("--save-every and --max-episode-steps must be positive")
    if not 0.0 <= args.curriculum_ratio <= 1.0:
        p.error("--curriculum-ratio must be between 0 and 1")
    if args.curriculum_anneal < 0 or args.curriculum_window < 0:
        p.error("curriculum anneal/window values cannot be negative")
    if not 0.0 <= args.stop_at_solo_clear <= 1.0:
        p.error("--stop-at-solo-clear must be between 0 and 1")

    device = pick_device(args.device)
    outdir = CKPT / (args.level + args.tag)
    outdir.mkdir(parents=True, exist_ok=True)
    latest = outdir / "latest.zip"
    requested_obs = ObsConfig(color=not args.grayscale,
                              crop_top=0 if args.keep_hud else 32)

    pool = None
    curriculum_manager = None
    curriculum_positions: list = []
    venv = None
    model = None
    live_publish = None
    cb = ProgressReport(args.level)
    obs_cfg = requested_obs
    resumed = False
    publish_final = False
    try:
        if args.curriculum:
            source_dir = CKPT / args.level
            pkl = source_dir / "curriculum.pkl"
            try:
                states = load_verified_curriculum(source_dir / "archive.pkl", pkl)
            except ValueError as e:
                raise SystemExit(f"refusing unverified curriculum for {args.level}: {e}") from e

            # Restore the stage by position, not index: extending the archive
            # rebuilds the curriculum with a different length.
            resume_focus = None
            stage_file = outdir / "curriculum_stage.json"
            if stage_file.exists():
                try:
                    saved = json.loads(stage_file.read_text())
                    positions = [(r, x) for r, x, _ in states]
                    resume_focus = CheckpointPool.index_for_position(
                        positions, (saved["room"], saved["x"]))
                    if resume_focus is not None:
                        print(f"resuming curriculum at stage {resume_focus} "
                              f"(saved position room {saved['room']} x {saved['x']})")
                except (json.JSONDecodeError, KeyError, OSError, TypeError):
                    pass
            pool = CheckpointPool(ratio=args.curriculum_ratio, states=states,
                                  anneal_resets=args.curriculum_anneal,
                                  window=args.curriculum_window,
                                  initial_focus=resume_focus)
            if args.n_envs > 1:
                curriculum_manager = mp.Manager()
                pool.enable_shared(curriculum_manager)
            curriculum_positions = [(r, x) for r, x, _ in states]
            mode_name = (f"fixed anneal over {args.curriculum_anneal} resets"
                         if args.curriculum_anneal else
                         f"success-gated, starting at stage {pool.focus}/{len(states) - 1}")
            print(f"curriculum: {len(states)} states, "
                  f"{args.curriculum_ratio:.0%} of resets, {mode_name}")

        if args.resume and latest.exists():
            existing = load_policy_runtime(latest, device="cpu", expected_level=args.level)
            obs_cfg = existing.obs_config
            if obs_cfg != requested_obs:
                print(f"resume contract overrides requested defaults: {obs_cfg}, "
                      f"stack={existing.frame_stack}")
            if existing.frame_stack != 4:
                raise SystemExit(
                    f"checkpoint uses unsupported frame stack {existing.frame_stack}; expected 4"
                )
            del existing
            resumed = True
        elif latest.exists() and not args.overwrite:
            raise SystemExit(
                f"{latest} already exists; use --resume, --tag for a separate run, "
                "or explicit --overwrite"
            )

        print(f"observation: {'colour' if obs_cfg.color else 'grayscale'} "
              f"{obs_cfg.size}x{obs_cfg.size}, "
              f"HUD {'kept' if not obs_cfg.crop_top else 'cropped'}")
        venv = make_vec_env(
            args.level,
            n_envs=args.n_envs,
            seed=args.seed,
            obs_cfg=obs_cfg,
            reward_cfg=RewardConfig(),
            episode_cfg=EpisodeConfig(max_steps=args.max_episode_steps),
            checkpoints=pool,
        )

        if resumed:
            model = PPO.load(latest, env=venv, device=device)
            print(f"resuming from {latest}")
        else:
            model = PPO(
                "CnnPolicy", venv, learning_rate=args.lr, n_steps=args.n_steps,
                batch_size=args.batch_size, n_epochs=4, gamma=0.99,
                gae_lambda=0.95, clip_range=0.2, ent_coef=args.ent_coef,
                vf_coef=0.5, max_grad_norm=0.5, tensorboard_log=str(RUNS),
                device=device, verbose=0, seed=args.seed,
            )

        print(f"training {args.level} on {device} with {args.n_envs} envs "
              f"for {args.steps:,} steps")
        live_publish = LivePublish(outdir, cb, args.save_every, obs_cfg,
                                   curriculum_positions)
        callbacks = [cb, live_publish,
                     StopWhenGoodEnough(cb, args.stop_at_solo_clear)]
        print(f"publishing a live policy to {latest} every {args.save_every:,} steps -- "
              f"watch it with:\n    python -m smwrl.live --level {args.level} --brain")
        try:
            model.learn(
                total_timesteps=args.steps,
                callback=callbacks,
                reset_num_timesteps=not resumed,
                tb_log_name=args.level,
            )
            publish_final = True
        except KeyboardInterrupt:
            print("\ninterrupted -- saving before exit")
            publish_final = True
    finally:
        # Publishing may itself fail (for example a full disk).  Resource
        # cleanup must still happen, so it is nested outside the save attempt.
        try:
            if publish_final and model is not None and live_publish is not None:
                live_publish._publish(model, int(model.num_timesteps))
                print(f"saved {latest} at {model.num_timesteps:,} timesteps "
                      f"(best_x={cb.best_x}, clears={cb.clears})")
        finally:
            try:
                if venv is not None:
                    venv.close()
            finally:
                if curriculum_manager is not None:
                    curriculum_manager.shutdown()


if __name__ == "__main__":
    main()
