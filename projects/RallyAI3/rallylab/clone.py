"""Run demonstration-only actor updates and save a PPO-compatible initializer."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import time

import yaml


def atomic_torch_save(torch, value, path: Path):
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def update_budget(steps: int, buffer_size: int) -> int:
    return max(1, math.ceil(steps / max(1, buffer_size)))


def cloning_schedule(steps: int, configured_batch_size: int) -> dict:
    batch_size = max(64, min(512, configured_batch_size))
    return {"batch_size": batch_size, "buffer_size": batch_size,
            "samples_per_update": batch_size, "epochs": 3,
            "updates": update_budget(steps, batch_size)}


def clone(config_path: Path, demonstrations: Path, steps: int, seed: int,
          output: Path, state_path: Path, initial_checkpoint: Path | None = None):
    from mlagents.plugins.trainer_type import register_trainer_plugins
    register_trainer_plugins()
    from mlagents.torch_utils import torch
    from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
    from mlagents.trainers.ppo.trainer import PPOTrainer
    from mlagents.trainers.settings import BehavioralCloningSettings, TrainerSettings
    from mlagents.trainers.demo_loader import demo_to_buffer

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    torch.manual_seed(seed)
    config = yaml.safe_load(config_path.read_text())
    raw = config["behaviors"]["RallyDriver"]
    settings = TrainerSettings.structure(raw, TrainerSettings)
    schedule = cloning_schedule(steps, settings.hyperparameters.batch_size)
    settings.hyperparameters.batch_size = schedule["batch_size"]
    settings.hyperparameters.buffer_size = schedule["buffer_size"]
    settings.init_path = str(initial_checkpoint) if initial_checkpoint else None
    settings.behavioral_cloning = BehavioralCloningSettings(
        demo_path=str(demonstrations), steps=steps, strength=1.0,
        samples_per_update=schedule["samples_per_update"],
        batch_size=schedule["batch_size"], num_epoch=schedule["epochs"]
    )

    behavior_spec, _ = demo_to_buffer(str(demonstrations), sequence_length=1)
    identifier = BehaviorIdentifiers("RallyDriver?team=0", "RallyDriver", 0)
    trainer = PPOTrainer("RallyDriver", 1, settings, True, False, seed,
                         str(output / "RallyDriver"))
    policy = trainer.create_policy(identifier, behavior_spec)
    trainer.add_policy(identifier, policy)
    bc_module = trainer.optimizer.bc_module
    buffer_size = schedule["buffer_size"]
    target_updates = schedule["updates"]

    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("schedule") != schedule or state.get("steps") != steps or state.get("seed") != seed:
            raise ValueError("Interrupted cloning state does not match the current run configuration")
        policy.actor.load_state_dict(state["actor"])
        bc_module.optimizer.load_state_dict(state["bc_optimizer"])
        completed_updates = int(state["completed_updates"])
    else:
        policy.actor.update_normalization(bc_module.demonstration_buffer)
        completed_updates = 0

    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def save_progress():
        policy.set_step(min(steps, completed_updates * buffer_size))
        atomic_torch_save(torch, {
            "actor": policy.actor.state_dict(),
            "bc_optimizer": bc_module.optimizer.state_dict(),
            "completed_updates": completed_updates,
            "seed": seed,
            "steps": steps,
            "schedule": schedule,
        }, state_path)
        progress = {"schema": 1, "phase": "cloning", "completed_updates": completed_updates,
                    "target_updates": target_updates, "completed_steps": min(steps, completed_updates * buffer_size),
                    "target_steps": steps, "batch_size": schedule["batch_size"],
                    "samples_per_update": schedule["samples_per_update"],
                    "epochs": schedule["epochs"], "updated": time.time()}
        state_path.with_suffix(".json").write_text(json.dumps(progress, indent=2))

    save_progress()
    while completed_updates < target_updates and not stop_requested:
        policy.set_step(min(steps, completed_updates * buffer_size))
        metrics = bc_module.update()
        completed_updates += 1
        save_progress()
        print(f"Cloning update {completed_updates}/{target_updates}: "
              f"loss={float(metrics['Losses/Pretraining Loss']):.6f}", flush=True)

    if stop_requested:
        print("Cloning paused after saving progress.", flush=True)
        return 130

    policy.set_step(steps)
    trainer.model_saver.save_checkpoint("RallyDriver", steps)
    save_progress()
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--demonstrations", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raise SystemExit(clone(args.config, args.demonstrations, args.steps, args.seed,
                           args.output, args.state, args.initial_checkpoint))


if __name__ == "__main__":
    main()
