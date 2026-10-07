"""Deterministic evaluation with frozen normalization and explicit outcome semantics.

A course success is a healthy supported landing at the finish, not a timeout.
Legacy checkpoint evaluation uses the current environment and is labeled as such.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def make_evaluation_env(env_id, env_kwargs=None, difficulty=None, video=False,
                        max_episode_steps=None):
    kwargs = dict(env_kwargs or {})
    if video:
        kwargs.update(render_mode="rgb_array", width=640, height=480)
    else:
        kwargs.pop("render_mode", None)
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = int(max_episode_steps)
    if env_id == "ParkourHumanoid":
        from envs.parkour_env import ParkourHumanoidEnv
        if difficulty is not None:
            kwargs["difficulty"] = float(difficulty)
        # Benchmark a declared level, never a stochastic mixture of easier ones.
        kwargs["replay_prob"] = 0.0
        return ParkourHumanoidEnv(**kwargs)
    return gym.make(env_id, **kwargs)


def checkpoint_context(checkpoint):
    """Resolve an alias once, verify its pair, and recover actual environment args."""
    checkpoint = Path(checkpoint).resolve(strict=True)
    meta_path = checkpoint / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    actual_hashes = {}
    for name in ("policy.zip", "vecnormalize.pkl"):
        actual_hashes[name] = hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
        expected = meta.get("sha256", {}).get(name)
        if expected and actual_hashes[name] != expected:
            raise ValueError(f"Checkpoint integrity failure: {name}")
    config = {}
    for parent in (checkpoint, *checkpoint.parents):
        if (parent / "config.yaml").exists():
            config = yaml.safe_load((parent / "config.yaml").read_text()) or {}
            break
    environment = meta.get("environment", {})
    env_id = meta.get("env_id", environment.get("env_id", config.get("env_id", "Hopper-v5")))
    kwargs = dict(config.get("env_kwargs", {}))
    kwargs.update(environment.get("env_kwargs", {}))
    selection_path = checkpoint / "selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    evaluation = meta.get("evaluation") or selection
    difficulty = evaluation.get("curriculum_level", environment.get("difficulty",
        config.get("curriculum", {}).get("initial_level", 1.0)))
    return dict(checkpoint=checkpoint, env_id=env_id, env_kwargs=kwargs,
                difficulty=difficulty, metadata=meta, sha256=actual_hashes)


def evaluate(model, normalizer, env_id, seeds, video=None, difficulty=None,
             env_kwargs=None, max_episode_steps=None):
    seeds = [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError("Evaluation requires at least one seed")
    env = make_evaluation_env(env_id, env_kwargs, difficulty, bool(video), max_episode_steps)
    if video:
        Path(video).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(video), fps=env.metadata["render_fps"]) if video else None
    rows = []
    u = env.unwrapped
    torso = mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    feet = {name: mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("left_foot", "right_foot")}
    contract = u.get_contract() if hasattr(u, "get_contract") else {
        "version": env_id, "control_dt": u.dt,
        "observation_shape": list(env.observation_space.shape),
        "action_shape": list(env.action_space.shape),
    }
    # normalize_obs never updates statistics, even when called from training.
    mean_before = normalizer.obs_rms.mean.copy()
    count_before = float(normalizer.obs_rms.count)
    try:
        for episode_index, seed in enumerate(seeds):
            obs, reset_info = env.reset(seed=seed)
            total, terms, upright, facing = 0.0, {}, [], []
            contacts = {name: [] for name in feet}
            com_distance = 0.0
            humanoid = "Humanoid" in env_id
            start_y = float(u.data.qpos[1]) if humanoid else 0.0
            start_x = float(u.data.qpos[0])
            limit = int(max_episode_steps or getattr(u, "max_episode_steps", 0)
                        or (env.spec.max_episode_steps if env.spec else 0) or 1000)
            last_info = reset_info or {}
            terminated = truncated = False
            for step in range(limit):
                if writer and episode_index == 0:
                    writer.append_data(env.render())
                action, _ = model.predict(normalizer.normalize_obs(obs[None, :]), deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action[0])
                last_info = info
                total += float(reward)
                com_distance += float(info.get("x_velocity", 0.0)) * u.dt
                upright.append(float(u.data.xmat[torso].reshape(3, 3)[2, 2]))
                facing.append(float(u.data.xmat[torso].reshape(3, 3)[0, 0]))
                for key, value in info.items():
                    if key.startswith("reward_"):
                        terms[key] = terms.get(key, 0.0) + float(value)
                supported = set()
                for contact in u.data.contact:
                    a, b = u.model.geom_bodyid[contact.geom1], u.model.geom_bodyid[contact.geom2]
                    death = getattr(u, "_death_geom_id", -1)
                    if a == 0 and contact.geom1 != death:
                        supported.add(b)
                    if b == 0 and contact.geom2 != death:
                        supported.add(a)
                for name, body in feet.items():
                    contacts[name].append(body >= 0 and body in supported)
                if terminated or truncated:
                    break
            # A defensive evaluator cap is a timeout, never an unfinished success.
            if not terminated and not truncated:
                truncated = True
                last_info = {**last_info, "termination_reason": "evaluation_time_limit"}
            if writer and episode_index == 0:
                writer.append_data(env.render())
            success = bool(last_info.get("success", last_info.get("completed", False)))
            failed = bool(last_info.get("failed", terminated and not success))
            healthy = bool(last_info.get("is_healthy", getattr(u, "is_healthy", not failed)))
            row = dict(
                seed=seed, reward=total, length=step + 1, duration_seconds=(step + 1) * u.dt,
                distance=float(u.data.qpos[0]) - start_x, com_distance=com_distance,
                speed=com_distance / ((step + 1) * u.dt),
                lateral_distance=float(u.data.qpos[1]) - start_y if humanoid else 0.0,
                upright=float(np.mean(upright)), facing=float(np.mean(facing)),
                clearance_rate=float(last_info.get("clearance_rate", 0.0)),
                completion_rate=float(success), success=success, failed=failed,
                completed=success, is_healthy=healthy,
                progress_fraction=float(last_info.get("progress_fraction", 0.0)),
                segments_cleared=int(last_info.get("segments_cleared", 0)),
                obstacles_total=int(last_info.get("obstacles_total", 0)),
                course_length=float(last_info.get("course_length", 0.0)),
                curriculum_level=float(last_info.get("curriculum_level", difficulty or 0.0)),
                effective_difficulty=float(last_info.get("effective_difficulty", difficulty or 0.0)),
                reward_terms=terms,
                foot_contact_fraction={k: float(np.mean(v)) for k, v in contacts.items()},
                foot_touchdowns={k: int(np.count_nonzero(np.diff(np.array(v, dtype=int)) == 1))
                                 for k, v in contacts.items()},
                terminated=bool(terminated), truncated=bool(truncated),
                termination_reason=last_info.get("termination_reason") or
                    ("success" if success else "unhealthy" if failed else "time_limit"),
            )
            if hasattr(u, "_segments"):
                row["course_segments"] = [dict(kind=s.kind, x_start=s.x_start, x_end=s.x_end,
                                               cleared=i in u._cleared_segments)
                                          for i, s in enumerate(u._segments)]
            rows.append(row)
    finally:
        if writer:
            writer.close()
        env.close()
    if not np.array_equal(mean_before, normalizer.obs_rms.mean) or count_before != normalizer.obs_rms.count:
        raise RuntimeError("Evaluation modified observation normalization statistics")
    result = dict(
        schema_version=2, environment_contract=contract, episodes=rows,
        mean_reward=float(np.mean([r["reward"] for r in rows])),
        success_rate=float(np.mean([r["success"] for r in rows])),
        failure_rate=float(np.mean([r["failed"] for r in rows])),
        timeout_rate=float(np.mean([r["truncated"] for r in rows])),
        # Retained for locomotion comparisons; a successful early finish isn't a
        # full-duration survival trial. Parkour should use success_rate instead.
        survival_fraction=float(np.mean([r["truncated"] and not r["failed"] and r["is_healthy"] for r in rows])),
        mean_reward_terms={k: float(np.mean([r["reward_terms"].get(k, 0) for r in rows]))
                           for k in set().union(*(r["reward_terms"] for r in rows))},
        evaluation_protocol=dict(deterministic=True, seeds=seeds, replay_prob=0.0,
                                 env_kwargs=dict(env_kwargs or {}), difficulty=difficulty,
                                 max_episode_steps=limit, video_seed=seeds[0] if video else None),
    )
    for key in ("speed", "distance", "upright", "facing", "clearance_rate", "completion_rate",
                "progress_fraction", "curriculum_level"):
        result[key] = float(np.mean([row[key] for row in rows]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--env", help="Inferred from checkpoint metadata/config")
    parser.add_argument("--difficulty", type=float, help="Override checkpoint's evaluation level (0=flat)")
    parser.add_argument("--video", type=Path, default=Path("evaluation.mp4"))
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--output", type=Path, help="JSON output; defaults to video path with .json")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--max-episode-steps", type=int)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    context = checkpoint_context(args.checkpoint)
    env_id = args.env or context["env_id"]
    difficulty = args.difficulty if args.difficulty is not None else context["difficulty"]
    if env_id != "ParkourHumanoid":
        difficulty = None
    torch.set_num_threads(1)
    vec = DummyVecEnv([lambda: make_evaluation_env(env_id, context["env_kwargs"], difficulty,
                                                   max_episode_steps=args.max_episode_steps)])
    normalizer = VecNormalize.load(context["checkpoint"] / "vecnormalize.pkl", vec)
    normalizer.training, normalizer.norm_reward = False, False
    try:
        model = PPO.load(context["checkpoint"] / "policy.zip", device="cpu")
        result = evaluate(model, normalizer, env_id, range(args.seed, args.seed + args.episodes),
                          None if args.no_video else args.video, difficulty=difficulty,
                          env_kwargs=context["env_kwargs"], max_episode_steps=args.max_episode_steps)
        source_version = context["metadata"].get("environment", {}).get("physics", {}).get("version", "legacy-unversioned")
        result["checkpoint"] = dict(path=str(context["checkpoint"]), sha256=context["sha256"],
                                    source_env_version=source_version)
        result["environment_changed_from_training"] = source_version != result["environment_contract"]["version"]
        output = args.output or args.video.with_suffix(".json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        normalizer.close()


if __name__ == "__main__":
    main()
